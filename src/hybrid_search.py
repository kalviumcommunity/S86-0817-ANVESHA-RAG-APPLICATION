"""Combine metadata filtering, semantic retrieval, and keyword matching."""

import argparse
from typing import Any, Callable

import chromadb


def keyword_score(text: str, keywords: list[str]) -> int:
    """Count supplied keywords that occur in a chunk, case-insensitively."""
    lowered = text.lower()
    return sum(keyword.lower() in lowered for keyword in keywords)


def hybrid_rank(
    vector_results: list[dict[str, Any]],
    keywords: list[str],
    vector_weight: float = 0.8,
    keyword_weight: float = 0.2,
) -> list[dict[str, Any]]:
    """Add lexical scores and sort results by weighted hybrid score."""
    if vector_weight < 0 or keyword_weight < 0 or vector_weight + keyword_weight <= 0:
        raise ValueError("weights must be non-negative and have a positive total")
    ranked = []
    for item in vector_results:
        lexical = keyword_score(item["text"], keywords)
        ranked.append(
            {
                **item,
                "keyword_score": lexical,
                "hybrid_score": vector_weight * item["score"] + keyword_weight * lexical,
            }
        )
    return sorted(ranked, key=lambda item: item["hybrid_score"], reverse=True)


class FilteredRetriever:
    """Retrieve semantic matches with optional metadata filtering."""

    def __init__(self, collection: Any, embed_query: Callable[[str], list[float]]) -> None:
        self.collection = collection
        self.embed_query = embed_query

    def retrieve(
        self,
        query: str,
        k: int = 3,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return scored text and metadata from an optional filtered subset."""
        if k < 1:
            raise ValueError("k must be greater than zero")
        kwargs = {
            "query_embeddings": [self.embed_query(query)],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if metadata_filter:
            kwargs["where"] = metadata_filter
        result = self.collection.query(**kwargs)
        return [
            {"score": 1.0 - distance, "text": text, "metadata": metadata}
            for text, metadata, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]


def build_demo_retriever() -> FilteredRetriever:
    """Build a deterministic collection for filtered and hybrid retrieval."""
    collection = chromadb.EphemeralClient().get_or_create_collection(
        name="hybrid_demo",
        configuration={"hnsw": {"space": "cosine"}},
    )
    collection.upsert(
        ids=["account:0", "security:1", "campus:3"],
        embeddings=[[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
        documents=[
            "Password reset instructions for learner accounts.",
            "Security policy covers account access reviews.",
            "The cafeteria menu changes every Friday.",
        ],
        metadatas=[
            {"source": "account-guide.md", "section": "Account access"},
            {"source": "security-policy.md", "section": "Security"},
            {"source": "campus-guide.md", "section": "Campus"},
        ],
    )
    return FilteredRetriever(collection, lambda query: [1.0, 0.0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="What are the password reset steps?")
    args = parser.parse_args()
    retriever = build_demo_retriever()
    unfiltered = retriever.retrieve(args.query, k=3)
    filtered = retriever.retrieve(args.query, k=3, metadata_filter={"section": "Account access"})
    hybrid = hybrid_rank(filtered, ["password", "reset"])
    for label, results in (("unfiltered", unfiltered), ("filtered", filtered), ("hybrid filtered", hybrid)):
        print(label)
        for result in results:
            print(f"score: {result['score']:.4f} source: {result['metadata']['source']} text: {result['text'][:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())