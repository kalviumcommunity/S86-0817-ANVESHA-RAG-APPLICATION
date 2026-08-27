"""Embed queries and retrieve top-k chunks from a vector collection."""

import argparse
from dataclasses import dataclass
from typing import Any, Callable

import chromadb


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved chunk with its similarity score and citation metadata."""

    score: float
    text: str
    metadata: dict[str, Any]


class Retriever:
    """Run query embedding and cosine retrieval against a Chroma collection."""

    def __init__(
        self,
        collection: Any,
        embed_query: Callable[[str], list[float]],
        embedding_model: str,
    ) -> None:
        self.collection = collection
        self.embed_query = embed_query
        self.embedding_model = embedding_model

    def retrieve(self, query: str, k: int = 3) -> list[RetrievalResult]:
        """Return up to k highest-scoring chunks for a query."""
        if k < 1:
            raise ValueError("k must be greater than zero")
        query_vector = self.embed_query(query)
        response = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        documents = response["documents"][0]
        metadatas = response["metadatas"][0]
        distances = response["distances"][0]
        return [
            RetrievalResult(
                score=1.0 - distance,
                text=text,
                metadata={**metadata, "embedding_model": self.embedding_model},
            )
            for text, metadata, distance in zip(documents, metadatas, distances)
        ]


def build_demo_retriever() -> Retriever:
    """Build an in-memory collection for a repeatable retrieval demonstration."""
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name="retrieval_demo",
        configuration={"hnsw": {"space": "cosine"}},
    )
    collection.upsert(
        ids=["account:0", "account:1", "campus:3", "generic:0", "policy:2"],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.4, 0.4, 0.0],
            [0.3, 0.2, 0.0],
        ],
        documents=[
            "Password reset instructions for learner accounts.",
            "Learners recover access using their registered email.",
            "The cafeteria menu changes every Friday.",
            "General campus information.",
            "Policy documents are reviewed annually.",
        ],
        metadatas=[
            {"source": "account-guide.md", "chunk_index": 0},
            {"source": "account-guide.md", "chunk_index": 1},
            {"source": "campus-guide.md", "chunk_index": 3},
            {"source": "generic.md", "chunk_index": 0},
            {"source": "policy.md", "chunk_index": 2},
        ],
    )
    return Retriever(collection, lambda query: [1.0, 0.0, 0.0], "demo-model")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="How can a learner reset their password?")
    args = parser.parse_args()
    retriever = build_demo_retriever()
    for k in (1, 3, 5):
        print(f"k = {k}")
        for rank, result in enumerate(retriever.retrieve(args.query, k), start=1):
            print(
                f"rank={rank} score={result.score:.4f} "
                f"source={result.metadata['source']} text={result.text[:100]}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())