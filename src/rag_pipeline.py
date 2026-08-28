"""A small, testable retrieval-augmented generation pipeline."""

import argparse
from dataclasses import dataclass
from typing import Any, Callable


FALLBACK_ANSWER = "I could not find relevant context for that question."


@dataclass(frozen=True)
class PipelineResult:
    """The user-facing answer and metadata for its evidence."""

    answer: str
    sources: list[dict[str, Any]]


def embed_query(query: str, embed: Callable[[str], list[float]]) -> list[float]:
    """Create one embedding vector for a user query."""
    return embed(query)


def retrieve_context(
    query_vector: list[float],
    retrieve: Callable[[list[float], int], list[dict[str, Any]]],
    k: int = 4,
) -> list[dict[str, Any]]:
    """Retrieve up to k evidence chunks from the vector store."""
    if k < 1:
        raise ValueError("k must be greater than zero")
    return retrieve(query_vector, k)


def assemble_context(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks with numbered source citations."""
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        source = chunk["metadata"]["source"]
        parts.append(f"[{index}] Source: {source}\n{chunk['text']}")
    return "\n\n".join(parts)


def generate_answer(
    query: str,
    context: str,
    generate: Callable[[str, str], str],
) -> str:
    """Generate an answer from only the assembled retrieved context."""
    return generate(query, context)


def answer_query(
    query: str,
    embed: Callable[[str], list[float]],
    retrieve: Callable[[list[float], int], list[dict[str, Any]]],
    generate: Callable[[str, str], str],
    k: int = 4,
) -> PipelineResult:
    """Run embedding, retrieval, context assembly, and grounded generation."""
    query_vector = embed_query(query, embed)
    chunks = retrieve_context(query_vector, retrieve, k)
    if not chunks:
        return PipelineResult(FALLBACK_ANSWER, [])
    context = assemble_context(chunks)
    answer = generate_answer(query, context, generate)
    sources = [chunk["metadata"] for chunk in chunks]
    return PipelineResult(answer, sources)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="What evidence is required for project submission?")
    args = parser.parse_args()
    chunks = [
        {
            "text": "Project submissions require screenshots and a written report.",
            "metadata": {"source": "submission-rubric.md", "chunk_index": 2},
        }
    ]
    result = answer_query(
        args.query,
        embed=lambda query: [1.0, 0.0],
        retrieve=lambda vector, k: chunks[:k],
        generate=lambda query, context: f"Based on the evidence: {context.splitlines()[-1]}",
    )
    print(result.answer)
    print(f"sources: {result.sources}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())