"""Rank embedded chunks by cosine similarity to a query vector."""

import argparse
import math
from typing import Any


def cosine_similarity(first: list[float], second: list[float]) -> float:
    """Return cosine similarity, where higher scores are more similar."""
    if len(first) != len(second) or not first:
        raise ValueError("vectors must be non-empty and have equal dimensions")
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return sum(a * b for a, b in zip(first, second)) / (first_norm * second_norm)


def rank_chunks(
    query_embedding: list[float],
    chunk_records: list[dict[str, Any]],
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Score and rank chunk records without discarding source metadata."""
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be greater than zero")
    ranked = [
        {
            **record,
            "score": cosine_similarity(query_embedding, record["embedding"]),
        }
        for record in chunk_records
    ]
    ranked.sort(key=lambda record: record["score"], reverse=True)
    return ranked[:top_k] if top_k is not None else ranked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    query = "How can a learner reset their password?"
    query_embedding = [1.0, 0.0, 0.0]
    records = [
        {
            "text": "Password reset instructions for learner accounts.",
            "metadata": {"source": "account-guide.md", "chunk_index": 0},
            "embedding": [0.99, 0.1, 0.0],
        },
        {
            "text": "The cafeteria menu changes every Friday.",
            "metadata": {"source": "campus-guide.md", "chunk_index": 3},
            "embedding": [0.0, 0.1, 0.99],
        },
        {
            "text": "Learners can recover access using their registered email.",
            "metadata": {"source": "account-guide.md", "chunk_index": 1},
            "embedding": [0.95, 0.2, 0.0],
        },
    ]
    print(f"query: {query}")
    for result in rank_chunks(query_embedding, records, args.top_k):
        print(f"score={result['score']:.4f} | {result['text']} | {result['metadata']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())