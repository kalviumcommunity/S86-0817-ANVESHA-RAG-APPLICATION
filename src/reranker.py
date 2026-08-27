"""Re-rank retrieved candidates for more precise final context."""

import argparse
import re
from typing import Any, Callable


def rerank_score(query: str, chunk: dict[str, Any]) -> float:
    """Score query/chunk lexical overlap as a simple transparent reranker."""
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    chunk_terms = set(re.findall(r"[a-z0-9]+", chunk["text"].lower()))
    if not query_terms:
        return 0.0
    return len(query_terms & chunk_terms) / len(query_terms)


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    final_k: int = 3,
    scorer: Callable[[str, dict[str, Any]], float] = rerank_score,
) -> list[dict[str, Any]]:
    """Apply a second score to candidates and return the final top-k."""
    if final_k < 1:
        raise ValueError("final_k must be greater than zero")
    scored = [{**candidate, "rerank_score": scorer(query, candidate)} for candidate in candidates]
    scored.sort(key=lambda candidate: candidate["rerank_score"], reverse=True)
    return scored[:final_k]


def compare_ordering(
    query: str,
    candidates: list[dict[str, Any]],
    final_k: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the initial candidate order and the reranked final context."""
    return candidates[:final_k], rerank(query, candidates, final_k)


def build_demo_candidates() -> list[dict[str, Any]]:
    """Return candidates where reranking improves the initial vector order."""
    return [
        {
            "score": 0.95,
            "text": "General project submission information.",
            "metadata": {"source": "general-guide.md", "chunk_index": 2},
        },
        {
            "score": 0.86,
            "text": "Evidence required for project submission includes screenshots and a report.",
            "metadata": {"source": "submission-rubric.md", "chunk_index": 1},
        },
        {
            "score": 0.81,
            "text": "Project deadlines are listed in the academic calendar.",
            "metadata": {"source": "calendar.md", "chunk_index": 4},
        },
    ]


def show(label: str, rows: list[dict[str, Any]]) -> None:
    print(label)
    for rank, item in enumerate(rows, start=1):
        print(
            f"rank={rank} vector_score={item['score']:.4f} "
            f"rerank_score={item.get('rerank_score', 'n/a')} "
            f"source={item['metadata']['source']} text={item['text'][:120]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-k", type=int, default=2)
    args = parser.parse_args()
    query = "What evidence is required for project submission?"
    candidates = build_demo_candidates()
    initial, final = compare_ordering(query, candidates, args.final_k)
    show("before re-ranking", initial)
    show("after re-ranking", final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())