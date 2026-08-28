"""Refuse generation when retrieved context is too weak to trust."""

import argparse
from dataclasses import dataclass
from typing import Any, Callable


REFUSAL_ANSWER = "I don't have enough reliable context to answer that."


@dataclass(frozen=True)
class GuardedResult:
    """The answer or refusal status returned by the guardrail."""

    answer: str
    sources: list[dict[str, Any]]
    status: str


def retrieval_is_strong(
    chunks: list[dict[str, Any]],
    min_top_score: float = 0.72,
    min_supporting_chunks: int = 1,
) -> bool:
    """Check whether enough retrieved chunks meet the relevance threshold."""
    if min_supporting_chunks < 1:
        raise ValueError("min_supporting_chunks must be greater than zero")
    if not chunks:
        return False
    strong_chunks = [chunk for chunk in chunks if chunk.get("score", 0.0) >= min_top_score]
    return len(strong_chunks) >= min_supporting_chunks


def guarded_answer(
    question: str,
    retrieve: Callable[[str], list[dict[str, Any]]],
    generate: Callable[[str, list[dict[str, Any]]], str],
    min_top_score: float = 0.72,
    min_supporting_chunks: int = 1,
) -> GuardedResult:
    """Block generation unless retrieval provides reliable supporting evidence."""
    chunks = retrieve(question)
    if not retrieval_is_strong(chunks, min_top_score, min_supporting_chunks):
        return GuardedResult(REFUSAL_ANSWER, [], "refused_weak_context")
    answer = generate(question, chunks)
    return GuardedResult(answer, [chunk["metadata"] for chunk in chunks], "answered")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-score", type=float, default=0.72)
    args = parser.parse_args()
    chunks = [{"score": 0.91, "text": "Screenshots and a report are required.", "metadata": {"source": "rubric.md"}}]
    retrieve = lambda question: chunks if "evidence" in question.lower() else [{"score": 0.2, "text": "unrelated", "metadata": {"source": "other.md"}}]
    generate = lambda question, evidence: "Evidence requires screenshots and a report."
    for question in ("What evidence is required?", "What is not in this corpus?"):
        result = guarded_answer(question, retrieve, generate, min_top_score=args.min_score)
        print(f"{result.status}: {result.answer} | sources={result.sources}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())