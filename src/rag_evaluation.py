"""Score RAG answer quality across correctness, grounding, and citation accuracy."""

import argparse
from dataclasses import dataclass
from typing import Any, Callable


DEFAULT_TEST_SET = [
    {
        "question": "What evidence is required for project submission?",
        "expected_points": ["PR link", "sample output", "video explanation"],
        "expected_sources": {"submission-rubric.md"},
    },
    {
        "question": "What should the system do when context is missing?",
        "expected_points": ["refuse", "not enough information"],
        "expected_sources": {"guardrails.md"},
    },
]


@dataclass(frozen=True)
class AnswerScore:
    """Quality scores for a single answer across three dimensions."""

    question: str
    answer: str
    correctness: float
    grounding: float
    citation_accuracy: float
    citations: dict[str, Any]


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregated quality metrics and failure details."""

    total_questions: int
    avg_correctness: float
    avg_grounding: float
    avg_citation_accuracy: float
    failures: list[AnswerScore]


def judge_expected_points(answer: str, expected_points: list[str]) -> float:
    """Score correctness as the fraction of expected points present in the answer."""
    if not expected_points:
        return 1.0
    lowered = answer.lower()
    found = sum(1 for point in expected_points if point.lower() in lowered)
    return found / len(expected_points)


def judge_grounding(answer: str, retrieved_sources: dict[str, dict[str, Any]]) -> float:
    """Score grounding as 1.0 if citations exist and 0.0 if none."""
    return 1.0 if retrieved_sources else 0.0


def check_citations(
    citations: dict[str, dict[str, Any]],
    expected_sources: set[str],
) -> float:
    """Score citation accuracy as the fraction of retrieved sources that are expected."""
    if not citations and not expected_sources:
        return 1.0
    if not citations or not expected_sources:
        return 0.0
    retrieved_sources = {c["source"] for c in citations.values()}
    matches = len(retrieved_sources & expected_sources)
    return matches / len(expected_sources) if expected_sources else 1.0


def score_answer(
    example: dict[str, Any],
    answer_generator: Callable[[str], tuple[str, dict[str, dict[str, Any]]]],
) -> AnswerScore:
    """Score a single answer on correctness, grounding, and citation accuracy."""
    answer, citations = answer_generator(example["question"])
    return AnswerScore(
        question=example["question"],
        answer=answer,
        correctness=judge_expected_points(answer, example["expected_points"]),
        grounding=judge_grounding(answer, citations),
        citation_accuracy=check_citations(citations, example["expected_sources"]),
        citations=citations,
    )


def evaluate_rag(
    test_set: list[dict[str, Any]],
    answer_generator: Callable[[str], tuple[str, dict[str, dict[str, Any]]]],
) -> EvaluationSummary:
    """Evaluate all test questions and summarize results."""
    rows = [score_answer(example, answer_generator) for example in test_set]
    failures = [row for row in rows if min(row.correctness, row.grounding, row.citation_accuracy) < 1.0]
    return EvaluationSummary(
        total_questions=len(rows),
        avg_correctness=sum(r.correctness for r in rows) / len(rows) if rows else 0.0,
        avg_grounding=sum(r.grounding for r in rows) / len(rows) if rows else 0.0,
        avg_citation_accuracy=sum(r.citation_accuracy for r in rows) / len(rows) if rows else 0.0,
        failures=failures,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    def demo_generator(question: str) -> tuple[str, dict[str, dict[str, Any]]]:
        if "evidence" in question.lower():
            return (
                "PR link, sample output, and video explanation are required.",
                {
                    "[1]": {"source": "submission-rubric.md", "text": "Evidence required..."}
                },
            )
        else:
            return ("I don't have enough information.", {})
    summary = evaluate_rag(DEFAULT_TEST_SET, demo_generator)
    print(f"total_questions: {summary.total_questions}")
    print(f"avg_correctness: {summary.avg_correctness:.2f}")
    print(f"avg_grounding: {summary.avg_grounding:.2f}")
    print(f"avg_citation_accuracy: {summary.avg_citation_accuracy:.2f}")
    print(f"failures: {len(summary.failures)}")
    for failure in summary.failures:
        print(f"  {failure.question}: {failure.answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
