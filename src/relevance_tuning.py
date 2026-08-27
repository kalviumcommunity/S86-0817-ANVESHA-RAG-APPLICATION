"""Compare retrieval settings against known query/source expectations."""

import argparse
from dataclasses import asdict, dataclass
from typing import Any, Callable


TEST_QUERIES = [
    {"query": "How can a learner reset their password?", "expected_source": "account-guide.md"},
    {"query": "When does the cafeteria menu change?", "expected_source": "campus-guide.md"},
    {"query": "What evidence is required for project submission?", "expected_source": "submission-rubric.md"},
]


@dataclass(frozen=True)
class RetrievalSetting:
    name: str
    k: int
    metadata_filter: dict[str, Any] | None = None
    min_score: float = 0.0


@dataclass(frozen=True)
class EvaluationRow:
    query: str
    expected_source: str
    returned_sources: list[str]
    hit: bool


@dataclass(frozen=True)
class EvaluationSummary:
    setting: str
    hit_rate: float
    details: list[EvaluationRow]


def evaluate_setting(
    setting: RetrievalSetting,
    retrieve: Callable[[str, int, dict[str, Any] | None], list[dict[str, Any]]],
    test_queries: list[dict[str, str]] = TEST_QUERIES,
) -> EvaluationSummary:
    """Evaluate one retrieval configuration using expected source hit rate."""
    if setting.k < 1:
        raise ValueError("k must be greater than zero")
    rows = []
    for test_query in test_queries:
        results = retrieve(test_query["query"], setting.k, setting.metadata_filter)
        kept = [result for result in results if result["score"] >= setting.min_score]
        sources = [result["metadata"]["source"] for result in kept]
        rows.append(
            EvaluationRow(
                query=test_query["query"],
                expected_source=test_query["expected_source"],
                returned_sources=sources,
                hit=test_query["expected_source"] in sources,
            )
        )
    hits = sum(row.hit for row in rows)
    return EvaluationSummary(setting.name, hits / len(rows) if rows else 0.0, rows)


def choose_best(summary: list[EvaluationSummary]) -> EvaluationSummary:
    """Choose the highest-hit-rate setting, preserving declaration order on ties."""
    if not summary:
        raise ValueError("at least one evaluation summary is required")
    return max(summary, key=lambda result: result.hit_rate)


def demo_retrieve(query: str, k: int, metadata_filter: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return deterministic retrieval results for the tuning demonstration."""
    records = {
        "password": {"source": "account-guide.md", "doc_type": "guide"},
        "cafeteria": {"source": "campus-guide.md", "doc_type": "guide"},
        "submission": {"source": "submission-rubric.md", "doc_type": "rubric"},
    }
    if "password" in query.lower():
        ordered = [("password", 0.91), ("cafeteria", 0.72), ("submission", 0.31)]
    elif "cafeteria" in query.lower():
        ordered = [("cafeteria", 0.89), ("password", 0.70), ("submission", 0.34)]
    else:
        ordered = [("submission", 0.88), ("password", 0.68), ("cafeteria", 0.32)]
    results = []
    for key, score in ordered:
        metadata = records[key]
        if metadata_filter and any(metadata.get(name) != value for name, value in metadata_filter.items()):
            continue
        results.append({"score": score, "metadata": metadata, "text": key})
    return results[:k]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()
    settings = [
        RetrievalSetting("baseline_k3", k=3),
        RetrievalSetting("filtered_k3", k=3, metadata_filter={"doc_type": "guide"}),
        RetrievalSetting("strict_k3", k=3, min_score=args.min_score),
    ]
    summaries = [evaluate_setting(setting, demo_retrieve) for setting in settings]
    for summary in summaries:
        print(f"{summary.setting} hit_rate: {summary.hit_rate:.2f}")
        for row in summary.details:
            print(asdict(row))
    best = choose_best(summaries)
    print(f"best_setting: {best.setting} hit_rate: {best.hit_rate:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())