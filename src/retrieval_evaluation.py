"""Measure recall and precision for labelled retrieval queries."""

import argparse
from dataclasses import asdict, dataclass
from typing import Any, Callable


LABELLED_QUERIES = [
    {
        "query": "How can a learner reset their password?",
        "relevant_chunk_ids": {"account-guide.md:0", "account-guide.md:1"},
    },
    {
        "query": "What evidence is required for project submission?",
        "relevant_chunk_ids": {"submission-rubric.md:2"},
    },
]


@dataclass(frozen=True)
class EvaluationRow:
    query: str
    retrieved_ids: list[str]
    relevant_chunk_ids: list[str]
    hits: list[str]
    recall: float
    precision: float


@dataclass(frozen=True)
class EvaluationReport:
    rows: list[EvaluationRow]
    average_recall: float
    average_precision: float

    @property
    def failures(self) -> list[EvaluationRow]:
        return [row for row in self.rows if row.recall < 1.0]


def evaluate_query(
    item: dict[str, Any],
    retrieve: Callable[[str, int], list[dict[str, Any]]],
    k: int = 5,
) -> EvaluationRow:
    """Evaluate one labelled query at top-k."""
    results = retrieve(item["query"], k)
    retrieved_ids = [result["id"] for result in results]
    relevant = set(item["relevant_chunk_ids"])
    hits = [chunk_id for chunk_id in retrieved_ids if chunk_id in relevant]
    return EvaluationRow(
        query=item["query"],
        retrieved_ids=retrieved_ids,
        relevant_chunk_ids=sorted(relevant),
        hits=hits,
        recall=len(set(hits)) / len(relevant) if relevant else 0.0,
        precision=len(hits) / len(retrieved_ids) if retrieved_ids else 0.0,
    )


def evaluate(
    retrieve: Callable[[str, int], list[dict[str, Any]]],
    k: int = 5,
    labelled_queries: list[dict[str, Any]] = LABELLED_QUERIES,
) -> EvaluationReport:
    """Evaluate all labelled queries and aggregate recall and precision."""
    rows = [evaluate_query(item, retrieve, k) for item in labelled_queries]
    return EvaluationReport(
        rows=rows,
        average_recall=sum(row.recall for row in rows) / len(rows) if rows else 0.0,
        average_precision=sum(row.precision for row in rows) / len(rows) if rows else 0.0,
    )


def demo_retrieve(query: str, k: int) -> list[dict[str, Any]]:
    """Return deterministic ranked records for the evaluation demonstration."""
    if "password" in query.lower():
        ids = ["account-guide.md:0", "account-guide.md:1", "campus-guide.md:3"]
    else:
        ids = ["submission-rubric.md:2", "account-guide.md:0", "campus-guide.md:3"]
    return [{"id": chunk_id} for chunk_id in ids[:k]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    report = evaluate(demo_retrieve, args.k)
    print(f"queries: {len(report.rows)}")
    print(f"recall@{args.k}: {report.average_recall:.3f}")
    print(f"precision@{args.k}: {report.average_precision:.3f}")
    for row in report.rows:
        print(asdict(row))
    for failure in report.failures:
        print(f"failed query: {failure.query}")
        print(f"expected: {failure.relevant_chunk_ids}")
        print(f"retrieved: {failure.retrieved_ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())