"""Run small sanity tests for embedding-based retrieval."""

import argparse
from dataclasses import asdict, dataclass
from typing import Any, Callable

from similarity_search import rank_chunks


TEST_CASES = [
    {
        "query": "How can a learner reset their password?",
        "expected_source": "account-guide.md",
    },
    {
        "query": "When does the cafeteria menu change?",
        "expected_source": "campus-guide.md",
    },
]


@dataclass(frozen=True)
class SanityResult:
    query: str
    expected_source: str
    top_source: str | None
    top_score: float | None
    passed: bool
    risk: str | None = None


def run_sanity_tests(
    chunk_records: list[dict[str, Any]],
    embed_query: Callable[[str], list[float]],
    test_cases: list[dict[str, str]] = TEST_CASES,
) -> list[SanityResult]:
    """Check whether known query/source pairs rank as expected."""
    results = []
    for case in test_cases:
        ranked = rank_chunks(embed_query(case["query"]), chunk_records)
        top = ranked[0] if ranked else None
        top_source = top["metadata"].get("source") if top else None
        passed = top_source == case["expected_source"]
        results.append(
            SanityResult(
                query=case["query"],
                expected_source=case["expected_source"],
                top_source=top_source,
                top_score=round(top["score"], 4) if top else None,
                passed=passed,
                risk=None if passed else "expected source did not rank first",
            )
        )
    return results


def build_demo_records() -> list[dict[str, Any]]:
    """Return deterministic vectors for the two known test cases."""
    return [
        {
            "text": "Password reset instructions.",
            "metadata": {"source": "account-guide.md", "chunk_index": 0, "model": "demo"},
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "text": "The cafeteria menu changes every Friday.",
            "metadata": {"source": "campus-guide.md", "chunk_index": 0, "model": "demo"},
            "embedding": [0.0, 1.0, 0.0],
        },
        {
            "text": "Generic campus information.",
            "metadata": {"source": "generic.md", "chunk_index": 0, "model": "demo"},
            "embedding": [0.1, 0.1, 0.0],
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-risk", action="store_true")
    args = parser.parse_args()

    records = build_demo_records()

    def embed_query(query: str) -> list[float]:
        return [1.0, 0.0, 0.0] if "password" in query else [0.0, 1.0, 0.0]

    report = run_sanity_tests(records, embed_query)
    passed = sum(result.passed for result in report)
    print(f"sanity report: tests={len(report)} passed={passed} failed={len(report) - passed}")
    for result in report:
        print(asdict(result))
    if args.show_risk:
        print("risk: rankings are only trustworthy when query and document vectors use the same model.")
    return 0 if passed == len(report) else 1


if __name__ == "__main__":
    raise SystemExit(main())