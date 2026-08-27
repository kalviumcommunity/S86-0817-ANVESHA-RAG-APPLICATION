"""Index embedded chunks in a vector collection and verify integrity."""

import argparse
from dataclasses import dataclass
from typing import Any

from vector_store import VectorCollection, VectorRecord


@dataclass(frozen=True)
class IndexReport:
    """Results of one batch indexing run."""

    expected_count: int
    inserted_count: int
    indexed_count: int
    failures: list[dict[str, str]]

    def validate(self) -> None:
        """Reject incomplete indexing rather than allowing silent omissions."""
        if self.indexed_count != self.expected_count:
            raise RuntimeError(
                f"indexed count {self.indexed_count} does not match "
                f"expected count {self.expected_count}"
            )


def to_vector_record(chunk: dict[str, Any]) -> VectorRecord:
    """Convert an embedded chunk to the vector store's stable record shape."""
    return VectorRecord(
        record_id=chunk["id"],
        vector=chunk["embedding"],
        text=chunk["text"],
        metadata={
            "source": chunk["metadata"]["source"],
            "chunk_index": chunk["metadata"]["chunk_index"],
            "section": chunk["metadata"].get("section"),
        },
    )


def index_embeddings(
    collection: VectorCollection,
    embedded_chunks: list[dict[str, Any]],
    batch_size: int = 100,
) -> IndexReport:
    """Upsert embedded chunks in batches and verify the final collection count."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    records = [to_vector_record(chunk) for chunk in embedded_chunks]
    inserted = 0
    failures = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        try:
            collection.upsert(batch)
            inserted += len(batch)
        except Exception as error:
            failures.append({"batch_start_id": batch[0].record_id, "error": str(error)})
    report = IndexReport(
        expected_count=len(records),
        inserted_count=inserted,
        indexed_count=collection.collection.count(),
        failures=failures,
    )
    report.validate()
    return report


def spot_check(collection: VectorCollection, chunk: dict[str, Any]) -> None:
    """Confirm a stored record still matches its original chunk."""
    stored = collection.get(chunk["id"])
    assert stored.text == chunk["text"], "stored text does not match source chunk"
    assert stored.metadata["source"] == chunk["metadata"]["source"]
    assert len(stored.vector) == len(chunk["embedding"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    chunks = [
        {
            "id": "account-guide.md:0",
            "text": "Password reset instructions for learner accounts.",
            "metadata": {"source": "account-guide.md", "chunk_index": 0, "section": "Account access"},
            "embedding": [0.1, 0.2, 0.3],
        }
    ]
    collection = VectorCollection(name="indexed_rag_chunks", dimension=3)
    report = index_embeddings(collection, chunks, args.batch_size)
    spot_check(collection, chunks[0])
    print(f"expected chunks: {report.expected_count}")
    print(f"inserted this run: {report.inserted_count}")
    print(f"indexed count: {report.indexed_count}")
    print(f"failures: {report.failures}")
    print(f"spot check passed: {chunks[0]['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())