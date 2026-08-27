"""Create a vector collection and verify stored records can be read back."""

import argparse
from dataclasses import dataclass
from typing import Any

import chromadb


COLLECTION_NAME = "rag_chunks"
VECTOR_DIMENSION = 3


@dataclass(frozen=True)
class VectorRecord:
    """Vector data and the source information needed for grounded answers."""

    record_id: str
    vector: list[float]
    text: str
    metadata: dict[str, Any]


class VectorCollection:
    """Small collection wrapper that enforces the application vector schema."""

    def __init__(self, name: str = COLLECTION_NAME, dimension: int = VECTOR_DIMENSION) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")
        self.dimension = dimension
        self.collection = chromadb.EphemeralClient().get_or_create_collection(
            name=name,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def upsert(self, records: list[VectorRecord]) -> None:
        """Insert or update records after checking vector dimensions."""
        for record in records:
            if len(record.vector) != self.dimension:
                raise ValueError(
                    f"{record.record_id} has dimension {len(record.vector)}; "
                    f"expected {self.dimension}"
                )
        self.collection.upsert(
            ids=[record.record_id for record in records],
            embeddings=[record.vector for record in records],
            documents=[record.text for record in records],
            metadatas=[record.metadata for record in records],
        )

    def get(self, record_id: str) -> VectorRecord:
        """Read one record back with vector, text, and metadata."""
        result = self.collection.get(ids=[record_id], include=["embeddings", "documents", "metadatas"])
        if not result["ids"]:
            raise KeyError(f"record not found: {record_id}")
        return VectorRecord(
            record_id=result["ids"][0],
            vector=[float(value) for value in result["embeddings"][0]],
            text=result["documents"][0],
            metadata=result["metadatas"][0],
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=VECTOR_DIMENSION)
    args = parser.parse_args()
    collection = VectorCollection(dimension=args.dimension)
    record = VectorRecord(
        record_id="account-guide.md:0",
        vector=[0.1] * args.dimension,
        text="Password reset instructions for learner accounts.",
        metadata={"source": "account-guide.md", "chunk_index": 0, "section": "Account access"},
    )
    collection.upsert([record])
    stored = collection.get(record.record_id)
    print(f"readback id: {stored.record_id}")
    print(f"vector length: {len(stored.vector)}")
    print(f"text: {stored.text}")
    print(f"metadata: {stored.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())