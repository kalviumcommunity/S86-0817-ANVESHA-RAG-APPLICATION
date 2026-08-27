"""Batch, retry, and account for embedding API work."""

import argparse
import time
from dataclasses import dataclass
from typing import Any, Callable

import tiktoken
from openai import RateLimitError


PRICE_PER_1K_TOKENS = 0.00002


@dataclass
class EmbeddingRunSummary:
    """Counters and cost information for one embedding run."""

    total_chunks: int
    skipped_existing: int = 0
    embedded: int = 0
    failed: int = 0
    input_tokens: int = 0

    @property
    def estimated_cost_usd(self) -> float:
        return self.input_tokens / 1_000 * PRICE_PER_1K_TOKENS


def batches(items: list[dict[str, Any]], size: int):
    """Yield consecutive batches of chunks."""
    if size <= 0:
        raise ValueError("batch size must be greater than zero")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def embed_with_retry(
    client: Any,
    model: str,
    texts: list[str],
    max_attempts: int = 5,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Retry rate-limit and transient errors with exponential backoff."""
    if max_attempts <= 0:
        raise ValueError("max_attempts must be greater than zero")
    for attempt in range(max_attempts):
        try:
            return client.embeddings.create(model=model, input=texts)
        except Exception:
            if attempt == max_attempts - 1:
                raise
            sleep(2**attempt)
    raise RuntimeError("unreachable")


def run_embedding_job(
    client: Any,
    model: str,
    chunks: list[dict[str, Any]],
    existing_embedding_ids: set[str] | None = None,
    batch_size: int = 64,
    encoding_name: str = "cl100k_base",
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[dict[str, Any]], EmbeddingRunSummary]:
    """Embed only pending chunks and return records plus a verified summary."""
    existing_ids = existing_embedding_ids or set()
    pending = [chunk for chunk in chunks if chunk["id"] not in existing_ids]
    summary = EmbeddingRunSummary(
        total_chunks=len(chunks), skipped_existing=len(chunks) - len(pending)
    )
    encoding = tiktoken.get_encoding(encoding_name)
    records = []
    for batch in batches(pending, batch_size):
        summary.input_tokens += sum(len(encoding.encode(chunk["text"])) for chunk in batch)
        try:
            response = embed_with_retry(
                client,
                model,
                [chunk["text"] for chunk in batch],
                sleep=sleep,
            )
            items = sorted(response.data, key=lambda item: item.index)
            if len(items) != len(batch):
                raise ValueError("embedding API returned an unexpected number of vectors")
            records.extend(
                {
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "embedding": item.embedding,
                }
                for chunk, item in zip(batch, items)
            )
            summary.embedded += len(batch)
        except Exception:
            summary.failed += len(batch)
    return records, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    print(f"batch_size={args.batch_size}")
    print(f"price_per_1k_tokens=${PRICE_PER_1K_TOKENS:.5f}")
    print("Use run_embedding_job() with a configured embeddings client for a live run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())