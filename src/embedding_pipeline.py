"""Generate and store embeddings for prepared chunks in batches."""

import argparse
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI, RateLimitError

from chat_completion import create_client


def embed_chunks(
    client: OpenAI,
    model: str,
    chunks: list[dict[str, Any]],
    batch_size: int = 100,
) -> list[dict[str, Any]]:
    """Embed chunks in batches while preserving text and metadata."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    records = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        response = client.embeddings.create(
            model=model,
            input=[chunk["text"] for chunk in batch],
        )
        items = sorted(response.data, key=lambda item: item.index)
        if len(items) != len(batch):
            raise ValueError("embedding API returned an unexpected number of vectors")
        records.extend(
            {
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "embedding": item.embedding,
                "embedding_model": model,
            }
            for chunk, item in zip(batch, items)
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    model = os.getenv("EMBEDDING_MODEL", os.getenv("EMBED_MODEL", "text-embedding-3-small"))
    if args.dry_run:
        print(f"model: {model}")
        print(f"batch_size: {args.batch_size}")
        print("Embedding API configuration loaded.")
        return 0
    try:
        records = embed_chunks(
            create_client(),
            model,
            [
                {
                    "text": "Password reset instructions for learner accounts.",
                    "metadata": {"source": "account-guide.md", "chunk_index": 0},
                },
                {
                    "text": "Learners can recover access using their registered email.",
                    "metadata": {"source": "account-guide.md", "chunk_index": 1},
                },
            ],
            args.batch_size,
        )
    except AuthenticationError:
        print("Auth failed (401): check OPENAI_API_KEY in your .env")
        return 0
    except RateLimitError:
        print("Rate limited (429): slow down and retry with backoff")
        return 0
    except ValueError as error:
        print(f"Configuration error: {error}")
        return 0

    print(f"model: {model}")
    print(f"records: {len(records)}")
    print(f"vector length: {len(records[0]['embedding'])}")
    print(f"sample values: {records[0]['embedding'][:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())