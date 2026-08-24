"""Generate embeddings and compare semantic similarity."""

import argparse
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI, RateLimitError

from chat_completion import create_client


DEFAULT_TEXTS = [
    "How do I reset my account password?",
    "Steps to recover access to my login",
    "The cafeteria menu has pasta today",
]


def cosine_similarity(first: list[float], second: list[float]) -> float:
    """Compare vector direction using cosine similarity."""
    if len(first) != len(second) or not first:
        raise ValueError("vectors must be non-empty and have equal dimensions")
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return sum(a * b for a, b in zip(first, second)) / (first_norm * second_norm)


def generate_embeddings(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    """Generate one embedding vector for each input text."""
    if not texts:
        return []
    response = client.embeddings.create(model=model, input=texts)
    ordered = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("texts", nargs="*", default=DEFAULT_TEXTS)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    model = args.model or os.getenv("EMBED_MODEL", "").strip()
    if not model:
        print("Configuration error: EMBED_MODEL is missing from .env")
        return 0

    try:
        vectors = generate_embeddings(create_client(), model, args.texts)
    except AuthenticationError:
        print("Auth failed (401): check OPENAI_API_KEY in your .env")
        return 0
    except RateLimitError:
        print("Rate limited (429): slow down and retry with backoff")
        return 0
    except ValueError as error:
        print(f"Configuration error: {error}")
        return 0

    print(f"dimension: {len(vectors[0])}")
    print(f"first 8 values: {vectors[0][:8]}")
    print(f"password vs login recovery: {cosine_similarity(vectors[0], vectors[1]):.4f}")
    print(f"password vs cafeteria menu: {cosine_similarity(vectors[0], vectors[2]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())