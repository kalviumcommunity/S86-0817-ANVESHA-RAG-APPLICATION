"""Control chat-completion randomness, length, and stopping behavior."""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI, RateLimitError

from chat_completion import create_client


MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are a factual RAG assistant. Use only the supplied context, "
            "answer concisely, and say I don't know when the context is insufficient."
        ),
    },
    {"role": "user", "content": "Context: The refund window is 30 days.\nQuestion: How long is it?"},
]


@dataclass(frozen=True)
class OutputSettings:
    """Generation settings chosen for grounded, cost-controlled answers."""

    temperature: float = 0.1
    max_tokens: int = 300
    top_p: float = 1.0
    stop: tuple[str, ...] = ("\n\nUser:",)

    def __post_init__(self) -> None:
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be greater than zero")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be greater than zero and at most 1")


def build_request(
    model: str,
    messages: list[dict[str, str]],
    settings: OutputSettings = OutputSettings(),
) -> dict[str, object]:
    """Build the complete request payload without sending it."""
    return {
        "model": model,
        "messages": messages,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "top_p": settings.top_p,
        "stop": list(settings.stop),
    }


def request_completion(client: OpenAI, payload: dict[str, object]) -> str:
    """Send a parameterized request and return the generated text."""
    response = client.chat.completions.create(**payload)
    return response.choices[0].message.content or ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    model = os.getenv("CHAT_MODEL", "").strip()
    settings = OutputSettings(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
    )
    payload = build_request(model, MESSAGES, settings)
    if args.dry_run:
        print(payload)
        return 0
    if not model:
        print("Configuration error: CHAT_MODEL is missing from .env")
        return 0

    try:
        print(request_completion(create_client(), payload))
    except AuthenticationError:
        print("Auth failed (401): check OPENAI_API_KEY in your .env")
    except RateLimitError:
        print("Rate limited (429): slow down and retry with backoff")
    except ValueError as error:
        print(f"Configuration error: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())