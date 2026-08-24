"""Request, parse, and validate structured JSON model responses."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI, RateLimitError

from chat_completion import create_client


SYSTEM_PROMPT = (
    "Reply with ONLY a JSON object containing the keys answer and source. "
    "The answer and source values must be strings, with no extra prose."
)
REQUIRED_FIELDS = ("answer", "source")


def parse_response(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse JSON and validate its required fields and value types."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, "malformed JSON"

    if not isinstance(data, dict):
        return None, "response must be a JSON object"

    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        return None, f"missing fields: {missing}"

    invalid = [field for field in REQUIRED_FIELDS if not isinstance(data[field], str)]
    if invalid:
        return None, f"fields must be strings: {invalid}"
    return data, None


def build_request(model: str, question: str, system_prompt: str = SYSTEM_PROMPT) -> dict[str, Any]:
    """Build a low-randomness JSON-mode chat request."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }


def request_json(client: OpenAI, model: str, question: str) -> tuple[dict[str, Any] | None, str | None]:
    """Request valid JSON and retry once after malformed or incomplete output."""
    request = build_request(model, question)
    for attempt in range(2):
        response = client.chat.completions.create(**request)
        data, error = parse_response(response.choices[0].message.content or "")
        if data is not None:
            return data, None
        if attempt == 0:
            request["messages"] = [
                *request["messages"],
                {
                    "role": "user",
                    "content": "Return valid JSON only with both string fields: answer and source.",
                },
            ]
    return None, error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="What is the refund window?")
    parser.add_argument("--raw", help="Parse a raw JSON response without calling the API.")
    args = parser.parse_args()

    if args.raw is not None:
        data, error = parse_response(args.raw)
        print(data if data is not None else f"recover: {error}")
        return 0

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    model = os.getenv("CHAT_MODEL", "").strip()
    if not model:
        print("Configuration error: CHAT_MODEL is missing from .env")
        return 0
    try:
        data, error = request_json(create_client(), model, args.question)
    except AuthenticationError:
        print("Auth failed (401): check OPENAI_API_KEY in your .env")
        return 0
    except RateLimitError:
        print("Rate limited (429): slow down and retry with backoff")
        return 0
    except ValueError as config_error:
        print(f"Configuration error: {config_error}")
        return 0

    print(data if data is not None else f"recover: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())