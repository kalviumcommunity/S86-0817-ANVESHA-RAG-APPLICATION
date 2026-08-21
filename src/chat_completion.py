"""Send a first chat completion through an OpenAI-compatible API."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI, RateLimitError


LOGGER = logging.getLogger(__name__)
MESSAGES = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Say hello in one sentence."},
]


def create_client() -> OpenAI:
    """Create a client from the repository's .env configuration."""
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key in {"your_actual_key_here", "your-api-key"}:
        raise ValueError("OPENAI_API_KEY is missing from .env")

    client_options = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if base_url:
        client_options["base_url"] = base_url
    return OpenAI(**client_options)


def ask_assistant() -> str:
    """Send the sample request and return the generated message content."""
    model = os.getenv("CHAT_MODEL", "").strip()
    if not model:
        raise ValueError("CHAT_MODEL is missing from .env")

    client = create_client()
    LOGGER.info("REQUEST: model=%s messages=%s", model, MESSAGES)
    response = client.chat.completions.create(model=model, messages=MESSAGES)
    content = response.choices[0].message.content or ""
    LOGGER.info("RESPONSE: %s", content)
    LOGGER.info("USAGE: %s", response.usage)
    return content


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        print(ask_assistant())
    except AuthenticationError:
        print("Auth failed (401): check OPENAI_API_KEY in your .env")
    except RateLimitError:
        print("Rate limited (429): slow down and retry with backoff")
    except ValueError as error:
        print(f"Configuration error: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())