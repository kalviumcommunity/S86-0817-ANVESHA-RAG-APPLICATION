"""Compare vague and clear prompts while keeping roles explicit."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI, RateLimitError

from chat_completion import create_client


LOGGER = logging.getLogger(__name__)
SYSTEM_PROMPT = (
    "You are a support assistant for an internal documents tool. "
    "Answer using only information provided by the user. "
    "Be concise and factual. If the information is missing, say: I don't know."
)
PROMPTS = {
    "vague": "Explain our refund policy.",
    "clear": (
        "In one sentence, state the refund window in days. "
        "Reply with only a JSON object containing the keys answer and source. "
        "If the refund window is not provided, set answer to I don't know and source to null."
    ),
}


def build_messages(user_prompt: str) -> list[dict[str, str]]:
    """Build a chat request with separate system and user roles."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def compare_prompts(client: OpenAI, model: str) -> dict[str, str]:
    """Run both prompt variations and return their generated responses."""
    results = {}
    for name, prompt in PROMPTS.items():
        messages = build_messages(prompt)
        LOGGER.info("REQUEST [%s]: model=%s messages=%s", name, model, messages)
        response = client.chat.completions.create(model=model, messages=messages)
        content = response.choices[0].message.content or ""
        results[name] = content
        LOGGER.info("RESPONSE [%s]: %s", name, content)
    return results


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    model = os.getenv("CHAT_MODEL", "").strip()
    if not model:
        print("Configuration error: CHAT_MODEL is missing from .env")
        return 0

    try:
        results = compare_prompts(create_client(), model)
    except AuthenticationError:
        print("Auth failed (401): check OPENAI_API_KEY in your .env")
        return 0
    except RateLimitError:
        print("Rate limited (429): slow down and retry with backoff")
        return 0
    except ValueError as error:
        print(f"Configuration error: {error}")
        return 0

    for name, answer in results.items():
        print(f"{name}: {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())