"""Count tokens and estimate chat completion cost."""

import argparse
from pathlib import Path

import tiktoken


ENCODING_NAME = "cl100k_base"
INPUT_COST_PER_1K = 0.0005
OUTPUT_COST_PER_1K = 0.0015
DEFAULT_CONTEXT_LIMIT = 128_000


def count_tokens(text: str, encoding_name: str = ENCODING_NAME) -> int:
    """Count tokens using the encoding commonly used by OpenAI models."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    input_rate: float = INPUT_COST_PER_1K,
    output_rate: float = OUTPUT_COST_PER_1K,
) -> float:
    """Estimate cost when rates are expressed in dollars per 1,000 tokens."""
    return (input_tokens / 1_000 * input_rate) + (output_tokens / 1_000 * output_rate)


def estimate_text(prompt: str, answer: str) -> dict[str, float | int]:
    """Return token counts and estimated cost for one request and response."""
    input_tokens = count_tokens(prompt)
    output_tokens = count_tokens(answer)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost": estimate_cost(input_tokens, output_tokens),
    }


def fits_context(
    input_tokens: int,
    output_tokens: int,
    context_limit: int = DEFAULT_CONTEXT_LIMIT,
) -> bool:
    """Check whether input and expected output fit the model context window."""
    return input_tokens + output_tokens <= context_limit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="What is our refund window?")
    parser.add_argument("--answer", default="The refund window is 30 days.")
    parser.add_argument("--file", type=Path, help="Count a document as the input prompt.")
    parser.add_argument("--context-limit", type=int, default=DEFAULT_CONTEXT_LIMIT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt = args.file.read_text(encoding="utf-8") if args.file else args.prompt
    estimate = estimate_text(prompt, args.answer)
    within_limit = fits_context(
        estimate["input_tokens"], estimate["output_tokens"], args.context_limit
    )

    print(f"encoding={ENCODING_NAME}")
    print(f"input_tokens={estimate['input_tokens']}")
    print(f"output_tokens={estimate['output_tokens']}")
    print(f"total_tokens={estimate['total_tokens']}")
    print(f"estimated_cost=${estimate['estimated_cost']:.6f}")
    print(f"fits_context_limit={within_limit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())