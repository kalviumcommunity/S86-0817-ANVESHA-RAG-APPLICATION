"""Demonstrate consuming the shared answer prompt."""

from prompts.answer import build_answer_prompt


def main() -> int:
    prompt = build_answer_prompt(
        context="The refund window is 30 days from the purchase date.",
        question="What is the refund window?",
    )
    print(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())