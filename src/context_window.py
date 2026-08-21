"""Manage chat history under a fixed context-window budget."""

import argparse
from dataclasses import dataclass, field

from token_cost_estimator import count_tokens


def total_tokens(messages: list[dict[str, str]]) -> int:
    """Count the text tokens in every message in a conversation."""
    return sum(count_tokens(message["content"]) for message in messages)


def trim_history(
    messages: list[dict[str, str]],
    budget: int,
) -> list[dict[str, str]]:
    """Drop the oldest non-system messages until the history fits the budget."""
    if budget < 0:
        raise ValueError("budget must be zero or greater")

    trimmed = list(messages)
    system_message = trimmed[:1] if trimmed and trimmed[0]["role"] == "system" else []
    first_turn = 1 if system_message else 0
    while total_tokens(trimmed) > budget and len(trimmed) > first_turn:
        trimmed.pop(first_turn)
    return trimmed


@dataclass
class ConversationHistory:
    """Track a conversation and prepare requests within a token budget."""

    system_prompt: str
    context_limit: int = 6_000
    messages: list[dict[str, str]] = field(init=False)

    def __post_init__(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def prepare_request(self, expected_output_tokens: int = 512) -> list[dict[str, str]]:
        """Trim history while reserving space for the next model response."""
        available_input = self.context_limit - expected_output_tokens
        if available_input < 0:
            raise ValueError("expected output exceeds the context limit")
        self.messages = trim_history(self.messages, available_input)
        return list(self.messages)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=40)
    args = parser.parse_args()

    conversation = ConversationHistory(
        "You are a concise assistant.", context_limit=args.budget
    )
    for turn in range(1, 6):
        conversation.add_user_message(
            f"Turn {turn}: explain how retrieved context supports a grounded answer."
        )
        conversation.add_assistant_message(
            f"Turn {turn}: retrieved context provides evidence for the answer."
        )

    before = total_tokens(conversation.messages)
    prepared = conversation.prepare_request(expected_output_tokens=8)
    after = total_tokens(prepared)
    print(f"before_tokens={before}")
    print(f"after_tokens={after}")
    print(f"reserved_output_tokens=8")
    print(f"context_limit={args.budget}")
    print(f"messages_kept={len(prepared)}")
    print(f"system_message_kept={prepared[0]['role'] == 'system'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())