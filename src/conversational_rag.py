"""Handle follow-up questions with query rewriting and grounded retrieval."""

import argparse
from dataclasses import dataclass, field
from typing import Any, Callable

from guardrails import REFUSAL_ANSWER, retrieval_is_strong


@dataclass
class Conversation:
    """Store a bounded sequence of user and assistant turns."""

    max_turns: int = 6
    history: list[dict[str, str]] = field(default_factory=list)

    def add_turn(self, user_question: str, answer: str) -> None:
        self.history.extend(
            [
                {"role": "user", "content": user_question},
                {"role": "assistant", "content": answer},
            ]
        )
        self.history = self.history[-(self.max_turns * 2) :]


def rewrite_followup(
    history: list[dict[str, str]],
    question: str,
    rewrite: Callable[[list[dict[str, str]], str], str],
) -> str:
    """Rewrite a contextual follow-up as a standalone retrieval query."""
    standalone = rewrite(history, question).strip()
    if not standalone:
        raise ValueError("follow-up rewrite returned an empty query")
    return standalone


@dataclass(frozen=True)
class ConversationalResult:
    rewritten_query: str
    answer: str
    sources: list[dict[str, Any]]


def conversational_answer(
    conversation: Conversation,
    user_question: str,
    rewrite: Callable[[list[dict[str, str]], str], str],
    retrieve: Callable[[str], list[dict[str, Any]]],
    generate: Callable[[str, list[dict[str, Any]]], str],
) -> ConversationalResult:
    """Rewrite, retrieve, guard, generate, and append the current turn."""
    standalone_query = rewrite_followup(conversation.history, user_question, rewrite)
    chunks = retrieve(standalone_query)
    if not retrieval_is_strong(chunks):
        answer = REFUSAL_ANSWER
        sources = []
    else:
        answer = generate(user_question, chunks)
        sources = [chunk["metadata"] for chunk in chunks]
    conversation.add_turn(user_question, answer)
    return ConversationalResult(standalone_query, answer, sources)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="What about the video?")
    args = parser.parse_args()
    conversation = Conversation()
    conversation.add_turn(
        "What evidence is required for project submission?",
        "The submission needs a PR link and a video explanation.",
    )
    result = conversational_answer(
        conversation,
        args.question,
        rewrite=lambda history, question: "What video explanation is required for project submission?",
        retrieve=lambda query: [{"score": 0.9, "metadata": {"source": "rubric.md"}, "text": "A video explanation is required."}],
        generate=lambda question, chunks: "A video explanation is required. [1]",
    )
    print(f"rewritten_query: {result.rewritten_query}")
    print(f"answer: {result.answer}")
    print(f"sources: {result.sources}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())