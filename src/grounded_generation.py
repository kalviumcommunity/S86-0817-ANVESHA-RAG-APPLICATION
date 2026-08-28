"""Generate answers from retrieved context and verify their support."""

import argparse
from dataclasses import dataclass
from typing import Any, Callable

from context_injection import FALLBACK_INSTRUCTION, build_prompt


@dataclass(frozen=True)
class GroundedAnswer:
    """Answer text together with its prompt context and source metadata."""

    question: str
    answer: str
    context: str
    sources: list[dict[str, Any]]


def generate_grounded_answer(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    call_llm: Callable[[str], str],
    max_context_tokens: int = 5_000,
) -> GroundedAnswer:
    """Generate only when retrieved evidence exists, otherwise return a fallback."""
    prompt_data = build_prompt(question, retrieved_chunks, max_context_tokens)
    if not retrieved_chunks or not prompt_data.sources_used:
        return GroundedAnswer(question, FALLBACK_INSTRUCTION, prompt_data.prompt, [])
    answer = call_llm(prompt_data.prompt)
    return GroundedAnswer(question, answer, prompt_data.prompt, prompt_data.sources_used)


def verify_grounding(result: GroundedAnswer) -> dict[str, Any]:
    """Report whether the response has evidence and at least one citation marker."""
    has_support = bool(result.sources)
    has_citation = any(f"[{index}]" in result.answer for index in range(1, len(result.sources) + 1))
    return {
        "supported": has_support and has_citation,
        "has_context": bool(result.context),
        "has_sources": has_support,
        "has_citation": has_citation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="What evidence is required for project submission?")
    args = parser.parse_args()
    chunks = [
        {
            "text": "Project submissions require screenshots and a written report.",
            "metadata": {"source": "submission-rubric.md", "chunk_index": 2},
        }
    ]
    result = generate_grounded_answer(
        args.question,
        chunks,
        lambda prompt: "[1] Submission evidence includes screenshots and a written report.",
    )
    print(f"answer: {result.answer}")
    print(f"sources: {result.sources}")
    print(f"grounding_check: {verify_grounding(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())