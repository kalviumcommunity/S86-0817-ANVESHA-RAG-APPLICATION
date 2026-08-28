"""Assemble retrieved evidence into a token-budgeted grounded prompt."""

import argparse
from dataclasses import dataclass
from typing import Any

from token_cost_estimator import count_tokens


FALLBACK_INSTRUCTION = "I don't have enough information in the provided context."


@dataclass(frozen=True)
class AugmentedPrompt:
    """Prompt plus the evidence and token usage included in it."""

    prompt: str
    context_tokens: int
    sources_used: list[dict[str, Any]]


def format_chunk(index: int, chunk: dict[str, Any]) -> str:
    """Format a retrieved chunk with an auditable source marker."""
    metadata = chunk["metadata"]
    source = metadata["source"]
    chunk_index = metadata.get("chunk_index")
    marker = f"[{index}] {source}#{chunk_index}"
    return f"{marker}\n{chunk['text']}"


def assemble_context(
    chunks: list[dict[str, Any]],
    max_context_tokens: int = 5_000,
) -> tuple[str, int, list[dict[str, Any]]]:
    """Select ranked chunks until the context token budget is exhausted."""
    if max_context_tokens < 0:
        raise ValueError("max_context_tokens must be zero or greater")
    selected = []
    formatted_parts = []
    used_tokens = 0
    for index, chunk in enumerate(chunks, start=1):
        formatted = format_chunk(index, chunk)
        token_count = count_tokens(formatted)
        if used_tokens + token_count > max_context_tokens:
            break
        formatted_parts.append(formatted)
        selected.append(chunk["metadata"])
        used_tokens += token_count
    return "\n\n---\n\n".join(formatted_parts), used_tokens, selected


def build_prompt(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    max_context_tokens: int = 5_000,
) -> AugmentedPrompt:
    """Build a grounded prompt from the highest-ranked retrieved chunks."""
    context, context_tokens, sources = assemble_context(retrieved_chunks, max_context_tokens)
    prompt = (
        "You are a grounded assistant.\n"
        "Answer the question using only the provided context.\n"
        f"If the answer is not in the context, say: {FALLBACK_INSTRUCTION}\n"
        "When possible, cite sources using markers like [1] or [2].\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )
    return AugmentedPrompt(prompt, context_tokens, sources)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="What evidence is required?")
    parser.add_argument("--max-context-tokens", type=int, default=100)
    args = parser.parse_args()
    chunks = [
        {
            "text": "Screenshots and a written report are required.",
            "metadata": {"source": "submission-rubric.md", "chunk_index": 2},
        }
    ]
    result = build_prompt(args.question, chunks, args.max_context_tokens)
    print(result.prompt)
    print(f"context_tokens: {result.context_tokens}")
    print(f"sources_used: {result.sources_used}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())