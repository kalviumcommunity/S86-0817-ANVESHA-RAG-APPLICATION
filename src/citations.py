"""Build and verify source citations for grounded answers."""

import argparse
import re
from dataclasses import dataclass
from typing import Any, Callable

from context_injection import FALLBACK_INSTRUCTION, assemble_context


@dataclass(frozen=True)
class CitedAnswer:
    """An answer paired with its source citation map."""

    answer: str
    citations: dict[str, dict[str, Any]]


def build_citation_map(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map numbered markers to real retrieved chunk metadata and text."""
    citation_map = {}
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]
        citation_map[f"[{index}]"] = {
            "source": metadata["source"],
            "chunk_id": metadata.get("chunk_id", chunk.get("id")),
            "chunk_index": metadata.get("chunk_index"),
            "section": metadata.get("section"),
            "page": metadata.get("page"),
            "text": chunk["text"],
        }
    return citation_map


def build_cited_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """Build a prompt requiring citations from the supplied context only."""
    context, _, _ = assemble_context(chunks)
    return (
        "Answer using only the context below.\n"
        "Cite every factual claim using source markers like [1] or [2].\n"
        "Only use source markers that appear in the context.\n"
        f"If unsupported, say: {FALLBACK_INSTRUCTION}\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )


def verify_citations(answer: str, citation_map: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    """Ensure every marker used in an answer maps to retrieved evidence."""
    used = sorted(set(re.findall(r"\[\d+\]", answer)))
    invalid = [marker for marker in used if marker not in citation_map]
    return not invalid, invalid


def answer_with_citations(
    question: str,
    retrieve: Callable[[str], list[dict[str, Any]]],
    call_llm: Callable[[str], str],
) -> CitedAnswer:
    """Retrieve evidence, generate a cited answer, and return its citation map."""
    chunks = retrieve(question)
    if not chunks:
        return CitedAnswer(FALLBACK_INSTRUCTION, {})
    citation_map = build_citation_map(chunks)
    answer = call_llm(build_cited_prompt(question, chunks))
    valid, invalid = verify_citations(answer, citation_map)
    if not valid:
        return CitedAnswer(FALLBACK_INSTRUCTION, {})
    return CitedAnswer(answer, citation_map)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="What evidence is required?")
    args = parser.parse_args()
    chunks = [{
        "id": "rubric:2",
        "text": "Screenshots and a written report are required.",
        "metadata": {"source": "submission-rubric.md", "chunk_index": 2, "section": "Evidence"},
    }]
    result = answer_with_citations(
        args.question,
        retrieve=lambda question: chunks,
        call_llm=lambda prompt: "[1] Screenshots and a written report are required.",
    )
    print(f"answer: {result.answer}")
    print(f"citations: {result.citations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())