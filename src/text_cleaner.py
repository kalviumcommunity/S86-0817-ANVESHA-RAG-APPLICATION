"""Normalize extracted document text for consistent downstream retrieval."""

import argparse
import re
import unicodedata
from collections import Counter
from pathlib import Path

from document_loader import LoadedDocument, load_documents


PAGE_MARKER = re.compile(r"^\s*Page\s+\d+(?:\s+of\s+\d+)?\s*$", re.IGNORECASE)


def _repeated_lines(text: str, minimum_repetitions: int = 2) -> set[str]:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    counts = Counter(lines)
    return {line for line, count in counts.items() if count >= minimum_repetitions}


def clean_text(text: str) -> str:
    """Remove common extraction noise while preserving readable content."""
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    repeated = _repeated_lines(normalized)
    kept_lines = []
    for line in normalized.split("\n"):
        stripped = line.strip()
        if PAGE_MARKER.match(stripped) or (stripped and stripped in repeated):
            continue
        kept_lines.append(line)
    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def clean_documents(documents: list[LoadedDocument]) -> list[LoadedDocument]:
    """Apply the same cleaning function to every loaded document."""
    return [LoadedDocument(source=document.source, text=clean_text(document.text)) for document in documents]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("data"))
    args = parser.parse_args()

    documents, skipped = load_documents(args.directory)
    cleaned_documents = clean_documents(documents)
    for before, after in zip(documents, cleaned_documents):
        print(f"{before.source}: {len(before.text)} -> {len(after.text)} chars")
        print(f"BEFORE: {before.text[:120]!r}")
        print(f"AFTER:  {after.text[:120]!r}")
    for message in skipped:
        print(message)
    print(f"Cleaned {len(cleaned_documents)} document(s); skipped {len(skipped)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())