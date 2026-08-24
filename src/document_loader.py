"""Load supported documents into text while preserving source identity."""

import argparse
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".html", ".htm"}


@dataclass(frozen=True)
class LoadedDocument:
    """Normalized document content and its original source name."""

    source: str
    text: str


def load_text(path: Path) -> str:
    """Extract plain text from a supported PDF, text, Markdown, or HTML file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages).strip()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        html = path.read_text(encoding="utf-8", errors="ignore")
        return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    raise ValueError(f"unsupported file type: {suffix or '<none>'}")


def load_documents(directory: Path) -> tuple[list[LoadedDocument], list[str]]:
    """Load all supported files, returning successful documents and skip messages."""
    documents = []
    skipped = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            documents.append(LoadedDocument(source=path.name, text=load_text(path)))
        except Exception as error:
            skipped.append(f"SKIP {path.name}: {error}")
    return documents, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("data"))
    args = parser.parse_args()

    documents, skipped = load_documents(args.directory)
    for document in documents:
        print(f"OK {document.source}: {len(document.text)} chars | {document.text[:60]!r}")
    for message in skipped:
        print(message)
    print(f"Loaded {len(documents)} document(s); skipped {len(skipped)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())