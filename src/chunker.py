"""Split documents into retrievable chunks and report strategy statistics."""

import argparse
from dataclasses import dataclass
from pathlib import Path

from document_loader import LoadedDocument, load_documents


@dataclass(frozen=True)
class Chunk:
    """A retrievable text segment with its source and position."""

    source: str
    index: int
    text: str


def fixed_chunks(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into fixed character windows with boundary overlap."""
    if size <= 0:
        raise ValueError("size must be greater than zero")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")
    step = size - overlap
    return [text[start : start + size] for start in range(0, len(text), step)]


def paragraph_chunks(text: str) -> list[str]:
    """Split text at blank lines while discarding empty paragraphs."""
    return [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]


def add_metadata(source: str, chunks: list[str]) -> list[Chunk]:
    """Attach stable source and zero-based position metadata to chunks."""
    return [Chunk(source=source, index=index, text=text) for index, text in enumerate(chunks)]


def chunk_document(
    document: LoadedDocument,
    strategy: str = "fixed",
    size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Chunk one loaded document using the selected strategy."""
    if strategy == "fixed":
        chunks = fixed_chunks(document.text, size=size, overlap=overlap)
    elif strategy == "paragraph":
        chunks = paragraph_chunks(document.text)
    else:
        raise ValueError(f"unsupported chunking strategy: {strategy}")
    return add_metadata(document.source, chunks)


def average_size(chunks: list[Chunk]) -> float:
    """Return the average chunk length in characters."""
    return sum(len(chunk.text) for chunk in chunks) / len(chunks) if chunks else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("data"))
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    documents, skipped = load_documents(args.directory)
    for document in documents:
        for strategy in ("fixed", "paragraph"):
            chunks = chunk_document(document, strategy, args.size, args.overlap)
            print(
                f"{document.source} [{strategy}]: {len(chunks)} chunks, "
                f"avg {average_size(chunks):.1f} chars"
            )
            if chunks:
                print(f"sample: {chunks[0].text[:100]!r}")
    for message in skipped:
        print(message)
    print(f"Chunked {len(documents)} document(s); skipped {len(skipped)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())