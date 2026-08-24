"""Run and validate the complete document ingestion pipeline."""

import argparse
from dataclasses import dataclass
from pathlib import Path

from chunker import Chunk
from document_loader import LoadedDocument, load_text
from text_cleaner import clean_text
from token_chunker import token_chunk_document


@dataclass(frozen=True)
class IngestionResult:
    """Complete ingestion report for one corpus run."""

    files: list[Path]
    documents: int
    chunks: list[Chunk]
    failures: list[tuple[str, str]]

    def validate(self) -> None:
        """Ensure every discovered file was ingested or reported as failed."""
        if self.documents + len(self.failures) != len(self.files):
            raise RuntimeError("a document was silently dropped")


def ingest(
    directory: Path,
    chunk_size: int = 400,
    overlap: int = 60,
) -> IngestionResult:
    """Load, clean, chunk, and tag every file under a directory."""
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    chunks = []
    failures = []
    documents = 0

    for path in files:
        try:
            text = clean_text(load_text(path))
            loaded_chunks = token_chunk_document(
                LoadedDocument(source=path.name, text=text),
                size=chunk_size,
                overlap=overlap,
            )
            search_start = 0
            for chunk in loaded_chunks:
                start = text.find(chunk.text, search_start)
                start = max(start, 0)
                chunks.append(
                    Chunk(
                        text=chunk.text,
                        metadata={
                            **chunk.metadata,
                            "char_start": start,
                            "char_end": start + len(chunk.text),
                        },
                    )
                )
                search_start = start + len(chunk.text)
            documents += 1
        except Exception as error:
            failures.append((path.name, str(error)))

    result = IngestionResult(files, documents, chunks, failures)
    result.validate()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("data"))
    parser.add_argument("--size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=60)
    args = parser.parse_args()

    result = ingest(args.directory, args.size, args.overlap)
    print(
        f"files={len(result.files)} docs={result.documents} "
        f"chunks={len(result.chunks)} failures={len(result.failures)}"
    )
    for name, error in result.failures:
        print(f"FAILED: {name}: {error}")
    if result.chunks:
        print(f"sample: {result.chunks[0].text[:80]!r} | {result.chunks[0].metadata}")
    print("Validation: files == docs + failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())