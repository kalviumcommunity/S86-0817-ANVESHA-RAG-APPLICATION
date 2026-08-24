"""Split documents by token count while preserving boundary context."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from document_loader import LoadedDocument, load_documents


ENCODING_NAME = "cl100k_base"


@dataclass(frozen=True)
class TokenChunk:
    """A token-sized chunk with source and token-position metadata."""

    text: str
    metadata: dict[str, str | int]


def token_chunks(
    text: str,
    size: int = 400,
    overlap: int = 60,
    encoding_name: str = ENCODING_NAME,
) -> list[str]:
    """Split text into token windows with controlled overlap."""
    if size <= 0:
        raise ValueError("size must be greater than zero")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")
    tokens = tiktoken.get_encoding(encoding_name).encode(text)
    step = size - overlap
    return [
        tiktoken.get_encoding(encoding_name).decode(tokens[start : start + size])
        for start in range(0, len(tokens), step)
    ]


def token_chunk_document(
    document: LoadedDocument,
    size: int = 400,
    overlap: int = 60,
    encoding_name: str = ENCODING_NAME,
) -> list[TokenChunk]:
    """Create token chunks and retain source and token offset metadata."""
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(document.text)
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("size must be greater than zero and overlap must be smaller than size")
    step = size - overlap
    chunks = []
    for index, start in enumerate(range(0, len(tokens), step)):
        chunk_tokens = tokens[start : start + size]
        chunks.append(
            TokenChunk(
                text=encoding.decode(chunk_tokens),
                metadata={
                    "source": document.source,
                    "chunk_index": index,
                    "token_start": start,
                    "token_end": start + len(chunk_tokens),
                },
            )
        )
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("data"))
    parser.add_argument("--size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=60)
    args = parser.parse_args()

    documents, skipped = load_documents(args.directory)
    for document in documents:
        for overlap in (0, args.overlap):
            chunks = token_chunk_document(document, args.size, overlap)
            print(f"{document.source} overlap={overlap}: {len(chunks)} chunks")
            if chunks:
                print(f"sample: {chunks[0].text[:100]!r}")
    for message in skipped:
        print(message)
    print(f"Chunked {len(documents)} document(s); skipped {len(skipped)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())