"""Document upload and indexing endpoint for runtime knowledge base growth."""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


# Configuration
UPLOAD_DIR = Path("uploads")
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@dataclass(frozen=True)
class UploadError:
    """Upload error details."""

    status_code: int
    detail: str


@dataclass(frozen=True)
class TaggedChunk:
    """A chunk with source metadata."""

    source: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class EmbeddedChunk:
    """A chunk with embedding vector."""

    source: str
    chunk_index: int
    text: str
    embedding: list[float]


@dataclass(frozen=True)
class ProcessingResult:
    """Result of document processing."""

    document: str
    chunks_created: int
    chunks_indexed: int


@dataclass(frozen=True)
class UploadResponse:
    """Response after successful upload and indexing."""

    status: str
    filename: str
    summary: ProcessingResult


def validate_upload(filename: str, content_size: int) -> Optional[UploadError]:
    """Validate uploaded file extension and size."""
    if not filename:
        return UploadError(status_code=400, detail="Filename is required")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return UploadError(
            status_code=415,
            detail=f"Unsupported file type: {suffix}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    if content_size == 0:
        return UploadError(status_code=400, detail="File is empty")

    if content_size > MAX_FILE_SIZE:
        return UploadError(
            status_code=413,
            detail=f"File exceeds maximum size of {MAX_FILE_SIZE / 1024 / 1024:.0f} MB",
        )

    return None


def store_upload(filename: str, content: bytes) -> Path:
    """Store uploaded file with safe path handling."""
    UPLOAD_DIR.mkdir(exist_ok=True)
    safe_name = Path(filename).name  # Strips directory traversal attempts
    path = UPLOAD_DIR / safe_name
    path.write_bytes(content)
    return path


def load_text(path: Path) -> str:
    """Load text content from file."""
    return path.read_text(encoding="utf-8", errors="replace")


def clean_text(raw_text: str) -> str:
    """Clean and normalize text content."""
    # Remove excess whitespace
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    return "\n".join(lines)


def token_chunks(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    """Split text into token-sized chunks with overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = max(start + 1, end - overlap)  # Always advance by at least 1
    return chunks if chunks else [""]


def tag_chunks(source: str, chunks: list[str]) -> list[TaggedChunk]:
    """Add source metadata to chunks."""
    return [TaggedChunk(source=source, chunk_index=i, text=chunk) for i, chunk in enumerate(chunks)]


def embed_chunks(
    tagged_chunks: list[TaggedChunk],
    embed_fn: Callable[[str], list[float]],
) -> list[EmbeddedChunk]:
    """Embed chunks using provided embedding function."""
    return [
        EmbeddedChunk(
            source=chunk.source,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            embedding=embed_fn(chunk.text),
        )
        for chunk in tagged_chunks
    ]


def index_chunks(
    embedded_chunks: list[EmbeddedChunk],
    index_fn: Callable[[list[EmbeddedChunk]], int],
) -> int:
    """Index embedded chunks into vector database."""
    return index_fn(embedded_chunks)


def process_uploaded_document(
    path: Path,
    embed_fn: Callable[[str], list[float]],
    index_fn: Callable[[list[EmbeddedChunk]], int],
) -> ProcessingResult:
    """Process uploaded document through full pipeline."""
    raw_text = load_text(path)
    cleaned = clean_text(raw_text)
    chunks = token_chunks(cleaned)
    tagged_chunks = tag_chunks(source=str(path), chunks=chunks)
    embedded_chunks = embed_chunks(tagged_chunks, embed_fn)
    indexed_count = index_chunks(embedded_chunks, index_fn)

    return ProcessingResult(
        document=str(path),
        chunks_created=len(tagged_chunks),
        chunks_indexed=indexed_count,
    )


def handle_upload(
    filename: str,
    content: bytes,
    embed_fn: Callable[[str], list[float]],
    index_fn: Callable[[list[EmbeddedChunk]], int],
) -> UploadResponse:
    """Handle complete upload workflow: validate, store, process, index."""
    # Validate
    validation_error = validate_upload(filename, len(content))
    if validation_error:
        raise ValueError(f"Upload validation failed: {validation_error.detail}")

    # Store
    path = store_upload(filename, content)

    # Process and index
    summary = process_uploaded_document(path, embed_fn, index_fn)

    return UploadResponse(status="indexed", filename=filename, summary=summary)


def main() -> int:
    """Demo: show file validation, storage, processing, and indexing."""
    print("=== Document Upload & Indexing Endpoint ===\n")

    print("=== File Validation ===")
    # Valid file
    error = validate_upload("policy.md", 5000)
    print(f"✓ policy.md (5000 bytes): {'valid' if error is None else error.detail}")

    # Invalid extension
    error = validate_upload("document.docx", 5000)
    print(f"✓ document.docx rejected: {error.detail if error else 'unexpected'}")

    # Empty file
    error = validate_upload("empty.txt", 0)
    print(f"✓ empty.txt rejected: {error.detail if error else 'unexpected'}")

    # File too large
    error = validate_upload("huge.txt", 20 * 1024 * 1024)
    print(f"✓ huge.txt rejected: {error.detail if error else 'unexpected'}")

    print("\n=== Document Processing Pipeline ===")
    # Create demo document
    demo_text = "The submission requires a PR link. Evidence must include sample output. Video explanation is mandatory."
    print(f"Raw text ({len(demo_text)} chars): {demo_text[:50]}...")

    # Clean
    cleaned = clean_text(demo_text)
    print(f"Cleaned: {len(cleaned)} chars, {len(cleaned.split())} words")

    # Chunk
    chunks = token_chunks(cleaned, chunk_size=5, overlap=2)
    print(f"Chunks: {len(chunks)} created")
    for i, chunk in enumerate(chunks[:2]):
        print(f"  [{i}] {chunk[:40]}...")

    # Tag
    tagged = tag_chunks(source="demo.md", chunks=chunks)
    print(f"Tagged: {len(tagged)} chunks with source metadata")

    print("\n=== Mock Embedding & Indexing ===")

    def mock_embed(text: str) -> list[float]:
        """Mock embedding: deterministic vector based on text."""
        return [float(hash(text) % 100) / 100 for _ in range(3)]

    def mock_index(embedded: list[EmbeddedChunk]) -> int:
        """Mock indexing: return count of indexed chunks."""
        return len(embedded)

    # Full upload workflow
    try:
        demo_content = b"Policy update: All submissions must include PR, output, and video explanation."
        response = handle_upload("policy.md", demo_content, mock_embed, mock_index)
        print(f"✓ Upload successful")
        print(f"  Filename: {response.filename}")
        print(f"  Status: {response.status}")
        print(f"  Chunks created: {response.summary.chunks_created}")
        print(f"  Chunks indexed: {response.summary.chunks_indexed}")
    except ValueError as e:
        print(f"✗ Upload failed: {e}")

    print("\n=== Error Handling ===")
    # Invalid file type
    try:
        invalid_content = b"Some data"
        handle_upload("data.xlsx", invalid_content, mock_embed, mock_index)
        print("✗ Should have failed")
    except ValueError as e:
        print(f"✓ Caught error: {str(e)[:60]}...")

    # Empty file
    try:
        empty_content = b""
        handle_upload("empty.txt", empty_content, mock_embed, mock_index)
        print("✗ Should have failed")
    except ValueError as e:
        print(f"✓ Caught error: {str(e)[:60]}...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
