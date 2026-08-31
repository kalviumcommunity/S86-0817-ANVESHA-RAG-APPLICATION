"""Backend API for the RAG service using FastAPI and Pydantic."""

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = object
    Field = lambda **kwargs: None


# Load config from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "demo-key")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
VECTOR_DB_URL = os.getenv("VECTOR_DB_URL", "http://localhost:8000")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_chunks")


# Pydantic request/response models
class QueryRequest(BaseModel):
    """Incoming query from client."""

    question: str = Field(min_length=3, max_length=1000)

    class Config:
        json_schema_extra = {
            "example": {"question": "What evidence is required for project submission?"}
        }


class Source(BaseModel):
    """Source metadata for a retrieved chunk."""

    source: str
    chunk_id: Optional[str] = None
    score: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {"source": "submission-rubric.md", "chunk_id": "submission-rubric.md:1", "score": 0.92}
        }


class QueryResponse(BaseModel):
    """Response with answer and sources."""

    answer: str
    sources: list[Source]
    status: str

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The submission requires a PR link, sample output, and a video explanation.",
                "sources": [{"source": "submission-rubric.md", "chunk_id": "submission-rubric.md:1", "score": 0.92}],
                "status": "answered",
            }
        }


@dataclass(frozen=True)
class APIError:
    """Structured error response."""

    detail: str
    status_code: int


def validate_request(question: str) -> Optional[APIError]:
    """Validate incoming question."""
    if not question or len(question.strip()) < 3:
        return APIError(detail="Question must be at least 3 characters.", status_code=400)
    if len(question) > 1000:
        return APIError(detail="Question must not exceed 1000 characters.", status_code=400)
    return None


def query_rag(
    request: QueryRequest,
    pipeline_fn: Callable[[str], tuple[str, list[dict[str, Any]]]],
) -> QueryResponse:
    """Call RAG pipeline and return structured response."""
    # Validate input
    validation_error = validate_request(request.question)
    if validation_error:
        raise ValueError(validation_error.detail)

    try:
        # Call RAG pipeline
        answer, sources_list = pipeline_fn(request.question)

        # Convert sources to response model
        sources = [
            Source(
                source=source.get("source", "unknown"),
                chunk_id=source.get("chunk_id"),
                score=source.get("score"),
            )
            for source in sources_list
        ]

        status = "answered" if sources else "refused_weak_context"

        return QueryResponse(answer=answer, sources=sources, status=status)

    except ValueError as error:
        raise ValueError(f"Validation error: {str(error)}")
    except Exception as error:
        raise RuntimeError(f"RAG service failed: {str(error)}")


def load_config() -> dict[str, str]:
    """Load and return configuration from environment."""
    return {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "VECTOR_DB_URL": VECTOR_DB_URL,
        "COLLECTION_NAME": COLLECTION_NAME,
    }


def main() -> int:
    """Demo: show config loading, request validation, and response generation."""
    print("=== Backend RAG API Configuration ===")
    config = load_config()
    for key, value in config.items():
        print(f"{key}: {value}")

    print("\n=== Request Validation ===")
    # Valid request
    try:
        req = QueryRequest(question="What evidence is required for project submission?")
        print(f"✓ Valid request: {req.question[:50]}...")
    except Exception as e:
        print(f"✗ Validation failed: {e}")

    # Invalid request (too short)
    try:
        req = QueryRequest(question="ab")
        print(f"✓ Valid request: {req.question}")
    except Exception as e:
        print(f"✓ Caught invalid request: too short")

    print("\n=== Response Generation ===")
    # Demo pipeline function
    def demo_pipeline(question: str) -> tuple[str, list[dict[str, Any]]]:
        if "evidence" in question.lower():
            return (
                "The submission requires a PR link, sample output, and a video explanation.",
                [
                    {"source": "submission-rubric.md", "chunk_id": "submission-rubric.md:1", "score": 0.92},
                    {"source": "submission-rubric.md", "chunk_id": "submission-rubric.md:2", "score": 0.85},
                ],
            )
        elif "refuse" in question.lower():
            return ("I don't have enough information.", [])
        else:
            return ("Query processed.", [])

    try:
        req = QueryRequest(question="What evidence is required for project submission?")
        response = query_rag(req, demo_pipeline)
        print(f"Answer: {response.answer}")
        print(f"Sources: {len(response.sources)} retrieved")
        for src in response.sources:
            print(f"  - {src.source} (score: {src.score})")
        print(f"Status: {response.status}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("\n=== Empty Response Handling ===")
    try:
        req = QueryRequest(question="What happens when I ask about refusing?")
        response = query_rag(req, demo_pipeline)
        print(f"Answer: {response.answer}")
        print(f"Sources: {len(response.sources)} retrieved")
        print(f"Status: {response.status}")
    except Exception as e:
        print(f"✗ Error: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
