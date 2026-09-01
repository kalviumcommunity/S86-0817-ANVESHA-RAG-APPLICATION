"""Chat interface and query UI for the RAG backend."""

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class Source:
    """A single source from the RAG response."""

    source: str
    chunk_id: Optional[str] = None
    score: Optional[float] = None


@dataclass(frozen=True)
class Answer:
    """Answer response from the RAG backend."""

    answer: str
    sources: list[Source]
    status: str


@dataclass(frozen=True)
class UIState:
    """UI state for a single query interaction."""

    question: Optional[str] = None
    loading: bool = False
    answer: Optional[Answer] = None
    error: Optional[str] = None


def ask_question(
    question: str,
    api_client: Callable[[str], dict[str, Any]],
) -> Answer:
    """Send a question to the backend and parse the response."""
    if not question or len(question.strip()) < 3:
        raise ValueError("Question must be at least 3 characters")

    response_data = api_client(question)

    if not response_data:
        raise RuntimeError("Empty response from RAG API")

    sources = [
        Source(
            source=src.get("source"),
            chunk_id=src.get("chunk_id"),
            score=src.get("score"),
        )
        for src in response_data.get("sources", [])
    ]

    return Answer(
        answer=response_data.get("answer", ""),
        sources=sources,
        status=response_data.get("status", "answered"),
    )


def handle_submit(
    question: str,
    api_client: Callable[[str], dict[str, Any]],
) -> UIState:
    """Handle a question submission: validate, call API, return state."""
    state = UIState(question=question, loading=True)

    try:
        # Call the API
        answer = ask_question(question, api_client)
        state = UIState(question=question, loading=False, answer=answer)
    except ValueError as e:
        state = UIState(question=question, loading=False, error=f"Input error: {str(e)}")
    except Exception as e:
        state = UIState(question=question, loading=False, error=f"Could not get an answer: {str(e)}")

    return state


def format_sources(sources: list[Source]) -> str:
    """Format sources for display."""
    if not sources:
        return "No sources retrieved"

    lines = []
    for i, source in enumerate(sources, 1):
        source_info = f"{i}. {source.source}"
        if source.chunk_id:
            source_info += f" ({source.chunk_id})"
        if source.score is not None:
            source_info += f" [score: {source.score:.2f}]"
        lines.append(source_info)
    return "\n".join(lines)


def format_answer(answer: Answer) -> str:
    """Format answer and sources for display."""
    output = []
    output.append("=== Answer ===")
    output.append(answer.answer)
    output.append("")
    output.append("=== Sources ===")
    output.append(format_sources(answer.sources))
    output.append("")
    output.append(f"Status: {answer.status}")
    return "\n".join(output)


def render_ui_state(state: UIState) -> str:
    """Render the current UI state as formatted text."""
    if state.loading:
        return "[Loading...] Processing your question..."

    if state.error:
        return f"[Error] {state.error}"

    if state.answer:
        return format_answer(state.answer)

    return "[Ready] Ask a question about the knowledge base."


def main() -> int:
    """Demo: show chat UI with API calls, loading states, error handling, and answer display."""
    print("=== Chat & Query UI Demo ===\n")

    # Mock API client
    def mock_api(question: str) -> dict[str, Any]:
        """Simulate a backend RAG API response."""
        if "evidence" in question.lower():
            return {
                "answer": "The submission requires a PR link, sample output, and a video explanation.",
                "sources": [
                    {"source": "submission-rubric.md", "chunk_id": "submission-rubric.md:1", "score": 0.92},
                    {"source": "submission-rubric.md", "chunk_id": "submission-rubric.md:2", "score": 0.87},
                ],
                "status": "answered",
            }
        elif "refuse" in question.lower():
            return {
                "answer": "I don't have enough information in the knowledge base to answer that.",
                "sources": [],
                "status": "refused_weak_context",
            }
        elif "error" in question.lower():
            raise RuntimeError("Simulated API error")
        else:
            return {
                "answer": f"I can help with that question about '{question.split()[-1]}'.",
                "sources": [{"source": "general.md", "chunk_id": "general.md:0", "score": 0.75}],
                "status": "answered",
            }

    print("=== Test 1: Valid Question ===")
    state = handle_submit("What evidence is required for submission?", mock_api)
    print(render_ui_state(state))

    print("\n=== Test 2: Question With No Sources ===")
    state = handle_submit("What happens when I refuse?", mock_api)
    print(render_ui_state(state))

    print("\n=== Test 3: Invalid Question (Too Short) ===")
    state = handle_submit("ab", mock_api)
    print(render_ui_state(state))

    print("\n=== Test 4: API Error Handling ===")
    state = handle_submit("This will cause an error", mock_api)
    print(render_ui_state(state))

    print("\n=== Test 5: Loading State Simulation ===")
    print(render_ui_state(UIState(question="test", loading=True)))

    print("\n=== Test 6: Source Display ===")
    sources = [
        Source(source="doc1.md", chunk_id="doc1:0", score=0.95),
        Source(source="doc2.txt", chunk_id="doc2:5", score=0.88),
        Source(source="doc3.md", score=0.72),
    ]
    answer = Answer(answer="Test answer", sources=sources, status="answered")
    print(format_answer(answer))

    print("\n=== Test 7: Chat Interaction Flow ===")
    questions = [
        "What evidence is required for submission?",
        "Tell me about project guidelines",
        "error",
    ]
    for q in questions:
        print(f"\n>>> User: {q}")
        state = handle_submit(q, mock_api)
        print(render_ui_state(state))
        if state.answer:
            print(f"Sources retrieved: {len(state.answer.sources)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
