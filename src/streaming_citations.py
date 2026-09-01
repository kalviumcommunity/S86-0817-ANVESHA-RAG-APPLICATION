"""Streaming responses and citation display for real-time answer generation."""

import json
from dataclasses import dataclass
from typing import Any, Generator, Optional


# Stream event types
@dataclass(frozen=True)
class StreamToken:
    """A token event from the stream."""

    type: str = "token"
    text: str = ""


@dataclass(frozen=True)
class Citation:
    """A single citation with source metadata."""

    id: str
    label: str
    document: str
    chunk_id: str
    text: str


@dataclass(frozen=True)
class CitationsEvent:
    """A citations event from the stream."""

    type: str = "citations"
    sources: list[Citation] = None

    def __post_init__(self):
        if self.sources is None:
            object.__setattr__(self, "sources", [])


@dataclass(frozen=True)
class DoneEvent:
    """A done event signaling stream completion."""

    type: str = "done"


@dataclass(frozen=True)
class ErrorEvent:
    """An error event from the stream."""

    type: str = "error"
    message: str = ""


@dataclass(frozen=True)
class StreamState:
    """UI state during streaming."""

    answer: str = ""
    citations: list[Citation] = None
    streaming: bool = False
    error: Optional[str] = None
    partial: bool = False

    def __post_init__(self):
        if self.citations is None:
            object.__setattr__(self, "citations", [])


def serialize_event(event: Any) -> str:
    """Serialize an event to JSON for streaming."""
    if isinstance(event, StreamToken):
        return json.dumps({"type": "token", "text": event.text})
    elif isinstance(event, CitationsEvent):
        return json.dumps(
            {
                "type": "citations",
                "sources": [
                    {
                        "id": c.id,
                        "label": c.label,
                        "document": c.document,
                        "chunk_id": c.chunk_id,
                        "text": c.text,
                    }
                    for c in event.sources
                ],
            }
        )
    elif isinstance(event, DoneEvent):
        return json.dumps({"type": "done"})
    elif isinstance(event, ErrorEvent):
        return json.dumps({"type": "error", "message": event.message})
    else:
        return json.dumps({"type": "unknown"})


def deserialize_event(data: str) -> Optional[Any]:
    """Deserialize an event from JSON."""
    try:
        obj = json.loads(data)
        event_type = obj.get("type")

        if event_type == "token":
            return StreamToken(text=obj.get("text", ""))
        elif event_type == "citations":
            sources = [
                Citation(
                    id=s.get("id"),
                    label=s.get("label"),
                    document=s.get("document"),
                    chunk_id=s.get("chunk_id"),
                    text=s.get("text"),
                )
                for s in obj.get("sources", [])
            ]
            return CitationsEvent(sources=sources)
        elif event_type == "done":
            return DoneEvent()
        elif event_type == "error":
            return ErrorEvent(message=obj.get("message", ""))
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def rag_pipeline_stream(question: str) -> Generator[Any, None, None]:
    """Simulate a streaming RAG pipeline that yields events."""
    if "refuse" in question.lower():
        # No citations, short answer
        yield ErrorEvent(message="I don't have enough information to answer that.")
        return

    # First, yield citations
    citations = [
        Citation(
            id="source-1",
            label="[1]",
            document="submission-rubric.md",
            chunk_id="submission-rubric-03",
            text="Submissions must include: PR link, sample output, and video explanation.",
        ),
        Citation(
            id="source-2",
            label="[2]",
            document="submission-rubric.md",
            chunk_id="submission-rubric-05",
            text="Video explanations should be under 5 minutes and include a code walkthrough.",
        ),
    ]
    yield CitationsEvent(sources=citations)

    # Then stream tokens
    answer_tokens = [
        "The ",
        "submission ",
        "requires ",
        "[1] ",
        "a ",
        "PR ",
        "link, ",
        "sample ",
        "output, ",
        "and ",
        "video ",
        "explanation. ",
        "[2] ",
        "Videos ",
        "should ",
        "be ",
        "under ",
        "5 ",
        "minutes.",
    ]

    for token in answer_tokens:
        yield StreamToken(text=token)

    # Signal completion
    yield DoneEvent()


def process_stream_events(
    events: list[Any],
) -> tuple[str, list[Citation], Optional[str]]:
    """Process a sequence of stream events into final answer and citations."""
    state = StreamState()

    for event in events:
        if isinstance(event, StreamToken):
            state = StreamState(
                answer=state.answer + event.text,
                citations=state.citations,
                streaming=False,
                error=state.error,
                partial=False,
            )
        elif isinstance(event, CitationsEvent):
            state = StreamState(
                answer=state.answer,
                citations=event.sources,
                streaming=False,
                error=state.error,
                partial=False,
            )
        elif isinstance(event, ErrorEvent):
            state = StreamState(
                answer=state.answer,
                citations=state.citations,
                streaming=False,
                error=event.message,
                partial=True,
            )
        elif isinstance(event, DoneEvent):
            state = StreamState(
                answer=state.answer,
                citations=state.citations,
                streaming=False,
                error=state.error,
                partial=False,
            )

    return state.answer, state.citations, state.error


def format_answer_with_citations(answer: str, citations: list[Citation]) -> str:
    """Format answer with inline citation markers."""
    lines = []
    lines.append("=== Answer ===")
    lines.append(answer)
    lines.append("")
    lines.append("=== Sources ===")

    if citations:
        for citation in citations:
            lines.append(f"{citation.label} {citation.document} ({citation.chunk_id})")
            lines.append(f"   {citation.text}")
            lines.append("")
    else:
        lines.append("No sources retrieved")

    return "\n".join(lines)


def simulate_streaming_ui(question: str) -> str:
    """Simulate streaming to UI progressively."""
    output = []
    output.append(f">>> User: {question}")
    output.append("[Streaming...]")

    state = StreamState(streaming=True)
    events = list(rag_pipeline_stream(question))

    # Process events progressively
    for i, event in enumerate(events):
        if isinstance(event, StreamToken):
            state = StreamState(
                answer=state.answer + event.text,
                citations=state.citations,
                streaming=i < len(events) - 1,
                error=state.error,
                partial=False,
            )
            output.append(f"[Token: {repr(event.text)}] Answer so far: {state.answer[:50]}...")

        elif isinstance(event, CitationsEvent):
            state = StreamState(
                answer=state.answer,
                citations=event.sources,
                streaming=i < len(events) - 1,
                error=state.error,
                partial=False,
            )
            output.append(f"[Citations received: {len(event.sources)} sources]")

        elif isinstance(event, ErrorEvent):
            state = StreamState(
                answer=state.answer,
                citations=state.citations,
                streaming=False,
                error=event.message,
                partial=True,
            )
            output.append(f"[Error: {event.message}]")

        elif isinstance(event, DoneEvent):
            state = StreamState(
                answer=state.answer,
                citations=state.citations,
                streaming=False,
                error=state.error,
                partial=False,
            )
            output.append("[Done]")

    output.append("")
    output.append(format_answer_with_citations(state.answer, state.citations))

    return "\n".join(output)


def main() -> int:
    """Demo: show streaming events, progressive updates, and citation display."""
    print("=== Streaming Responses & Citation Display ===\n")

    print("=== Test 1: Stream Event Serialization ===")
    token_event = StreamToken(text="Hello ")
    citations_event = CitationsEvent(
        sources=[
            Citation(
                id="1",
                label="[1]",
                document="test.md",
                chunk_id="test:1",
                text="Test text",
            )
        ]
    )
    done_event = DoneEvent()

    print(f"Token: {serialize_event(token_event)}")
    print(f"Citations: {serialize_event(citations_event)[:60]}...")
    print(f"Done: {serialize_event(done_event)}")

    print("\n=== Test 2: Stream Event Deserialization ===")
    token_json = '{"type": "token", "text": "Hello"}'
    deserialized = deserialize_event(token_json)
    assert isinstance(deserialized, StreamToken)
    print(f"✓ Deserialized token: {deserialized.text}")

    print("\n=== Test 3: Progressive Streaming ===")
    print(simulate_streaming_ui("What evidence is required for submission?"))

    print("\n=== Test 4: Error Handling in Stream ===")
    print(simulate_streaming_ui("What happens when I refuse?"))

    print("\n=== Test 5: Complete Stream Processing ===")
    events = list(rag_pipeline_stream("What evidence is required?"))
    answer, citations, error = process_stream_events(events)
    print(f"Final answer length: {len(answer)} chars")
    print(f"Citations received: {len(citations)}")
    print(f"Error: {error}")
    print(f"Answer preview: {answer[:50]}...")

    print("\n=== Test 6: Citation Display ===")
    formatted = format_answer_with_citations(answer, citations)
    print(formatted)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
