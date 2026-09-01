"""Caching, logging, and usage monitoring for RAG system observability."""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Optional


# Configuration
CACHE_TTL_SECONDS = 15 * 60  # 15 minutes
MODEL_INPUT_COST_PER_1K = 0.00015
MODEL_OUTPUT_COST_PER_1K = 0.00060


# Global cache
query_cache: dict[str, dict[str, Any]] = {}
log_records: list[dict[str, Any]] = []


@dataclass(frozen=True)
class CacheEntry:
    """A cached query response."""

    created_at: float
    response: dict[str, Any]


@dataclass(frozen=True)
class UsageMetadata:
    """Usage information for a request."""

    input_tokens: int
    output_tokens: int
    estimated_cost: float
    cache_hit: bool


@dataclass(frozen=True)
class RequestLog:
    """Structured log entry for a RAG request."""

    timestamp: str
    request_id: str
    question: str
    answer_preview: str
    sources: list[str]
    cache_hit: bool
    input_tokens: int
    output_tokens: int
    estimated_cost: float
    latency_ms: float


def cache_key(question: str, filters: Optional[dict[str, Any]] = None) -> str:
    """Generate a cache key for a question and optional filters."""
    raw = {
        "question": question.strip().lower(),
        "filters": filters or {},
    }
    raw_str = json.dumps(raw, sort_keys=True)
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()


def get_cached_answer(question: str, filters: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    """Retrieve a cached answer if it exists and hasn't expired."""
    key = cache_key(question, filters)
    cached = query_cache.get(key)

    if not cached:
        return None

    # Check TTL
    age = time.time() - cached["created_at"]
    if age > CACHE_TTL_SECONDS:
        query_cache.pop(key, None)
        return None

    return cached["response"]


def save_cached_answer(
    question: str,
    response: dict[str, Any],
    filters: Optional[dict[str, Any]] = None,
) -> None:
    """Save an answer to the cache."""
    key = cache_key(question, filters)
    query_cache[key] = {
        "created_at": time.time(),
        "response": response,
    }


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate the cost of a request based on token usage."""
    input_cost = (input_tokens / 1000) * MODEL_INPUT_COST_PER_1K
    output_cost = (output_tokens / 1000) * MODEL_OUTPUT_COST_PER_1K
    return round(input_cost + output_cost, 6)


def log_rag_request(record: dict[str, Any]) -> None:
    """Log a structured RAG request to the log records."""
    log_entry = RequestLog(
        timestamp=record.get("timestamp", datetime.utcnow().isoformat()),
        request_id=record.get("request_id", str(uuid.uuid4())),
        question=record.get("question", ""),
        answer_preview=record.get("answer", "")[:180],
        sources=record.get("sources", []),
        cache_hit=record.get("cache_hit", False),
        input_tokens=record.get("input_tokens", 0),
        output_tokens=record.get("output_tokens", 0),
        estimated_cost=record.get("estimated_cost", 0.0),
        latency_ms=record.get("latency_ms", 0.0),
    )
    log_records.append(asdict(log_entry))


def summarize_usage(records: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    """Generate a usage summary from log records."""
    if records is None:
        records = log_records

    if not records:
        return {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_hit_rate": 0.0,
            "total_estimated_cost": 0.0,
            "average_latency_ms": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }

    total_requests = len(records)
    cache_hits = sum(1 for item in records if item.get("cache_hit", False))
    total_cost = sum(item.get("estimated_cost", 0.0) for item in records)
    total_latency = sum(item.get("latency_ms", 0.0) for item in records)
    total_input_tokens = sum(item.get("input_tokens", 0) for item in records)
    total_output_tokens = sum(item.get("output_tokens", 0) for item in records)

    return {
        "total_requests": total_requests,
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / max(total_requests, 1), 2),
        "total_estimated_cost": round(total_cost, 6),
        "average_latency_ms": round(total_latency / max(total_requests, 1), 2),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }


def process_rag_request(
    question: str,
    generate_answer_fn,
    filters: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], UsageMetadata]:
    """Process a RAG request with caching and logging."""
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Check cache
    cached_answer = get_cached_answer(question, filters)
    cache_hit = cached_answer is not None

    if cache_hit:
        answer_data = cached_answer
        input_tokens = 0
        output_tokens = len(cached_answer.get("answer", "").split())
    else:
        # Generate answer
        answer_data = generate_answer_fn(question)
        input_tokens = len(question.split())
        output_tokens = len(answer_data.get("answer", "").split())

        # Save to cache
        save_cached_answer(question, answer_data, filters)

    # Calculate latency and cost
    latency_ms = (time.time() - start_time) * 1000
    estimated_cost = estimate_cost(input_tokens, output_tokens)

    # Log the request
    log_rag_request(
        {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "answer": answer_data.get("answer", ""),
            "sources": answer_data.get("sources", []),
            "cache_hit": cache_hit,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost": estimated_cost,
            "latency_ms": latency_ms,
        }
    )

    usage = UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost=estimated_cost,
        cache_hit=cache_hit,
    )

    return answer_data, usage


def main() -> int:
    """Demo: show caching, logging, usage tracking, and reporting."""
    print("=== Caching, Logging & Usage Monitoring ===\n")

    # Mock answer generator
    def mock_generator(question: str) -> dict[str, Any]:
        """Simulate a RAG answer generator."""
        if "evidence" in question.lower():
            return {
                "answer": "Project submissions require a PR link, sample output, and video explanation.",
                "sources": ["submission-rubric.md"],
            }
        else:
            return {
                "answer": "I don't have specific information about that topic.",
                "sources": [],
            }

    print("=== Test 1: Cache Key Generation ===")
    key1 = cache_key("What evidence is required?")
    key2 = cache_key("What evidence is required?")  # Same question
    key3 = cache_key("What evidence?")  # Different question
    print(f"Key 1 (full): {key1[:16]}...")
    print(f"Key 2 (same): {key2[:16]}...")
    print(f"Match: {key1 == key2}")
    print(f"Key 3 (diff): {key3[:16]}...")
    print(f"Different: {key1 != key3}")

    print("\n=== Test 2: Caching & Cache Hits ===")
    # First request (cache miss)
    answer1, usage1 = process_rag_request("What evidence is required?", mock_generator)
    print(f"Request 1:")
    print(f"  Cache hit: {usage1.cache_hit}")
    print(f"  Tokens: in={usage1.input_tokens}, out={usage1.output_tokens}")
    print(f"  Cost: ${usage1.estimated_cost}")

    # Second request (cache hit)
    answer2, usage2 = process_rag_request("What evidence is required?", mock_generator)
    print(f"Request 2 (same question):")
    print(f"  Cache hit: {usage2.cache_hit}")
    print(f"  Tokens: in={usage2.input_tokens}, out={usage2.output_tokens}")
    print(f"  Cost: ${usage2.estimated_cost}")

    # Third request (cache miss)
    answer3, usage3 = process_rag_request("What is the policy?", mock_generator)
    print(f"Request 3 (different question):")
    print(f"  Cache hit: {usage3.cache_hit}")
    print(f"  Tokens: in={usage3.input_tokens}, out={usage3.output_tokens}")
    print(f"  Cost: ${usage3.estimated_cost}")

    print("\n=== Test 3: Cost Estimation ===")
    cost1 = estimate_cost(100, 50)
    cost2 = estimate_cost(1000, 500)
    print(f"100 input, 50 output tokens: ${cost1}")
    print(f"1000 input, 500 output tokens: ${cost2}")

    print("\n=== Test 4: Structured Logging ===")
    print(f"Log records collected: {len(log_records)}")
    if log_records:
        first_log = log_records[0]
        print(f"First log entry:")
        print(f"  Request ID: {first_log['request_id']}")
        print(f"  Question: {first_log['question'][:40]}...")
        print(f"  Cache hit: {first_log['cache_hit']}")
        print(f"  Latency: {first_log['latency_ms']:.2f}ms")

    print("\n=== Test 5: Usage Summary ===")
    summary = summarize_usage()
    print(f"Total requests: {summary['total_requests']}")
    print(f"Cache hits: {summary['cache_hits']}/{summary['total_requests']}")
    print(f"Cache hit rate: {summary['cache_hit_rate'] * 100:.1f}%")
    print(f"Total cost: ${summary['total_estimated_cost']}")
    print(f"Average latency: {summary['average_latency_ms']:.2f}ms")
    print(f"Total tokens (input): {summary['total_input_tokens']}")
    print(f"Total tokens (output): {summary['total_output_tokens']}")

    print("\n=== Test 6: Cache Expiration ===")
    # Artificially expire a cache entry
    key = cache_key("What evidence is required?")
    if key in query_cache:
        query_cache[key]["created_at"] = time.time() - (CACHE_TTL_SECONDS + 1)
    
    expired_answer = get_cached_answer("What evidence is required?")
    print(f"After TTL expiration: {expired_answer is None}")
    print(f"Cache entries remaining: {len(query_cache)}")

    print("\n=== Test 7: Log Export ===")
    print("Sample log entry (JSON):")
    if log_records:
        print(json.dumps(log_records[0], indent=2))

    print("\n=== Test 8: Multiple Requests Report ===")
    # Simulate more requests
    questions = [
        "What evidence is required?",
        "What is the policy?",
        "What evidence is required?",  # Cache hit
        "What is the campus like?",
        "What evidence is required?",  # Cache hit
    ]
    
    for q in questions:
        process_rag_request(q, mock_generator)
    
    final_summary = summarize_usage()
    print(f"Final summary after {final_summary['total_requests']} requests:")
    print(f"  Cache hit rate: {final_summary['cache_hit_rate'] * 100:.1f}%")
    print(f"  Total cost: ${final_summary['total_estimated_cost']}")
    print(f"  Avg latency: {final_summary['average_latency_ms']:.2f}ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
