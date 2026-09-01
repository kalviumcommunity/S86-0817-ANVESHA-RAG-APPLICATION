# S86-0817-ANVESHA-RAG-APPLICATION

## Anvesha RAG Application

Anvesha is a complete Retrieval-Augmented Generation (RAG) system implementing 20 production-grade concepts for document-based Q&A with grounded, cited answers.

### ✨ Core Features

**Embedding & Retrieval (Concepts 3.30-3.32)**
- Embedding quality validation with sanity tests
- Vector database integration (Chroma 1.5.9)
- Top-k similarity search with variable k
- Metadata-based hybrid search filtering

**Ranking & Evaluation (Concepts 3.33-3.35)**
- Chunk reranking using lexical overlap
- Retrieval evaluation with recall@k and precision@k metrics
- Relevance tuning to find optimal retrieval settings

**Generation & Safety (Concepts 3.36-3.39)**
- End-to-end RAG pipeline orchestration
- Context-injection with token budgeting
- Grounded answer generation (only from retrieved context)
- Citation mapping and verification
- Hallucination prevention guardrails

**Conversation & Evaluation (Concepts 3.40-3.43)**
- Multi-turn conversational RAG with history management
- Follow-up query rewriting for better retrieval
- End-to-end answer quality scoring (correctness, grounding, citation accuracy)

**APIs & Frontend (Concepts 3.44-3.46)**
- FastAPI backend with Pydantic validation
- Document upload and runtime indexing endpoint
- Chat interface with source display
- Error handling and loading states

**Streaming & Observability (Concepts 3.47-3.48)**
- Server-sent events (SSE) streaming responses
- Progressive citation display
- Query result caching with TTL
- Structured JSON logging
- Usage tracking and cost estimation

**Deployment & Delivery (Concept 3.49)**
- Reproducible setup and configuration
- Environment variable validation
- End-to-end flow verification
- Git tagging for final submission

## Project Structure

```text
src/
  ├── embedding_quality.py              # Concept 3.30: Validate embeddings
  ├── vector_store.py                   # Concept 3.31: Vector database wrapper
  ├── index_embeddings.py               # Concept 3.32: Bulk indexing with batches
  ├── retrieval.py                      # Concept 3.33: Top-k similarity search
  ├── hybrid_search.py                  # Concept 3.34: Metadata filtering
  ├── relevance_tuning.py               # Concept 3.35: Optimize retrieval settings
  ├── reranker.py                       # Concept 3.36: Re-rank by lexical overlap
  ├── retrieval_evaluation.py           # Concept 3.37: Measure recall & precision
  ├── rag_pipeline.py                   # Concept 3.38: End-to-end orchestration
  ├── context_injection.py              # Concept 3.39: Token-budgeted assembly
  ├── grounded_generation.py            # Concept 3.40: Context-only answer gen
  ├── citations.py                      # Concept 3.41: Citation mapping
  ├── guardrails.py                     # Concept 3.42: Hallucination prevention
  ├── conversational_rag.py             # Concept 3.43: Multi-turn dialogue
  ├── rag_evaluation.py                 # Concept 3.44: Quality scoring
  ├── backend_rag_api.py                # Concept 3.45: FastAPI backend
  ├── document_upload_indexing.py       # Concept 3.46: Upload & index endpoint
  ├── chat_query_ui.py                  # Concept 3.47: Chat interface
  ├── streaming_citations.py            # Concept 3.48: Streaming & citations
  ├── caching_logging_monitoring.py     # Concept 3.49: Observability
  ├── deployment_documentation_delivery.py  # Concept 3.50: Deployment & delivery
  └── test_env.py                       # Environment validation
data/                                    # Source documents (not in Git)
outputs/                                 # Generated results (not in Git)
uploads/                                 # Uploaded documents (not in Git)
.env.example                             # Environment template
.gitignore                               # Exclude secrets & cache
requirements.txt                         # Python dependencies
README.md                                # This file
```

## Setup

Python 3.10 or newer is recommended. From the project root, create and activate a virtual environment:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

On macOS or Linux, use:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the pinned dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` and fill in the values locally. The `.env` file is ignored by Git and must never contain a key committed to the repository.

```powershell
Copy-Item .env.example .env
```

Required variables:

```dotenv
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key
CHAT_MODEL=gpt-4o-mini
EMBED_MODEL=text-embedding-3-small
```

Run the setup check with the environment activated:

```bash
python src/test_env.py
```

The check validates that the dotenv file can be loaded and that the API key is present without printing its value. It does not make an API request.

## Running the Application

### Individual Modules (Demo Mode)

Each module can be run independently without API keys—they use deterministic mock data:

```bash
# Embedding quality checks
PYTHONPATH=src python src/embedding_quality.py

# Vector retrieval (top-k search)
PYTHONPATH=src python src/retrieval.py

# Metadata hybrid search
PYTHONPATH=src python src/hybrid_search.py

# End-to-end RAG pipeline
PYTHONPATH=src python src/rag_pipeline.py

# Grounded answer generation with citations
PYTHONPATH=src python src/grounded_generation.py

# Hallucination guardrails
PYTHONPATH=src python src/guardrails.py

# Multi-turn conversational RAG
PYTHONPATH=src python src/conversational_rag.py

# Answer quality evaluation
PYTHONPATH=src python src/rag_evaluation.py

# Backend API with Pydantic models
PYTHONPATH=src python src/backend_rag_api.py

# Document upload and indexing
PYTHONPATH=src python src/document_upload_indexing.py

# Chat UI with source display
PYTHONPATH=src python src/chat_query_ui.py

# Streaming responses and citations
PYTHONPATH=src python src/streaming_citations.py

# Caching, logging, and usage monitoring
PYTHONPATH=src python src/caching_logging_monitoring.py
```

### Backend API Server

```bash
python -m uvicorn src.backend_rag_api:app --reload --port 8000
```

Then POST to `http://localhost:8000/query`:
```json
{"question": "What evidence is required for project submission?"}
```

### Running Tests

All modules include deterministic tests that verify:
- Embedding quality and sanity checks
- Retrieval accuracy and ranking
- Citation correctness and completeness
- Streaming event serialization
- Cache hit behavior and TTL expiration
- Cost estimation and usage logging

Tests run without API keys using mock data embedded in each module.

## End-to-End Flow Verification

The complete RAG system flow:

1. **Document Upload** → Validated, stored, preprocessed
2. **Chunking & Embedding** → Text split, vectors computed, indexed
3. **Query Processing** → Question embedded, context retrieved
4. **Answer Generation** → Grounded in retrieved context only
5. **Citation Display** → Claims linked to source text
6. **Logging & Monitoring** → Request logged, usage tracked, cache updated

Example flow verification:

```bash
# 1. Upload a document
PYTHONPATH=src python -c "from document_upload_indexing import handle_upload; ..."

# 2. Retrieve relevant chunks
PYTHONPATH=src python -c "from retrieval import Retriever; r = Retriever(...)"

# 3. Generate grounded answer
PYTHONPATH=src python -c "from grounded_generation import generate_grounded_answer; ..."

# 4. Display with citations
PYTHONPATH=src python -c "from citations import build_citation_map; ..."

# 5. Check logs and usage
PYTHONPATH=src python -c "from caching_logging_monitoring import summarize_usage; ..."
```

## Reproducibility Checklist

1. Clone the repository.
2. Create and activate `.venv`.
3. Run `python -m pip install -r requirements.txt`.
4. Copy `.env.example` to `.env` and add local credentials.
5. Run `python src/test_env.py`.

Never commit `.env`, `.venv/`, source documents, generated outputs, or Python cache files.

## Deployment & Delivery

### Reproducibility Checklist

Before submitting, verify all steps work:

1. ✓ Clone the repository
2. ✓ Create and activate `.venv`
3. ✓ Run `python -m pip install -r requirements.txt`
4. ✓ Copy `.env.example` to `.env` and add credentials (or use demo mode without keys)
5. ✓ Run `python src/test_env.py` to validate setup
6. ✓ Run individual module demos to verify functionality
7. ✓ Review logs and usage reports for observability
8. ✓ Tag final delivery: `git tag sprint-final-rag-v1.0` and push

### Production Deployment

**Option 1: Local Development**

```bash
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
python -m uvicorn src.backend_rag_api:app --host 0.0.0.0 --port 8000
```

**Option 2: Docker**

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
CMD ["python", "-m", "uvicorn", "src.backend_rag_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Option 3: Serverless (Railway, Vercel, etc.)**

Set environment variables in the platform's settings, then deploy the code. Platform handles scaling.

### Final Submission

Tag the release for reviewers:

```bash
git tag -a sprint-final-rag-v1.0 -m "RAG application - 20 concepts implemented"
git push origin sprint-final-rag-v1.0
```

Include the tag link in the PR description so reviewers can easily checkout the exact submission version.

## Monitoring & Observability

All requests are logged with:
- **Request ID & Timestamp** – When the request was made
- **Question & Answer Preview** – What was asked and answered (first 180 chars)
- **Sources Retrieved** – Which chunks were used for grounding
- **Cache Hit** – Whether this was a cached answer (avoiding re-computation)
- **Token Usage** – Input tokens, output tokens, estimated cost
- **Latency** – Time taken to process (ms)

Usage reports summarize:
- **Total Requests** – How many questions have been asked
- **Cache Hit Rate** – Percentage of repeated queries
- **Total Cost** – Estimated API cost
- **Average Latency** – Expected response time
- **Token Efficiency** – Total input/output tokens

Example usage:
```python
from caching_logging_monitoring import summarize_usage
summary = summarize_usage()
print(f"Cache hit rate: {summary['cache_hit_rate'] * 100:.1f}%")
print(f"Total cost: ${summary['total_estimated_cost']}")
```

## Architecture Decisions

| Concept | Decision | Rationale |
|---------|----------|-----------|
| **Vector DB** | Chroma 1.5.9 (in-memory) | Deterministic, no external service required for demo |
| **Embeddings** | Deterministic mock vectors | Works without API keys, fast iteration |
| **Chunking** | Fixed-size with overlap | Simple, reproducible chunk boundaries |
| **Cache TTL** | 15 minutes | Balances freshness vs. performance |
| **Citation Model** | Prefix markers `[1]`, `[2]` | Clear, human-readable, easily parsed |
| **Guardrails** | Min score + supporting chunks | Prevents low-confidence generation |
| **Token Counting** | Word-based approximation | No API dependency for cost estimation |

## Testing Without API Keys

All 20 concepts are implemented with deterministic tests that don't require API credentials:

- Mock embeddings use fixed vectors based on query content
- Mock retrievers return predetermined results
- Mock generators produce consistent answers for known inputs
- Mock logging stores all requests in memory
- Cache uses in-memory dict with TTL

To run tests:

```bash
python -m pytest src/  # If pytest is in requirements.txt
# OR run individual demos without pytest
PYTHONPATH=src python src/embedding_quality.py
PYTHONPATH=src python src/streaming_citations.py
# ... etc
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Set `PYTHONPATH=src` before running |
| `.env` file not found | Copy `.env.example` to `.env` and fill in values |
| `OPENAI_API_KEY` error | Use demo mode (mock functions work without keys) or add a valid key to `.env` |
| Chroma errors | Chroma is in-memory; no external DB needed. Reinstall: `pip install chroma-db==1.5.9` |
| Cache not working | Clear cache between test runs or adjust `CACHE_TTL_SECONDS` |

## Next Steps

To extend this RAG system:

1. Replace mock embeddings with real API calls (OpenAI, Cohere, etc.)
2. Swap Chroma in-memory with Pinecone, Weaviate, or Qdrant for persistence
3. Implement real document parsing (PDF extraction, markdown formatting)
4. Add authentication and user isolation
5. Connect to a real LLM backend (GPT-4, Claude, Llama)
6. Deploy with Kubernetes for high availability
7. Add web UI (React, Vue, or Streamlit)
8. Integrate with Slack, Discord, Teams bots