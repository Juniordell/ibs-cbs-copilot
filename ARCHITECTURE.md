# Architecture

## Overview

RAG copilot answering questions about Brazil's Tax Reform (IBS/CBS), grounded
in official legislation with mandatory article citations.

## Data pipeline (offline)

### Sources

- **LC 214/2025** — main law (IBS, CBS, IS)
- **EC 132/2023** — constitutional foundation
- **Decree 12,955/2026** — CBS regulation

See [docs/download-sources.md](docs/download-sources.md).

### Extraction (`pdf_to_text.py`)

- **PyMuPDF** for text extraction (fast, handles Portuguese well)
- Strips planalto.gov.br headers/footers
- Forces structural markers (`Art.`, `§`, `CAPÍTULO`) onto their own lines
  so the chunker can split cleanly

### Chunking (`chunker.py`)

- **Article-aware**: one chunk per `Art. X`
- Long articles (>2000 chars) subchunk by `§`
- Each chunk carries metadata: `article`, `paragraph`, `item`, `source`
- Why: legal citations require exact reference — losing structure kills the citation feature

### Storage (`ingest.py`)

- **Postgres 16 + pgvector**
- Table `chunks`: text, metadata, `vector(1536)`
- HNSW index (cosine) for vector search
- UNIQUE constraint `(source, article, paragraph, item)` for idempotent re-ingestion
- Embeddings: **OpenAI `text-embedding-3-small`** (cheap, good enough for v1)
- Batches of 100 chunks per API call
- Token safety via `tiktoken` truncation to 8000 tokens

## Query pipeline (online)

## Query pipeline (online)

### Retrieval (`retrieval/`)

Three retrievers, sharing a common `RetrievedChunk` type:

- **`VectorRetriever`** — cosine similarity via pgvector. Strong on paraphrasing
  and semantic questions ("como pagar imposto" → "recolhimento tributário").
  Weak on exact keywords the LLM hasn't seen in the exact form.

- **`BM25Retriever`** — Postgres `ts_rank` over a Portuguese `tsvector`.
  Strong on technical terms and article numbers ("Art. 31", "CBS").
  Weak on paraphrasing.

- **`HybridRetriever`** — runs both, merges via Reciprocal Rank Fusion (RRF).
  Fetches `2 × k` from each side, fuses with `k=60` (standard default), returns
  top `k`. RRF depends only on rank position, not on the retrievers' native
  scores — so vector's [0,1] cosine and BM25's unbounded `ts_rank` combine cleanly.

### Generation (`generation/`)

- **Model:** Claude Sonnet 4.6 via Anthropic SDK (async).
- **Prompts** (`prompts.py`, versioned): system prompt enforces 4 rules —
  strict grounding, mandatory citation, explicit refusal, JSON output.
  New prompt versions become new files (`prompts_v2.py`) so MLflow can compare.
- **Schema** (`schemas.py`): Pydantic `Answer` with `Citation` list.
  Quote capped at 500 chars — prevents the LLM from pasting whole articles.
  `confidence` restricted to `high | medium | low` via `Literal`.
- **Retry:** 2 attempts on `JSONDecodeError` or Pydantic `ValidationError`.
  Network/rate-limit errors bubble up (different handling needed).
- **Markdown-fence stripping:** the model occasionally wraps JSON in ` ```json `.
  Cleaned before parsing.

### Pipeline (`pipeline.py`)

`answer_question(question, k=5) -> PipelineResult`

- Retriever and Generator instantiated once (`@lru_cache`) — cheap reuse across
  requests.
- Returns both the `Answer` and the retrieved `chunks` — Day 6 (Ragas)
  needs the chunks for faithfulness scoring.

### API (`api/`)

- **Framework:** FastAPI with async routes throughout.
- **Lifespan:** `AppState` opens Postgres pool, Redis connection, and builds
  the Generator on startup. Closed on shutdown. Single source of truth for
  shared resources.
- **Endpoints:**
  - `GET /v1/health` — liveness for load balancers.
  - `GET /v1/sources` — explicit list of indexed documents. Transparency:
    users can verify what the copilot is grounded in.
  - `POST /v1/ask` — full pipeline. Cached, rate-limited.
- **Cache** (`cache.py`): Redis, 1h TTL, key = `sha256(top_k:normalized_question)`.
  Same question with different `top_k` cached separately. Corrupt entries treated
  as misses. Cost impact: ~100x reduction on repeated questions.
- **Rate limit:** slowapi, 10 req/min per IP on `/ask` only. Health/sources unlimited.
  Currently keyed by socket IP — day 9 will switch to `X-Forwarded-For` behind Fly.io.
- **Validation:** Pydantic on request (`min_length=5`, `max_length=500` on
  question). FastAPI returns 422 on schema violations before the handler runs.

### Container (`Dockerfile`)

Multi-stage build:

1. **`deps` stage** — installs Poetry + all Python packages. Fat image, not shipped.
2. **`runtime` stage** — copies only the installed packages and `src/`. Slim (~250MB).

The `libpq5` system package is required at runtime for `psycopg` to connect to
Postgres — without it the container can't talk to the DB.

### Local system

`docker-compose.yml` starts three services: `postgres` (pgvector/pg16), `redis`,
and `api`. Postgres and Redis have healthchecks; `api` waits for `service_healthy`
before starting. `.env` values propagate through the `environment:` block.

### Why hybrid

Legal Portuguese mixes technical terms ("split payment", "IBS", "Art. 31")
with natural-language questions. Neither retriever alone handles both cases.
Vector missed "cashback tributário" (concept exists but the exact term is rare in
the chunks); BM25 nailed it. Vector nailed "como funciona o split payment?"
by paraphrase; BM25 missed the subtler articles. Hybrid keeps both wins.

### Not doing (yet)

- **Cross-encoder reranking** — worth trying after Ragas eval on Day 6.
- **Query expansion** — same.
- **HyDE (Hypothetical Document Embeddings)** — same.

Rule: don't add complexity before measuring. Day 6 tells us what to fix.

## Evaluation (`evals/`)

### Golden set (`golden/golden_v1.jsonl`)

30 Portuguese questions covering rates, taxpayers, base, split payment, cashback,
exports, imports, basic goods basket, special regimes, selective tax. Each row
includes:

- `expected_answer_contains`: anchor keywords the answer should mention
- `expected_sources`: articles the citation should reference

Difficulty split: ~10 easy, ~13 medium, ~7 tricky.

The golden set is versioned. Changes to it become `golden_v2.jsonl` etc. —
lets us compare across dataset versions in MLflow.

### Ragas metrics (built-in)

Four LLM-judged metrics on every run:

- **Faithfulness** — is the answer grounded in retrieved chunks?
- **Answer Relevance** — does the answer address the question asked?
- **Context Precision** — are the retrieved chunks useful?
- **Context Recall** — did the retriever find all needed info?

### Citation accuracy (custom, `metrics/citation_accuracy.py`)

Deterministic. Compares `Answer.citations[*].article` against
`expected_sources`. Reports precision / recall / F1 per question, then averages.

Why custom: Ragas doesn't know the copilot's citation contract. Also runs
without any LLM calls — fast, free, safe to run on every commit.

### MLflow tracking

SQLite backend at `mlflow.db`. Each run logs:

- **Params:** `top_k`, `prompt_version`, `golden_size`, `dataset`
- **Metrics:** all 4 Ragas metrics + 3 citation metrics

Runs are named (`v0-baseline`, `v0-topk-3`, `v0-topk-10`) so the UI comparison
tells a story.

### v0 baseline observations

Ragas Faithfulness 0.79 with Recall 0.43 means the answers are grounded but
the retriever misses relevant chunks. Retrieval is the bottleneck, not
generation. First lever to try in v1: bump `top_k`, revisit chunker.

## Observability (`observability/`)

Two layers, different concerns:

### LLM layer (Langfuse)

Every `answer_question` call creates a hierarchical trace:

The `@observe` decorator wraps the top-level function.
Child spans are created explicitly via `start_as_current_observation`.
Cost is computed automatically by Langfuse from model + token counts.

Trace lifecycle: async request → trace opens → child spans → judge (optional) → flush. `langfuse.flush()` guarantees delivery even if the process shuts down right after the response.

### LLM-as-judge sampling (`judge.py`)

10% of requests trigger a background faithfulness check:

- Judge model: **Haiku 4.5** (10x cheaper than Sonnet, sufficient for extractive checks — "does this claim appear in this context?")
- Runs async via `asyncio.create_task` — zero latency added to the user response
- Score `faithfulness_online` posted back to the parent trace with a one-sentence reason
- Silent on failure — a judge crash never breaks the API

Never use the same model as the generator for judging — it biases toward its own style.

### Infra layer (Prometheus)

`GET /metrics` exposes standard Prometheus counters:

- `copilot_requests_total{endpoint, status}` — count by route + status
- `copilot_errors_total{endpoint}` — 5xx + exceptions
- `copilot_request_latency_seconds{endpoint}` — histogram with buckets 0.1, 0.5, 1, 2, 3, 5, 10s

Middleware wraps every request. No sampling — infra metrics are cheap.

Fly.io auto-scrapes this endpoint on Day 9. No Prometheus server to run.

### Why two layers

Langfuse tells you _why_ something went wrong (which retrieval fetched what, which prompt tokens, which citations).
Prometheus tells you _what_ went wrong (which endpoint, when, how bad).
When p99 latency spikes at 3 AM, you look at Prometheus. When one specific answer is bad, you look at Langfuse.

## Key decisions

| Decision            | Chosen                     | Alternative                                | Why                                                                                    |
| ------------------- | -------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------------- |
| Vector DB           | pgvector                   | Pinecone, Weaviate                         | Same Postgres, no extra service                                                        |
| Embeddings          | text-embedding-3-small     | text-embedding-3-large                     | 5x cheaper, sufficient for legal-domain retrieval                                      |
| Chunk unit          | Legal article              | Fixed 500-char                             | Preserves citation structure                                                           |
| Migrations          | Raw SQL                    | Alembic                                    | No schema changes yet — overkill                                                       |
| Retrieval           | Hybrid vector + BM25 + RRF | Vector only, BM25 only, cross-encoder      | Covers both semantic and keyword queries; RRF avoids score-scaling issues              |
| Generation model    | Claude Sonnet 4.6          | GPT-4o, Gemini 2.5                         | Best JSON adherence + instruction following in tests                                   |
| API framework       | FastAPI                    | Flask, Litestar                            | Async native, Pydantic native, Swagger free                                            |
| Cache backend       | Redis                      | In-process dict, Postgres                  | Fast, TTL native, external for horizontal scaling later                                |
| Rate limiter        | slowapi                    | fastapi-limiter, custom                    | Simple, works with slowapi decorator, in-memory OK for v1                              |
| Eval framework      | Ragas + custom             | DeepEval, promptfoo                        | Ragas covers the 4 core RAG metrics out of box; custom fills the citation-contract gap |
| Experiment tracking | MLflow + SQLite            | W&B, Braintrust                            | Free, local, versioned; SQLite is the modern default (file store deprecated)           |
| LLM observability   | Langfuse Cloud             | self-hosted Langfuse, Braintrust, Helicone | Cloud avoids ClickHouse/Redis stack; enough for portfolio scale                        |
| Judge model         | Claude Haiku 4.5           | GPT-4o, Sonnet                             | Faithfulness is extractive; cheap works, and avoids bias vs generator                  |
| Infra metrics       | Prometheus                 | Datadog, custom                            | Standard, free, Fly.io scrapes automatically                                           |
