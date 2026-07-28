# Architecture

Technical decisions and rationale for the IBS/CBS Copilot.

## System overview

RAG copilot that answers Portuguese questions about Brazil's Tax Reform,
grounded in official legislation with mandatory article citations.

**Offline (ingestion):** PDFs → text extraction → legal-aware chunking →
embeddings → Postgres with pgvector.

**Online (query):** question → hybrid retrieval → Claude with grounded prompt
→ structured JSON with citations. All wrapped in Langfuse traces and
Prometheus metrics.

---

## Data pipeline (offline)

### Sources

- **LC 214/2025** — main law (institutes IBS, CBS, IS)
- **EC 132/2023** — constitutional foundation
- **Decree 12,955/2026** — CBS regulation

See [docs/download-sources.md](docs/download-sources.md).

### Extraction (`ingestion/pdf_to_text.py`)

- **PyMuPDF** for text extraction (fast, handles Portuguese diacritics well)
- Strips planalto.gov.br headers/footers (they'd otherwise pollute embeddings)
- Forces structural markers (`Art.`, `§`, `CAPÍTULO`) onto their own lines so
  the chunker can split cleanly

### Chunking (`ingestion/chunker.py`)

- **Article-aware:** one chunk per `Art. X` by default
- Long articles (>2000 chars) subchunk by `§`
- Each chunk carries metadata: `article`, `paragraph`, `item`, `source`

Why not fixed-size chunking: legal citations require exact reference. Cutting
an article in the middle of a sentence produces chunks that neither embed well
nor cite well. Structure is the signal.

### Storage (`ingestion/ingest.py`)

- **Postgres 16 + pgvector**, table `chunks(id, text, article, paragraph, item,
source, embedding vector(1536), metadata jsonb, text_tsv tsvector, created_at)`
- **HNSW index** (cosine) for vector search
- **GIN index** on `text_tsv` for BM25 (Portuguese-tuned via `to_tsvector('portuguese', text)`)
- **UNIQUE** constraint on `(source, article, paragraph, item)` — idempotent re-ingestion
- **Embeddings:** OpenAI `text-embedding-3-small` (5x cheaper than `-large`,
  sufficient for legal-domain retrieval)
- Batches of 100 chunks per API call
- Token-safe truncation via `tiktoken` (Portuguese takes more tokens than English)

---

## Query pipeline (online)

### Retrieval (`retrieval/`)

Three retrievers sharing a common `RetrievedChunk` type:

- **`VectorRetriever`** — cosine similarity via pgvector. Strong on paraphrasing
  and semantic questions ("como pagar imposto" → "recolhimento tributário").
  Weak on exact keywords the LLM hasn't seen in the exact form.

- **`BM25Retriever`** — Postgres `ts_rank` over a Portuguese `tsvector`. Strong
  on technical terms and article numbers ("Art. 31", "CBS"). Weak on paraphrasing.

- **`HybridRetriever`** — runs both, merges via **Reciprocal Rank Fusion (RRF)**.
  Fetches `2 × k` from each side, fuses with `k=60` (standard default), returns
  top `k`. RRF depends only on rank position, not on the retrievers' native
  scores — so vector's `[0,1]` cosine and BM25's unbounded `ts_rank` combine
  cleanly without normalization tuning.

**Why hybrid.** Legal Portuguese mixes technical terms ("split payment", "IBS",
"Art. 31") with natural-language questions. Neither retriever alone handles
both cases. In eyeball evaluation, vector missed "cashback tributário" (concept
exists but the exact term is rare); BM25 nailed it. Vector nailed "como
funciona o split payment?" by paraphrase; BM25 missed the subtler articles.
Hybrid keeps both wins.

**Not yet doing:**

- Cross-encoder reranking (Cohere, ColBERT)
- Query expansion
- HyDE (Hypothetical Document Embeddings)

Rule: don't add complexity before measuring. Day 6 evals tell us what to fix.

### Generation (`generation/`)

- **Model:** Claude Sonnet 4.6 via Anthropic SDK (async)
- **Prompts** (`prompts.py`, versioned): system prompt enforces 4 rules —
  strict grounding, mandatory citation, explicit refusal, JSON output. New
  prompt versions become new files (`prompts_v2.py`) so MLflow can compare.
- **Schema** (`schemas.py`): Pydantic `Answer` with `Citation` list. Quote
  capped at 500 chars — prevents the LLM from pasting whole articles.
  `confidence` restricted to `high | medium | low` via `Literal`.
- **Retry:** 2 attempts on `JSONDecodeError` or Pydantic `ValidationError`.
  Network/rate-limit errors bubble up (different handling needed).
- **Markdown-fence stripping:** the model occasionally wraps JSON in
  ` ```json `. Cleaned before parsing.

### Pipeline (`pipeline.py`)

`answer_question(question, k=5) -> PipelineResult`

- Retriever and Generator instantiated once (`@lru_cache`) — cheap reuse across
  requests
- Returns both the `Answer` and the retrieved `chunks` — Day 6 (Ragas) needs
  the chunks for faithfulness scoring
- Wrapped in `@observe` for Langfuse tracing

### API (`api/`)

- **Framework:** FastAPI with async routes throughout
- **Lifespan:** `AppState` opens Postgres pool, Redis connection, and builds
  the Generator on startup. Closed on shutdown. Single source of truth for
  shared resources.
- **Endpoints:**
  - `GET /v1/health` — liveness for load balancers
  - `GET /v1/sources` — explicit list of indexed documents. Transparency:
    users can verify what the copilot is grounded in.
  - `POST /v1/ask` — full pipeline. Cached, rate-limited.
  - `GET /metrics` — Prometheus scrape target
- **Cache** (`cache.py`): Redis, 1h TTL, key = `sha256(top_k:normalized_question)`.
  Same question with different `top_k` cached separately. Corrupt entries
  treated as misses. Cost impact: ~100x reduction on repeated questions.
- **Rate limit:** slowapi, 10 req/min per IP on `/ask` only. Health / sources
  unlimited. Currently keyed by socket IP; would switch to `X-Forwarded-For`
  behind a shared load balancer.
- **Validation:** Pydantic on request (`min_length=5`, `max_length=500` on
  question). FastAPI returns 422 on schema violations before the handler runs.

### Container (`Dockerfile`)

Multi-stage build:

1. **`deps` stage** — installs Poetry + all Python packages. Fat image, not shipped.
2. **`runtime` stage** — copies only the installed packages and `src/`. Slim (~250MB).

The `libpq5` system package is required at runtime for `psycopg` to connect
to Postgres.

### Local orchestration (`docker-compose.yml`)

Three services: `postgres` (pgvector/pg16), `redis`, `api`. Postgres and Redis
have healthchecks; `api` waits for `service_healthy` before starting. `.env`
values propagate through the `environment:` block.

---

## Evaluation (`evals/`)

### Golden set (`golden/golden_v1.jsonl`)

30 Portuguese questions covering rates, taxpayers, base, split payment,
cashback, exports, imports, basic goods basket, special regimes, selective tax.
Each row includes:

- `expected_answer_contains`: anchor keywords the answer should mention
- `expected_sources`: articles the citation should reference

Difficulty split: ~10 easy, ~13 medium, ~7 tricky.

Versioned — changes become `golden_v2.jsonl` etc. Lets us compare across
dataset versions in MLflow.

### Ragas metrics (built-in)

Four LLM-judged metrics per run:

- **Faithfulness** — is the answer grounded in retrieved chunks?
- **Answer Relevance** — does the answer address the question asked?
- **Context Precision** — are the retrieved chunks useful?
- **Context Recall** — did the retriever find all needed info?

### Citation accuracy (custom, `metrics/citation_accuracy.py`)

Deterministic. Compares `Answer.citations[*].article` against
`expected_sources`. Reports precision / recall / F1 per question, averaged.

Why custom: Ragas doesn't know the copilot's citation contract. Also runs
without any LLM calls — fast, free, safe to run on every commit.

### MLflow tracking

SQLite backend at `mlflow.db`. Each run logs:

- **Params:** `top_k`, `prompt_version`, `golden_size`, `dataset`
- **Metrics:** all 4 Ragas metrics + 3 citation metrics

Runs are named (`v0-baseline`, `v0-topk-3`, `v0-topk-10`) so the UI comparison
tells a story.

### v0 baseline diagnosis

Faithfulness 0.79 with Recall 0.43 means the answers are grounded but the
retriever misses relevant chunks. Retrieval is the bottleneck, not generation.
First levers to try: raise `top_k`, revisit chunker, add a reranker.

---

## Observability (`observability/`)

Two layers, different concerns.

### LLM layer (Langfuse)

Every `answer_question` call creates a hierarchical trace:

```
answer_question (trace)
├── retrieve (span) — question, k, retrieved articles
└── generate (generation) — model, tokens in/out, cost, full I/O
```

The `@observe` decorator wraps the top-level function. Child spans are created
explicitly via `start_as_current_observation`. Cost is computed automatically
by Langfuse from model + token counts.

**LLM-as-judge sampling (`judge.py`):** 10% of requests trigger a background
faithfulness check.

- Judge model: **Claude Haiku 4.5** — 10x cheaper than Sonnet, sufficient for
  extractive checks. Faithfulness is "does claim X appear in context Y?" — the
  judge doesn't need the reasoning power of the generator.
- Runs via `asyncio.create_task` — zero latency added to the user response
- Score `faithfulness_online` posted back to the parent trace with a
  one-sentence reason
- Silent on failure — a judge crash never breaks the API

Never use the same model as the generator for judging — it biases toward its
own style.

### Infra layer (Prometheus)

`GET /metrics` exposes standard Prometheus counters:

- `copilot_requests_total{endpoint, status}` — count by route + status
- `copilot_errors_total{endpoint}` — 5xx + exceptions
- `copilot_request_latency_seconds{endpoint}` — histogram, buckets 0.1, 0.5,
  1, 2, 3, 5, 10s

Middleware wraps every request. No sampling — infra metrics are cheap.

Fly.io auto-scrapes this endpoint (`[metrics]` in `fly.toml`). No Prometheus
server to run.

### Why two layers

Langfuse tells you _why_ something went wrong (which retrieval fetched what,
which prompt tokens, which citations).
Prometheus tells you _what_ went wrong (which endpoint, when, how bad).
When p99 latency spikes at 3 AM, you look at Prometheus. When one specific
answer is bad, you look at Langfuse.

---

## CI/CD (`.github/workflows/`)

### `ci.yml` — every push, every PR

- **Services:** Postgres (pgvector/pg16) + Redis, both with healthchecks
- **Steps:** Poetry install → apply migrations → ruff lint → pytest (unit +
  integration)
- Secrets injected via GitHub repo secrets (Anthropic, OpenAI, Langfuse) —
  never committed
- Poetry venv cached by `poetry.lock` hash

### `eval.yml` — PRs touching pipeline or evals

- **Path filter:** only runs on `src/copilot/**`, `evals/**`, `migrations/**`
  — most PRs skip it entirely
- **Concurrency:** force-push kills the previous run so old CI doesn't burn
  budget
- **Timeout:** 20 minutes — protection against runaway loops
- **Fixtures** (`evals/fixtures/chunks_seed.sql`): pre-baked SQL insert for
  chunks matching the golden set. Small file, no OpenAI cost in CI.
- Runs Ragas on `golden_v1.jsonl` with `top_k=5`
- **Gate** (`check_gate.py`): exits 1 if `faithfulness < 0.75` or
  `answer_relevance < 0.48`. Thresholds calibrated to v0 baseline − 0.05 — the
  gate protects against regression, not against imperfection. Raise as the
  system improves.
- **PR comment:** GitHub Actions posts a markdown table with all Ragas +
  custom citation scores

### Why this matters

The eval gate is the single most valuable engineering artifact in this project.
It makes quality regression physically impossible to merge. Every LLM system
eventually degrades silently — a prompt tweak that improves 9 questions can
quietly break the 10th. Without a gate, nobody notices for weeks. With a gate,
the PR fails immediately with the exact metric that dropped.

This is not common in AI codebases yet. It's what separates a "notebook
shipped to prod" from an actual system.

---

## Production deployment

- **API** on [Fly.io](https://fly.io), region `gru` (São Paulo). Auto-sleep
  when idle (`min_machines_running = 0`), cold start ~5s.
- **Postgres** on [Neon](https://neon.tech) — managed, pgvector included,
  region `sa-east-1`.
- **Redis** on [Upstash](https://upstash.com) — managed, TLS, São Paulo.

Compute and data separated — standard production pattern. Scale each
independently. Neon and Upstash both do backups, point-in-time recovery, and
managed upgrades that unmanaged self-hosted Postgres/Redis wouldn't.

Config in `fly.toml`. Secrets via `flyctl secrets set`.

---

## Key decisions

| Decision            | Chosen                     | Alternative                                | Why                                                                       |
| ------------------- | -------------------------- | ------------------------------------------ | ------------------------------------------------------------------------- |
| Vector DB           | pgvector                   | Pinecone, Weaviate                         | Same Postgres, no extra service                                           |
| Embeddings          | `text-embedding-3-small`   | `text-embedding-3-large`                   | 5x cheaper, sufficient for legal-domain retrieval                         |
| Chunk unit          | Legal article              | Fixed 500-char                             | Preserves citation structure                                              |
| Migrations          | Raw SQL                    | Alembic                                    | No schema evolution complexity yet                                        |
| Retrieval           | Hybrid vector + BM25 + RRF | Vector only, cross-encoder                 | Covers both semantic and keyword queries; RRF avoids score-scaling issues |
| Generation model    | Claude Sonnet 4.6          | GPT-4o, Gemini 2.5                         | Best JSON adherence + instruction following in tests                      |
| API framework       | FastAPI                    | Flask, Litestar                            | Async native, Pydantic native, Swagger free                               |
| Cache backend       | Redis                      | In-process dict, Postgres                  | Fast, TTL native, external for horizontal scaling later                   |
| Rate limiter        | slowapi                    | fastapi-limiter, custom                    | Simple, in-memory OK for v1                                               |
| Eval framework      | Ragas + custom             | DeepEval, promptfoo                        | Ragas covers 4 core RAG metrics; custom fills the citation contract gap   |
| Experiment tracking | MLflow + SQLite            | W&B, Braintrust                            | Free, local, SQLite is the modern default                                 |
| LLM observability   | Langfuse Cloud             | self-hosted Langfuse, Braintrust, Helicone | Cloud avoids ClickHouse/Redis stack; enough for portfolio scale           |
| Judge model         | Claude Haiku 4.5           | GPT-4o, Sonnet                             | Faithfulness is extractive; cheap works, avoids bias vs generator         |
| Infra metrics       | Prometheus                 | Datadog                                    | Standard, free, Fly.io scrapes automatically                              |
| CI                  | GitHub Actions             | GitLab CI, CircleCI                        | Free for public repos, native to GitHub                                   |
| Lint                | ruff                       | flake8 + black + isort                     | 10x faster, all three combined                                            |
| Eval-in-CI          | Ragas + custom gate on PRs | manual review                              | Blocks silent quality regression at the merge point                       |
| Compute host        | Fly.io                     | Railway, Render                            | Auto-sleep for portfolio idle cost, São Paulo region                      |
| DB host             | Neon                       | Supabase, RDS                              | Free tier with pgvector, managed backups                                  |
| Cache host          | Upstash                    | ElastiCache                                | Free tier, São Paulo, low ops                                             |

---

## Roadmap

- Bump `top_k` and re-evaluate — target Recall > 0.60
- Add cross-encoder reranker (Cohere Rerank or ColBERT)
- Streaming responses via SSE
- Streamlit UI at `/app` for non-technical users
- Ingest NT 2025.002 (NF-e technical spec) for ERP developer queries
- Multi-tenant + per-user rate limiting
