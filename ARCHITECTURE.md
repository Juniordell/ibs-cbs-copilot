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

## Key decisions

| Decision         | Chosen                     | Alternative                           | Why                                                                       |
| ---------------- | -------------------------- | ------------------------------------- | ------------------------------------------------------------------------- |
| Vector DB        | pgvector                   | Pinecone, Weaviate                    | Same Postgres, no extra service                                           |
| Embeddings       | text-embedding-3-small     | text-embedding-3-large                | 5x cheaper, sufficient for legal-domain retrieval                         |
| Chunk unit       | Legal article              | Fixed 500-char                        | Preserves citation structure                                              |
| Migrations       | Raw SQL                    | Alembic                               | No schema changes yet — overkill                                          |
| Retrieval        | Hybrid vector + BM25 + RRF | Vector only, BM25 only, cross-encoder | Covers both semantic and keyword queries; RRF avoids score-scaling issues |
| Generation model | Claude Sonnet 4.6          | GPT-4o, Gemini 2.5                    | Best JSON adherence + instruction following in tests                      |
