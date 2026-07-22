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

_Day 3+. Not yet implemented._

## Key decisions

| Decision   | Chosen                 | Alternative            | Why                                               |
| ---------- | ---------------------- | ---------------------- | ------------------------------------------------- |
| Vector DB  | pgvector               | Pinecone, Weaviate     | Same Postgres, no extra service                   |
| Embeddings | text-embedding-3-small | text-embedding-3-large | 5x cheaper, sufficient for legal-domain retrieval |
| Chunk unit | Legal article          | Fixed 500-char         | Preserves citation structure                      |
| Migrations | Raw SQL                | Alembic                | No schema changes yet — overkill                  |
