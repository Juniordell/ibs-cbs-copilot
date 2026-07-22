-- migrations/001_init.sql

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    text         TEXT NOT NULL,
    article      TEXT NOT NULL,
    paragraph    TEXT,
    item         TEXT,
    source       TEXT NOT NULL,
    embedding    vector(1536),
    metadata     JSONB NOT NULL DEFAULT '{}',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, article, paragraph, item)
);

-- HNSW index for vector similarity (cosine)
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- BTree for filtering by source
CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks (source);