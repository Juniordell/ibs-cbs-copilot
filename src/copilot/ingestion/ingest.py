
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import psycopg
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from src.copilot.ingestion.chunker import LegalChunk, chunk_file

load_dotenv()

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 100

MAX_CHARS = 20000  # ~7500 tokens, safe margin

_ENC = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 8000  # safe under 8192

def _truncate_tokens(text: str, max_tokens: int) -> str:
    tokens = _ENC.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _ENC.decode(tokens[:max_tokens])

def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    safe = [_truncate_tokens(t, MAX_TOKENS) for t in texts]
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=safe)
    return [item.embedding for item in resp.data]

def insert_chunks(
    conn: psycopg.Connection,
    chunks: list[LegalChunk],
    embeddings: list[list[float]],
) -> int:
    """Insert a batch. Skip duplicates by (source, article, paragraph, item)."""
    with conn.cursor() as cur:
        rows = [
            (
                c.text, c.article, c.paragraph, c.item,
                c.source, emb, Jsonb(c.metadata),
            )
            for c, emb in zip(chunks, embeddings)
        ]
        cur.executemany(
            """
            INSERT INTO chunks
                (text, article, paragraph, item, source, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source, article, paragraph, item) DO NOTHING
            """,
            rows,
        )
        return cur.rowcount


def ingest(txt_path: Path, source: str, db_url: str, api_key: str) -> int:
    logger.info("Chunking %s", txt_path)
    chunks = chunk_file(txt_path, source)
    logger.info("Got %d chunks", len(chunks))

    if not chunks:
        logger.warning("No chunks produced. Nothing to ingest.")
        return 0

    client = OpenAI(api_key=api_key)
    total_inserted = 0

    with psycopg.connect(db_url) as conn:
        register_vector(conn)

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            texts = [c.text for c in batch]

            logger.info(
                "Embedding batch %d/%d (%d chunks)",
                i // BATCH_SIZE + 1,
                (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE,
                len(batch),
            )
            embeddings = embed_batch(client, texts)
            inserted = insert_chunks(conn, batch, embeddings)
            total_inserted += inserted
            conn.commit()

    logger.info("Inserted %d new chunks (%d skipped as duplicates)",
                total_inserted, len(chunks) - total_inserted)
    return total_inserted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to the processed .txt file")
    parser.add_argument("--source", type=str, required=True,
                        help='Source label, e.g. "LC 214/2025"')
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    db_url = os.getenv("DATABASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")

    if not db_url:
        logger.error("DATABASE_URL not set")
        return 1
    if not api_key:
        logger.error("OPENAI_API_KEY not set")
        return 1

    try:
        ingest(args.input, args.source, db_url, api_key)
    except Exception:
        logger.exception("Ingestion failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())