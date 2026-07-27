from __future__ import annotations

import logging
from collections.abc import Sequence

import psycopg
from openai import OpenAI
from pgvector.psycopg import register_vector

from src.copilot.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"


class VectorRetriever:
    def __init__(self, db_url: str, openai_client: OpenAI):
        self.db_url = db_url
        self.client = openai_client

    def _embed(self, query: str) -> list[float]:
        resp = self.client.embeddings.create(
            model=EMBEDDING_MODEL, input=[query]
        )
        return resp.data[0].embedding

    def retrieve(self, query: str, k: int = 10) -> Sequence[RetrievedChunk]:
        query_emb = self._embed(query)

        with psycopg.connect(self.db_url) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, text, article, paragraph, source, metadata,
                           1 - (embedding <=> %s::vector) AS score
                    FROM chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_emb, query_emb, k),
                )
                rows = cur.fetchall()

        return [
            RetrievedChunk(
                id=r[0], text=r[1], article=r[2], paragraph=r[3],
                source=r[4], metadata=r[5] or {}, score=float(r[6]),
            )
            for r in rows
        ]