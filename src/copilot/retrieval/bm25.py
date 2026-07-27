from __future__ import annotations

import logging
from collections.abc import Sequence

import psycopg

from src.copilot.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)


class BM25Retriever:
    def __init__(self, db_url: str, lang: str = "portuguese"):
        self.db_url = db_url
        self.lang = lang

    def retrieve(self, query: str, k: int = 10) -> Sequence[RetrievedChunk]:
        with psycopg.connect(self.db_url) as conn, conn.cursor() as cur:
            cur.execute(
                """
                    SELECT id, text, article, paragraph, source, metadata,
                           ts_rank(text_tsv, plainto_tsquery(%s, %s)) AS score
                    FROM chunks
                    WHERE text_tsv @@ plainto_tsquery(%s, %s)
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                (self.lang, query, self.lang, query, k),
            )
            rows = cur.fetchall()

        return [
            RetrievedChunk(
                id=r[0], text=r[1], article=r[2], paragraph=r[3],
                source=r[4], metadata=r[5] or {}, score=float(r[6]),
            )
            for r in rows
        ]