# src/copilot/api/cache.py
"""Redis-backed answer cache. Key: sha256 of normalized question."""

from __future__ import annotations

import hashlib
import json
import logging

from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class AnswerCache:
    def __init__(self, redis: Redis, ttl: int = 3600):
        self.redis = redis
        self.ttl = ttl

    def _key(self, question: str, top_k: int) -> str:
        norm = f"{top_k}:{question.lower().strip()}"
        h = hashlib.sha256(norm.encode()).hexdigest()[:16]
        return f"answer:{h}"

    async def get(self, question: str, top_k: int) -> dict | None:
        raw = await self.redis.get(self._key(question, top_k))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Corrupt cache entry, ignoring")
            return None

    async def set(self, question: str, top_k: int, payload: dict) -> None:
        await self.redis.set(
            self._key(question, top_k),
            json.dumps(payload),
            ex=self.ttl,
        )