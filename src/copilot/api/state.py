from __future__ import annotations

import logging
from dataclasses import dataclass

from openai import OpenAI
from psycopg_pool import AsyncConnectionPool
from redis.asyncio import Redis

from src.copilot.generation.generator import Generator
from src.copilot.api.cache import AnswerCache

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    db_pool: AsyncConnectionPool
    redis: Redis
    openai: OpenAI
    generator: Generator
    cache: AnswerCache

    @classmethod
    async def create(
        cls,
        db_url: str,
        redis_url: str,
        openai_key: str,
        anthropic_key: str,
    ) -> "AppState":
        logger.info("Opening Postgres pool")
        db_pool = AsyncConnectionPool(db_url, min_size=1, max_size=10, open=False)
        await db_pool.open()

        logger.info("Opening Redis connection")
        redis = Redis.from_url(redis_url, decode_responses=True)
        await redis.ping()

        openai = OpenAI(api_key=openai_key)
        generator = Generator(api_key=anthropic_key)
        cache = AnswerCache(redis)

        return cls(
            db_pool=db_pool,
            redis=redis,
            openai=openai,
            generator=generator,
            cache=cache,
        )

    async def close(self):
        await self.db_pool.close()
        await self.redis.aclose()