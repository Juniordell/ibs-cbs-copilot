from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

from openai import OpenAI

from src.copilot.generation.generator import Generator, GenerationResult
from src.copilot.retrieval.hybrid import HybridRetriever
from src.copilot.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    generation: GenerationResult
    chunks: list[RetrievedChunk]


@lru_cache(maxsize=1)
def _get_retriever() -> HybridRetriever:
    return HybridRetriever(
        db_url=os.environ["DATABASE_URL"],
        openai_client=OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
    )


@lru_cache(maxsize=1)
def _get_generator() -> Generator:
    return Generator(api_key=os.environ["ANTHROPIC_API_KEY"])


async def answer_question(question: str, k: int = 5) -> PipelineResult:
    """Retrieve top-k chunks, generate a grounded answer."""
    retriever = _get_retriever()
    generator = _get_generator()

    logger.info("Retrieving for: %s", question[:80])
    chunks = list(retriever.retrieve(question, k=k))

    if not chunks:
        logger.warning("No chunks retrieved for: %s", question[:80])

    logger.info("Generating answer from %d chunks", len(chunks))
    generation = await generator.generate(question, chunks)

    return PipelineResult(generation=generation, chunks=chunks)