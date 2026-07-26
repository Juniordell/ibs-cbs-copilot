# src/copilot/pipeline.py
from __future__ import annotations

import logging
import os
import asyncio
from anthropic import AsyncAnthropic
from src.copilot.observability.judge import judge_faithfulness, should_sample
from dataclasses import dataclass
from functools import lru_cache

from langfuse import Langfuse, observe
from openai import OpenAI

from src.copilot.generation.generator import Generator, GenerationResult
from src.copilot.retrieval.hybrid import HybridRetriever
from src.copilot.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)

langfuse = Langfuse()  # reads env vars


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

@lru_cache(maxsize=1)
def _get_judge_client() -> AsyncAnthropic:
    return AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

@observe(name="answer_question")
async def answer_question(question: str, k: int = 5) -> PipelineResult:
    retriever = _get_retriever()
    generator = _get_generator()

    # ---- Retrieve span ----
    with langfuse.start_as_current_observation(
        as_type="span", name="retrieve",
        input={"question": question, "k": k},
    ) as span:
        chunks = list(retriever.retrieve(question, k=k))
        span.update(output={
            "num_chunks": len(chunks),
            "articles": [f"{c.article} · {c.source}" for c in chunks],
        })

    # ---- Generate span (typed as generation for tokens/cost) ----
    with langfuse.start_as_current_observation(
        as_type="generation",
        name="generate",
        model=generator.model,
        input=[{"role": "user", "content": question}],
    ) as gen:
        generation = await generator.generate(question, chunks)
        gen.update(
            output=generation.answer.model_dump(),
            usage_details={
                "input": generation.input_tokens,
                "output": generation.output_tokens,
            },
        )

    # Sample 10% for online eval
    if should_sample():
        trace_id = langfuse.get_current_trace_id()
        if trace_id:
            asyncio.create_task(judge_faithfulness(
                trace_id=trace_id,
                question=question,
                answer=generation.answer.answer,
                contexts=[c.text for c in chunks],
                langfuse=langfuse,
                anthropic=_get_judge_client(),
            ))

    langfuse.flush()
    return PipelineResult(generation=generation, chunks=chunks)