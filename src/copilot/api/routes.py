# src/copilot/api/routes.py
"""HTTP routes for the copilot API."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.copilot.api.cache import AnswerCache
from src.copilot.api.limiter import limiter
from src.copilot.generation.schemas import Answer
from src.copilot.pipeline import answer_question

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=5,
        max_length=500,
        description="Question in Portuguese about Brazil's Tax Reform (IBS/CBS).",
        examples=["Qual a alíquota do IBS?"],
    )
    top_k: int = Field(
        default=5, ge=1, le=20,
        description="Number of chunks to retrieve. Higher = more context, higher cost.",
    )


class AskResponse(BaseModel):
    answer: Answer
    input_tokens: int
    output_tokens: int
    model: str
    prompt_version: str
    retrieved_articles: list[str] = Field(
        description="Article IDs of the chunks used as context.",
    )


class Source(BaseModel):
    id: str
    title: str
    url: str
    kind: str


class SourcesResponse(BaseModel):
    sources: list[Source]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    summary="Health check",
    description="Returns 200 if the service is up. Used by Fly.io / load balancers.",
    tags=["ops"],
)
async def health():
    return {"status": "ok", "version": "0.1.0"}


@router.get(
    "/sources",
    response_model=SourcesResponse,
    summary="List indexed sources",
    description=(
        "Documents currently indexed by the copilot. Kept explicit so users "
        "can verify what the system is grounded in."
    ),
    tags=["ops"],
)
async def sources():
    return SourcesResponse(sources=[
        Source(
            id="lc-214-2025",
            title="Lei Complementar 214/2025",
            url="https://www.planalto.gov.br/ccivil_03/leis/lcp/lcp214.htm",
            kind="law",
        ),
        Source(
            id="ec-132-2023",
            title="Emenda Constitucional 132/2023",
            url="https://www.planalto.gov.br/ccivil_03/constituicao/emendas/emc/emc132.htm",
            kind="constitutional-amendment",
        ),
        Source(
            id="decreto-12955-2026",
            title="Decreto 12.955/2026",
            url="https://www.planalto.gov.br/ccivil_03/_ato2023-2026/2026/decreto/D12955.htm",
            kind="decree",
        ),
    ])


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question about the Tax Reform",
    description=(
        "Runs the full pipeline: hybrid retrieval (vector + BM25 + RRF) → "
        "Claude Sonnet with grounded prompt → structured answer with citations. "
        "Cached in Redis for 1 hour per (question, top_k)."
    ),
    tags=["copilot"],
)
@limiter.limit("10/minute")
async def ask(req: QuestionRequest, request: Request):
    cache: AnswerCache = request.app.state.app_state.cache

    cached = await cache.get(req.question, req.top_k)
    if cached is not None:
        logger.info("Cache hit for: %s", req.question[:60])
        return AskResponse(**cached)

    logger.info("Cache miss for: %s", req.question[:60])
    try:
        result = await answer_question(req.question, k=req.top_k)
    except Exception:
        logger.exception("answer_question failed")
        raise HTTPException(500, "Internal error while generating the answer.")

    response = AskResponse(
        answer=result.generation.answer,
        input_tokens=result.generation.input_tokens,
        output_tokens=result.generation.output_tokens,
        model=result.generation.model,
        prompt_version=result.generation.prompt_version,
        retrieved_articles=[f"{c.article} · {c.source}" for c in result.chunks],
    )

    await cache.set(req.question, req.top_k, response.model_dump())
    return response