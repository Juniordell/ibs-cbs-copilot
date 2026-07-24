# tests/test_api.py
"""Integration tests for the FastAPI service. LLM and DB are mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.copilot.generation.generator import GenerationResult
from src.copilot.generation.schemas import Answer, Citation
from src.copilot.pipeline import PipelineResult
from src.copilot.retrieval.types import RetrievedChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_pipeline_result() -> PipelineResult:
    return PipelineResult(
        generation=GenerationResult(
            answer=Answer(
                answer="A alíquota do IBS é definida pela soma...",
                citations=[Citation(article="Art. 15", source="LC 214/2025",
                                    quote="A alíquota do IBS...")],
                confidence="high",
                gaps=None,
            ),
            input_tokens=1500,
            output_tokens=200,
            model="claude-sonnet-4-6",
            prompt_version="v1",
        ),
        chunks=[
            RetrievedChunk(id=1, text="...", article="Art. 15", paragraph=None,
                           source="LC 214/2025", score=0.9, metadata={}),
        ],
    )


@pytest.fixture
def mock_app_state():
    """Patch AppState.create so the app doesn't need real DB/Redis/OpenAI."""
    fake_cache = AsyncMock()
    fake_cache.get.return_value = None  # always cache miss by default

    fake_state = AsyncMock()
    fake_state.cache = fake_cache

    with patch(
        "src.copilot.api.state.AppState.create",
        AsyncMock(return_value=fake_state),
    ):
        yield fake_state


@pytest.fixture
async def client(mock_app_state):
    """HTTP client bound to the app in-memory (no real server)."""
    from src.copilot.api.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Trigger lifespan startup
        async with app.router.lifespan_context(app):
            yield ac


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_sources(client):
    r = await client.get("/v1/sources")
    assert r.status_code == 200
    body = r.json()
    assert len(body["sources"]) == 3
    ids = {s["id"] for s in body["sources"]}
    assert "lc-214-2025" in ids


@pytest.mark.asyncio
async def test_ask_ok(client, fake_pipeline_result):
    with patch(
        "src.copilot.api.routes.answer_question",
        AsyncMock(return_value=fake_pipeline_result),
    ):
        r = await client.post("/v1/ask", json={"question": "Qual a alíquota do IBS?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]["confidence"] == "high"
    assert body["answer"]["citations"][0]["article"] == "Art. 15"
    assert body["input_tokens"] == 1500


@pytest.mark.asyncio
async def test_ask_too_short_question(client):
    r = await client.post("/v1/ask", json={"question": "IBS"})
    assert r.status_code == 422  # FastAPI returns 422 for validation errors


@pytest.mark.asyncio
async def test_ask_rate_limit(client, fake_pipeline_result):
    from src.copilot.api.limiter import limiter
    limiter.reset()

    with patch(
        "src.copilot.api.routes.answer_question",
        AsyncMock(return_value=fake_pipeline_result),
    ):
        codes = []
        for _ in range(12):
            r = await client.post("/v1/ask", json={"question": "Qual a alíquota do IBS?"})
            codes.append(r.status_code)
    assert codes.count(200) == 10
    assert 429 in codes