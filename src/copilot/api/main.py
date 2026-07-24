from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.copilot.api.limiter import limiter
from src.copilot.api.routes import router as v1_router
from src.copilot.api.state import AppState

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: build resources. Shutdown: release them."""
    logger.info("Starting up")
    state = await AppState.create(
        db_url=os.environ["DATABASE_URL"],
        redis_url=os.environ["REDIS_URL"],
        openai_key=os.environ["OPENAI_API_KEY"],
        anthropic_key=os.environ["ANTHROPIC_API_KEY"],
    )
    app.state.app_state = state
    logger.info("Ready")
    try:
        yield
    finally:
        logger.info("Shutting down")
        await state.close()
        logger.info("Bye")


app = FastAPI(
    title="IBS/CBS Copilot",
    description="RAG over Brazil's Tax Reform (LC 214/2025, EC 132/2023, Decree 12,955/2026).",
    version="0.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(v1_router)


@app.get("/")
async def root():
    return {"name": "IBS/CBS Copilot", "docs": "/docs", "version": "0.1.0"}