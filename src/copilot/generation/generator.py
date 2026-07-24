from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from anthropic import AsyncAnthropic
from pydantic import ValidationError

from src.copilot.generation.prompts import (
    SYSTEM_PROMPT,
    VERSION as PROMPT_VERSION,
    build_user_message,
)
from src.copilot.generation.schemas import Answer
from src.copilot.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024
MAX_RETRIES = 2


@dataclass
class GenerationResult:
    answer: Answer
    input_tokens: int
    output_tokens: int
    model: str
    prompt_version: str


class Generator:
    def __init__(self, api_key: str, model: str = MODEL):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model

    async def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> GenerationResult:
        user_message = build_user_message(question, chunks)

        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                message = await self.client.messages.create(
                    model=self.model,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                )
                raw = message.content[0].text
                data = self._parse_json(raw)
                answer = Answer(**data)

                return GenerationResult(
                    answer=answer,
                    input_tokens=message.usage.input_tokens,
                    output_tokens=message.usage.output_tokens,
                    model=self.model,
                    prompt_version=PROMPT_VERSION,
                )

            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                logger.warning(
                    "Generation attempt %d/%d failed: %s",
                    attempt, MAX_RETRIES, e,
                )
                continue

        raise RuntimeError(
            f"Generator failed after {MAX_RETRIES} attempts: {last_error}"
        )

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Strip markdown fences if present, then parse."""
        cleaned = raw.strip()
        # Handles ```json ... ``` and ``` ... ```
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return json.loads(cleaned)