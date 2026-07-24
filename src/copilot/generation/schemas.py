# src/copilot/generation/schemas.py
"""Pydantic schemas for the LLM response."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, field_validator


class Citation(BaseModel):
    article: str = Field(min_length=1, max_length=50)
    source: str = Field(min_length=1, max_length=100)
    quote: str = Field(min_length=1, max_length=500)


class Answer(BaseModel):
    answer: str = Field(min_length=1, max_length=4000)
    citations: list[Citation] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    gaps: str | None = None

    @field_validator("gaps", mode="before")
    @classmethod
    def _empty_to_none(cls, v):
        if v in ("", "null", "none", "None"):
            return None
        return v