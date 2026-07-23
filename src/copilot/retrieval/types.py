from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    id: int
    text: str
    article: str
    paragraph: str | None
    source: str
    score: float
    metadata: dict