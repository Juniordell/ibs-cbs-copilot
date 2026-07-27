# src/copilot/ingestion/chunker.py
"""Split legal text into article-level chunks with structural metadata."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class LegalChunk:
    text: str
    article: str            # "Art. 12"
    paragraph: str | None   # "§ 1º" or "Parágrafo único"
    item: str | None        # "II" (inciso) or "a" (alínea)
    source: str             # "LC 214/2025"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# Matches "Art. 12", "Art. 12º", "Art. 12." — captures the number
ARTICLE_RE = re.compile(r"^Art\.\s*(\d+[º°]?)\.?\s*", re.MULTILINE)

# Matches "§ 1º" or "Parágrafo único"
PARAGRAPH_RE = re.compile(
    r"^(§\s*\d+[º°]?|Parágrafo único\.?)",
    re.MULTILINE | re.IGNORECASE,
)

# Roman numeral incisos: "I -", "II -", "XIV -"
INCISO_RE = re.compile(r"^([IVXLCDM]+)\s*[-–]\s*", re.MULTILINE)

MAX_CHUNK_CHARS = 2000


def _subchunk_by_paragraph(
    article: str, body: str, source: str
) -> Iterator[LegalChunk]:
    """Split a long article by § markers."""
    parts = PARAGRAPH_RE.split(body)
    # parts[0] is the article caput (before the first §)
    caput = parts[0].strip()
    if caput:
        yield LegalChunk(
            text=f"{article}. {caput}",
            article=article,
            paragraph=None,
            item=None,
            source=source,
            metadata={"kind": "caput", "length": len(caput)},
        )

    # Then pairs of (marker, content)
    for i in range(1, len(parts), 2):
        marker = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        yield LegalChunk(
            text=f"{article}, {marker} {content}",
            article=article,
            paragraph=marker,
            item=None,
            source=source,
            metadata={"kind": "paragraph", "length": len(content)},
        )


def chunk_legal_text(text: str, source: str) -> Iterator[LegalChunk]:
    """Yield chunks, one per article (or per § when the article is long)."""
    matches = list(ARTICLE_RE.finditer(text))
    if not matches:
        return

    for i, match in enumerate(matches):
        art_num = match.group(1)
        article = f"Art. {art_num}"

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        if not body:
            continue

        if len(body) > MAX_CHUNK_CHARS:
            yield from _subchunk_by_paragraph(article, body, source)
        else:
            yield LegalChunk(
                text=f"{article}. {body}",
                article=article,
                paragraph=None,
                item=None,
                source=source,
                metadata={"kind": "article", "length": len(body)},
            )


def chunk_file(txt_path: Path, source: str) -> list[LegalChunk]:
    text = txt_path.read_text(encoding="utf-8")
    return list(chunk_legal_text(text, source))