# evals/metrics/citation_accuracy.py
"""Custom metric: does the answer cite the expected articles?"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class CitationScore:
    precision: float  # of what was cited, how much was expected
    recall: float     # of what was expected, how much was cited
    f1: float
    predicted: set[str]
    expected: set[str]


_ARTICLE_RE = re.compile(r"art\.?\s*(\d+)", re.IGNORECASE)


def _normalize(article: str) -> str:
    """'Art. 12º' → 'art_12'. Handles minor formatting variance."""
    m = _ARTICLE_RE.search(article)
    return f"art_{m.group(1)}" if m else article.lower().strip()


def score_citations(
    predicted_citations: list[dict],
    expected_sources: list[str],
) -> CitationScore:
    """
    predicted_citations: e.g. [{"article": "Art. 15", "source": "LC 214/2025"}]
    expected_sources:    e.g. ["Art. 15", "LC 214/2025"]  (mixed articles + source labels)
    """
    predicted = {_normalize(c["article"]) for c in predicted_citations}
    expected = {_normalize(s) for s in expected_sources if _ARTICLE_RE.search(s)}

    if not expected:
        return CitationScore(1.0, 1.0, 1.0, predicted, expected)

    tp = len(predicted & expected)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(expected)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return CitationScore(
        precision=precision,
        recall=recall,
        f1=f1,
        predicted=predicted,
        expected=expected,
    )


def aggregate(scores: list[CitationScore]) -> dict[str, float]:
    """Mean across the eval set."""
    n = len(scores) or 1
    return {
        "citation_precision": sum(s.precision for s in scores) / n,
        "citation_recall": sum(s.recall for s in scores) / n,
        "citation_f1": sum(s.f1 for s in scores) / n,
    }