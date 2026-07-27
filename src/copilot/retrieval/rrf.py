from __future__ import annotations

from collections.abc import Sequence

from src.copilot.retrieval.types import RetrievedChunk


def reciprocal_rank_fusion(
    result_sets: Sequence[Sequence[RetrievedChunk]],
    k: int = 60,
    top_k: int = 10,
) -> list[RetrievedChunk]:
    """Merge multiple ranked lists. Score = Σ 1/(k + rank_i)."""
    scores: dict[int, float] = {}
    chunk_by_id: dict[int, RetrievedChunk] = {}

    for results in result_sets:
        for rank, chunk in enumerate(results):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
            chunk_by_id[chunk.id] = chunk

    ranked_ids = sorted(scores, key=lambda cid: -scores[cid])
    out = []
    for cid in ranked_ids[:top_k]:
        c = chunk_by_id[cid]
        # Replace the retriever-specific score with the fused RRF score
        c.score = scores[cid]
        out.append(c)
    return out