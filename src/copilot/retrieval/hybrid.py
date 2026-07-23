from __future__ import annotations

from typing import Sequence

from openai import OpenAI

from src.copilot.retrieval.bm25 import BM25Retriever
from src.copilot.retrieval.rrf import reciprocal_rank_fusion
from src.copilot.retrieval.types import RetrievedChunk
from src.copilot.retrieval.vector import VectorRetriever


class HybridRetriever:
    def __init__(
        self,
        db_url: str,
        openai_client: OpenAI,
        rrf_k: int = 60,
    ):
        self.vector = VectorRetriever(db_url, openai_client)
        self.bm25 = BM25Retriever(db_url)
        self.rrf_k = rrf_k

    def retrieve(
        self, query: str, k: int = 10, oversample: int = 2
    ) -> Sequence[RetrievedChunk]:
        # Fetch more from each retriever than we need, then fuse
        vec_results = self.vector.retrieve(query, k=k * oversample)
        bm25_results = self.bm25.retrieve(query, k=k * oversample)
        return reciprocal_rank_fusion(
            [vec_results, bm25_results],
            k=self.rrf_k,
            top_k=k,
        )