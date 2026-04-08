"""
Cross-encoder reranker — CPU-friendly alternative to ColBERT.

Uses sentence-transformers CrossEncoder with cross-encoder/ms-marco-MiniLM-L-6-v2
(or similar) to score (query, passage) pairs. Significantly slower than dense
retrieval but acceptable on CPU for small candidate lists (topk ≤ 50).

Latency budget on CPU:
  - 50 candidates: ~300-500 ms
  - 100 candidates: ~600-1000 ms
  - 200 candidates: ~1200-2000 ms (exceeds 500 ms budget)
"""
from __future__ import annotations

import logging
from typing import Optional

from ..retrieval.dense_retriever import RetrievalResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks candidates by scoring each (query, passage) pair with a cross-encoder.

    Args:
        model_name: HuggingFace cross-encoder checkpoint.
                    Default: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, competent).
        topk:       Maximum number of candidates to rerank.
        batch_size: Batch size for the cross-encoder forward pass.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        topk: int = 50,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.topk = topk
        self.batch_size = batch_size
        self._model = None

    def load(self) -> None:
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.model_name)
        logger.info(f"CrossEncoderReranker loaded: {self.model_name}")

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        topk: Optional[int] = None,
    ) -> list[RetrievalResult]:
        if self._model is None:
            raise RuntimeError("Call load() before rerank().")
        if not candidates:
            return candidates

        cap = topk if topk is not None else self.topk
        to_rerank = candidates[:cap]
        tail = candidates[cap:]

        pairs = [
            (query, f"{r.title} {r.text}".strip() if r.title else r.text)
            for r in to_rerank
        ]

        try:
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}. Returning fused order.")
            return candidates

        scored = list(zip(to_rerank, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)

        reranked: list[RetrievalResult] = []
        for orig, sc in scored:
            reranked.append(
                RetrievalResult(
                    chunk_id=orig.chunk_id,
                    doc_id=orig.doc_id,
                    domain=orig.domain,
                    score=orig.score,
                    text=orig.text,
                    title=orig.title,
                    bm25_score=orig.bm25_score,
                    fused_score=orig.fused_score,
                    colbert_score=float(sc),  # reuse this slot to carry the rerank score
                )
            )

        reranked.extend(tail)
        return reranked
