"""
ColBERT v2 reranker — Mode B (rerank-only, no indexing).

Uses RAGatouille's RAGPretrainedModel.rerank() to score query-passage pairs
with ColBERT's MaxSim late-interaction. No ColBERT index is built.

GPU memory budget (RTX 3060 Ti 12GB):
  - ColBERT v2 weights ~400MB
  - Token interaction matrix for 200 passages × 180 tokens: ~2GB
  - Safe cap: topk ≤ 200 for full config, ≤ 50 for smoke test.

RAGatouille rerank() return format:
  list of {"content": str, "score": float, "rank": int, "document_id": int}
  where document_id is the 0-based index into the input documents list.
"""
from __future__ import annotations

import logging
from typing import Optional

from ..retrieval.dense_retriever import RetrievalResult

logger = logging.getLogger(__name__)


class ColBERTReranker:
    """
    Reranks a candidate list using ColBERT v2 MaxSim scoring.

    Args:
        model_name: HuggingFace checkpoint (default "colbert-ir/colbertv2.0").
        topk:       Maximum number of candidates to rerank (GPU memory constraint).
    """

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        topk: int = 200,
    ):
        self.model_name = model_name
        self.topk = topk
        self._model = None

    def load(self) -> None:
        try:
            from ragatouille import RAGPretrainedModel
        except ImportError as e:
            raise ImportError(
                "RAGatouille is required for ColBERT reranking. "
                "Install with: pip install ragatouille"
            ) from e

        self._model = RAGPretrainedModel.from_pretrained(self.model_name)
        logger.info(f"ColBERTReranker loaded: {self.model_name}")

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        topk: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """
        Rerank candidates using ColBERT v2.

        Args:
            query:      Normalised query string.
            candidates: Fused candidate list (up to K_fusion items).
            topk:       Override instance-level topk cap.

        Returns:
            Candidates sorted by ColBERT score descending.
            Unranked candidates (beyond topk cap) are appended at the end
            with their original fused scores preserved.
        """
        if self._model is None:
            raise RuntimeError("Call load() before rerank().")

        cap = topk if topk is not None else self.topk
        to_rerank = candidates[:cap]
        tail = candidates[cap:]  # candidates beyond cap, kept as-is

        texts = [
            f"{r.title} {r.text}".strip() if r.title else r.text
            for r in to_rerank
        ]

        try:
            raw = self._model.rerank(
                query=query,
                documents=texts,
                k=len(texts),
            )
        except Exception as e:
            logger.error(f"ColBERT reranking failed: {e}. Returning fused order.")
            return candidates

        # Map back: raw result's document_id is the 0-based index into `texts`
        reranked: list[RetrievalResult] = []
        for item in raw:
            orig_idx = item.get("document_id", item.get("result_index", 0))
            if orig_idx < 0 or orig_idx >= len(to_rerank):
                continue
            orig = to_rerank[orig_idx]
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
                    colbert_score=float(item["score"]),
                )
            )

        # Append tail candidates (unranked) after reranked
        reranked.extend(tail)
        return reranked
