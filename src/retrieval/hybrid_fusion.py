"""
Hybrid fusion — Reciprocal Rank Fusion (RRF) across multiple ranked lists.

Handles results from multiple domains and/or multiple modalities (dense + BM25).
Deduplicates by doc_id (keeps best metadata from highest-score chunk).
"""
from __future__ import annotations

from collections import defaultdict

from .dense_retriever import RetrievalResult


class RRFFusion:
    """
    Reciprocal Rank Fusion.

    score_RRF(doc) = sum_over_lists( 1 / (k + rank_in_list) )

    After fusion the returned RetrievalResult.score holds the RRF score;
    individual dense/bm25 scores are preserved in the best-matching result.
    """

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def fuse(
        self,
        ranked_lists: list[list[RetrievalResult]],
        topk: int = 1000,
    ) -> list[RetrievalResult]:
        """
        Args:
            ranked_lists: list of ranked result lists (each from one domain/modality).
            topk: maximum number of merged results to return.

        Returns:
            Single merged list sorted by RRF score descending,
            deduplicated by doc_id.
        """
        rrf_scores: dict[str, float] = defaultdict(float)
        best_result: dict[str, RetrievalResult] = {}

        for ranked_list in ranked_lists:
            for rank, result in enumerate(ranked_list):
                rrf_score = 1.0 / (self.rrf_k + rank + 1)
                rrf_scores[result.doc_id] += rrf_score

                # Keep the result with highest raw score as representative
                if (
                    result.doc_id not in best_result
                    or result.score > best_result[result.doc_id].score
                ):
                    best_result[result.doc_id] = result

        sorted_ids = sorted(
            rrf_scores, key=lambda d: rrf_scores[d], reverse=True
        )[:topk]

        merged = []
        for doc_id in sorted_ids:
            r = best_result[doc_id]
            merged.append(
                RetrievalResult(
                    chunk_id=r.chunk_id,
                    doc_id=r.doc_id,
                    domain=r.domain,
                    score=r.score,
                    text=r.text,
                    title=r.title,
                    bm25_score=r.bm25_score,
                    fused_score=rrf_scores[doc_id],
                    colbert_score=0.0,
                )
            )
        return merged
