"""
Evaluation metrics.

Wraps ir-measures to compute nDCG@10, MRR@10, Recall@100/1000.
Also measures per-stage query latency.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IR metrics via ir-measures
# ---------------------------------------------------------------------------

def _build_run(results_by_qid: dict[str, list]) -> list:
    """
    Convert {query_id: [RetrievalResult, ...]} to ir_measures ScoredDoc list.
    Deduplicates by doc_id within each query (keeps highest score).
    """
    from ir_measures import ScoredDoc
    scored = []
    for qid, results in results_by_qid.items():
        best: dict[str, float] = {}
        for r in results:
            if r.doc_id not in best or r.score > best[r.doc_id]:
                best[r.doc_id] = r.score
            # Use ColBERT score if available (it overrides fused/dense)
            if r.colbert_score and r.colbert_score > 0:
                best[r.doc_id] = r.colbert_score
        for doc_id, score in best.items():
            scored.append(ScoredDoc(qid, doc_id, score))
    return scored


def evaluate_domain(
    qrels_df: pd.DataFrame,
    results_by_qid: dict[str, list],
    domain: str = "?",
) -> dict[str, float]:
    """
    Compute retrieval metrics for one domain.

    Args:
        qrels_df:        DataFrame with columns [query_id, doc_id, relevance].
        results_by_qid:  {query_id: [RetrievalResult, ...]}
        domain:          Domain name for logging.

    Returns:
        {metric_name: float} e.g. {"nDCG@10": 0.42, "MRR@10": 0.35, ...}
    """
    import ir_measures
    from ir_measures import nDCG, Recall, MRR

    if qrels_df.empty:
        logger.warning(f"[eval:{domain}] qrels_df is empty — skipping.")
        return {}

    if not results_by_qid:
        logger.warning(f"[eval:{domain}] No results — skipping.")
        return {}

    # Write qrels to temp TREC format in memory
    qrels_records = qrels_df.rename(
        columns={"query_id": "query_id", "doc_id": "doc_id", "relevance": "relevance"}
    )
    qrels = ir_measures.read_trec_qrels(
        "\n".join(
            f"{row.query_id} 0 {row.doc_id} {row.relevance}"
            for row in qrels_records.itertuples()
        )
    )

    run_scored = _build_run(results_by_qid)
    if not run_scored:
        return {}

    measures = [nDCG @ 10, MRR @ 10, Recall @ 100, Recall @ 1000]

    try:
        agg = ir_measures.calc_aggregate(measures, qrels, run_scored)
        out = {str(m): float(v) for m, v in agg.items()}
    except Exception as e:
        logger.error(f"[eval:{domain}] ir_measures error: {e}")
        out = {}

    logger.info(f"[eval:{domain}] {out}")
    return out


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------

def measure_latency(
    search_fn: Callable[[str], dict],
    queries: list[str],
    n_trials: int = 100,
    warmup: int = 5,
) -> dict[str, float]:
    """
    Measure per-stage latency statistics over n_trials queries.

    Returns:
        {"{stage}_mean_ms": float, "{stage}_p95_ms": float, ...}
    """
    # Warmup
    for q in queries[:warmup]:
        search_fn(q)

    stage_keys = [
        "normalize_ms", "classify_ms", "retrieve_ms",
        "fuse_ms", "rerank_ms", "total_ms",
    ]
    stage_times: dict[str, list[float]] = {k: [] for k in stage_keys}

    trial_queries = (queries * ((n_trials // len(queries)) + 1))[:n_trials]
    for q in trial_queries:
        r = search_fn(q)
        lat = r.get("latency", {})
        for k in stage_keys:
            if k in lat:
                stage_times[k].append(lat[k])

    stats: dict[str, float] = {}
    for k, times in stage_times.items():
        if times:
            arr = np.array(times)
            stats[f"{k}_mean"] = float(arr.mean())
            stats[f"{k}_p95"] = float(np.percentile(arr, 95))
    return stats
