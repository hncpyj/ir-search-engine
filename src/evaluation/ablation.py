"""
Ablation harness.

Defines ablation conditions as config overrides and runs the full evaluation
under each condition, collecting metrics into a comparison DataFrame.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------

@dataclass
class AblationCondition:
    name: str
    description: str
    overrides: dict = field(default_factory=dict)


_BM25_ONLY_OVERRIDES = {
    "retrieval": {"topk_dense": 0},   # pipeline skips dense when topk=0
    "reranking": {"enabled": False},
}

_DENSE_ONLY_OVERRIDES = {
    "domains": {
        d: {"bm25_enabled": False}
        for d in ["general", "scidocs", "science", "finance", "medical", "biomedical"]
    },
    "reranking": {"enabled": False},
}

DEFAULT_CONDITIONS: list[AblationCondition] = [
    AblationCondition(
        "dense_only",
        "Dense retrieval only, no BM25, no reranking",
        _DENSE_ONLY_OVERRIDES,
    ),
    AblationCondition(
        "bm25_only",
        "BM25 only, no dense, no reranking",
        _BM25_ONLY_OVERRIDES,
    ),
    AblationCondition(
        "hybrid_no_rerank",
        "Dense + BM25 hybrid (RRF), no reranking",
        {"reranking": {"enabled": False}},
    ),
    AblationCondition(
        "hybrid_with_rerank",
        "Dense + BM25 hybrid (RRF) + ColBERT reranking",
        {"reranking": {"enabled": True}},
    ),
    AblationCondition(
        "routing_broadcast",
        "Query all domain indexes (broadcast)",
        {"routing": {"mode": "broadcast"}},
    ),
    AblationCondition(
        "routing_routed_general",
        "Routed to top-1 domain + general fallback",
        {"routing": {"mode": "routed_with_general"}},
    ),
    AblationCondition(
        "routing_routed_only",
        "Routed to top-1 domain only (no general fallback)",
        {"routing": {"mode": "routed_only"}},
    ),
]


# ---------------------------------------------------------------------------
# Config merging
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

def run_ablation(
    base_cfg: dict,
    pipeline_factory,
    domain: str,
    queries_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    conditions: list[AblationCondition] | None = None,
) -> pd.DataFrame:
    """
    Run each ablation condition for one domain and collect metrics.

    Args:
        base_cfg:         Base configuration dict.
        pipeline_factory: Callable(cfg) → SearchPipeline (already loaded).
        domain:           Domain name for logging.
        queries_df:       DataFrame with columns [query_id, text].
        qrels_df:         DataFrame with columns [query_id, doc_id, relevance].
        conditions:       List of AblationCondition; defaults to DEFAULT_CONDITIONS.

    Returns:
        DataFrame with one row per condition × metric.
    """
    from .metrics import evaluate_domain

    if conditions is None:
        conditions = DEFAULT_CONDITIONS

    rows = []
    for cond in conditions:
        # Skip reranking conditions when CUDA is not available
        if cond.overrides.get("reranking", {}).get("enabled", False):
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning(
                        f"[ablation:{domain}] Skipping {cond.name} (requires CUDA for ColBERT)"
                    )
                    continue
            except ImportError:
                logger.warning(f"[ablation:{domain}] Skipping {cond.name} (torch not available)")
                continue

        logger.info(f"[ablation:{domain}] Running condition: {cond.name}")
        cfg = _deep_merge(base_cfg, cond.overrides)
        pipeline = pipeline_factory(cfg)

        results_by_qid: dict[str, list] = {}
        for _, qrow in queries_df.iterrows():
            qid = str(qrow["query_id"])
            try:
                result = pipeline.search(str(qrow["text"]))
                results_by_qid[qid] = result["results"]
            except Exception as e:
                logger.warning(f"[ablation:{domain}:{cond.name}] query error: {e}")
                results_by_qid[qid] = []

        metrics = evaluate_domain(qrels_df, results_by_qid, domain=domain)
        row = {"domain": domain, "condition": cond.name, "description": cond.description}
        row.update(metrics)
        rows.append(row)
        logger.info(f"[ablation:{domain}] {cond.name}: {metrics}")

    return pd.DataFrame(rows)
