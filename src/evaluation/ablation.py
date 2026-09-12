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
    generator=None,
    n_gen_queries: int = 50,
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
        generator:        Optional BaseGenerator. When provided, also computes answer-
                          change rate vs. hybrid_with_rerank as the reference condition.
                          Pass None to skip generation metrics (backwards-compatible).
        n_gen_queries:    Max queries used for answer-change rate computation (generation
                          is slow; 50 queries gives a stable estimate).

    Returns:
        DataFrame with one row per condition. Columns always include retrieval metrics;
        answer_change_rate and citation_precision are added when generator is not None.
    """
    from .metrics import evaluate_domain
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from .grounding import (
        build_condition_passages, GroundingCondition,
        compute_answer_similarity, compute_citation_precision,
    )

    if conditions is None:
        conditions = DEFAULT_CONDITIONS

    # Build reference answers (hybrid_with_rerank, normal condition) for change-rate
    reference_answers: dict[str, str] = {}
    reference_pipeline = None
    embedder = None
    gen_queries_df = queries_df

    if generator is not None:
        logger.info(f"[ablation:{domain}] Pre-computing reference answers for answer-change rate …")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        # Sample queries for generation (expensive)
        seed = base_cfg.get("system", {}).get("seed", 42)
        if len(queries_df) > n_gen_queries:
            gen_queries_df = queries_df.sample(n=n_gen_queries, random_state=seed).reset_index(drop=True)

        # Reference condition: hybrid_with_rerank (or best available)
        ref_cfg = _deep_merge(base_cfg, {"reranking": {"enabled": True}})
        reference_pipeline = pipeline_factory(ref_cfg)
        ref_results: dict[str, list] = {}
        for _, qrow in gen_queries_df.iterrows():
            qid = str(qrow["query_id"])
            try:
                r = reference_pipeline.search(str(qrow["text"]))
                ref_results[qid] = r["results"]
            except Exception as e:
                logger.warning(f"[ablation:{domain}] reference pipeline error: {e}")
                ref_results[qid] = []
        topk = base_cfg.get("generation", {}).get("topk_passages", 5)
        gen_seed = base_cfg.get("generation", {}).get("seed", 42)
        for _, qrow in gen_queries_df.iterrows():
            qid = str(qrow["query_id"])
            passages = ref_results.get(qid, [])[:topk]
            try:
                gen = generator.generate(str(qrow["text"]), passages, seed=gen_seed)
                reference_answers[qid] = gen.answer
            except Exception as e:
                logger.warning(f"[ablation:{domain}] reference generation error: {e}")
                reference_answers[qid] = ""

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

        # Optional: answer-change rate vs. reference (hybrid_with_rerank, normal context)
        if generator is not None and reference_answers:
            topk = base_cfg.get("generation", {}).get("topk_passages", 5)
            gen_seed = base_cfg.get("generation", {}).get("seed", 42)
            sims: list[float] = []
            cit_precs: list[float] = []
            for _, qrow in gen_queries_df.iterrows():
                qid = str(qrow["query_id"])
                ref_answer = reference_answers.get(qid, "")
                passages = results_by_qid.get(qid, [])[:topk]
                try:
                    gen = generator.generate(str(qrow["text"]), passages, seed=gen_seed)
                    sim = compute_answer_similarity(ref_answer, gen.answer, embedder)
                    cp = compute_citation_precision(gen.answer, passages, gen.cited_ids)
                    sims.append(sim)
                    cit_precs.append(cp)
                except Exception as e:
                    logger.warning(f"[ablation:{domain}:{cond.name}] gen error: {e}")
            threshold = base_cfg.get("grounding", {}).get("similarity_threshold", 0.90)
            if sims:
                row["answer_change_rate"] = float(np.mean([s < threshold for s in sims]))
                row["mean_answer_similarity"] = float(np.mean(sims))
                row["citation_precision"] = float(np.mean(cit_precs))
            else:
                row["answer_change_rate"] = float("nan")
                row["mean_answer_similarity"] = float("nan")
                row["citation_precision"] = float("nan")

        rows.append(row)
        logger.info(f"[ablation:{domain}] {cond.name}: {metrics}")

    return pd.DataFrame(rows)
