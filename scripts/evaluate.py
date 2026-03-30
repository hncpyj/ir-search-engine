"""
evaluate.py — Run retrieval evaluation per domain.

Loads queries + qrels for a domain, runs the full pipeline,
computes metrics, optionally runs ablation, and saves results.

Usage:
    # Standard evaluation for one domain
    python scripts/evaluate.py --config configs/default.yaml --domain science

    # All enabled domains
    python scripts/evaluate.py --config configs/default.yaml

    # Full config + ablation table
    python scripts/evaluate.py --config configs/default.yaml --override configs/full.yaml --ablation

    # Measure latency
    python scripts/evaluate.py --config configs/default.yaml --domain science --latency
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.pipeline.online_pipeline import SearchPipeline
from src.evaluation.metrics import evaluate_domain, measure_latency
from src.evaluation.ablation import run_ablation, DEFAULT_CONDITIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("evaluate")


def load_pipeline(cfg: dict) -> SearchPipeline:
    pipeline = SearchPipeline(cfg)
    pipeline.load()
    return pipeline


def evaluate_single_domain(
    domain: str,
    cfg: dict,
    pipeline: SearchPipeline,
    results_root: Path,
    run_ablation_flag: bool = False,
    measure_lat: bool = False,
    max_queries: int | None = None,
    domain_filter: bool = False,
) -> dict[str, float]:
    data_root = Path(cfg["paths"]["data_root"])
    queries_path = data_root / domain / "queries.parquet"
    qrels_path = data_root / domain / "qrels.parquet"

    if not queries_path.exists():
        logger.error(f"[eval:{domain}] queries.parquet not found. Run build_corpus.py.")
        return {}
    if not qrels_path.exists():
        logger.error(f"[eval:{domain}] qrels.parquet not found. Run build_corpus.py.")
        return {}

    queries_df = pd.read_parquet(queries_path)
    qrels_df = pd.read_parquet(qrels_path)

    # Cap queries to those that have at least one qrel (avoids unannotated queries)
    annotated_qids = set(qrels_df["query_id"].astype(str))
    queries_df = queries_df[queries_df["query_id"].astype(str).isin(annotated_qids)].reset_index(drop=True)

    if max_queries and len(queries_df) > max_queries:
        queries_df = queries_df.sample(n=max_queries, random_state=42).reset_index(drop=True)
        logger.info(f"[eval:{domain}] Sampled {max_queries} queries (from {len(annotated_qids)} annotated)")

    logger.info(
        f"[eval:{domain}] Evaluating {len(queries_df)} queries "
        f"against {len(qrels_df)} qrels ..."
    )

    results_by_qid: dict[str, list] = {}
    for _, qrow in queries_df.iterrows():
        qid = str(qrow["query_id"])
        try:
            result = pipeline.search(str(qrow["text"]))
            hits = result["results"]
            # domain_filter: keep only results from the target domain
            # (gives BEIR-comparable single-domain scores)
            if domain_filter:
                hits = [r for r in hits if r.domain == domain]
            results_by_qid[qid] = hits
        except Exception as e:
            logger.warning(f"[eval:{domain}] query error for {qid}: {e}")
            results_by_qid[qid] = []

    tag = f"{domain}_domain_only" if domain_filter else domain
    metrics = evaluate_domain(qrels_df, results_by_qid, domain=tag)
    label = f"{domain.upper()} Metrics" + (" (domain-only)" if domain_filter else "")
    print(f"\n=== {label} ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save standard metrics
    results_root.mkdir(parents=True, exist_ok=True)
    fname = f"{domain}_domain_only_metrics.json" if domain_filter else f"{domain}_metrics.json"
    with open(results_root / fname, "w") as f:
        json.dump(metrics, f, indent=2)

    # Latency
    if measure_lat:
        lat_cfg = cfg.get("evaluation", {})
        query_texts = queries_df["text"].tolist()
        lat = measure_latency(
            pipeline.search,
            query_texts,
            n_trials=lat_cfg.get("latency_trials", 50),
            warmup=lat_cfg.get("latency_warmup", 5),
        )
        print(f"\n=== {domain.upper()} Latency (ms) ===")
        for k, v in lat.items():
            print(f"  {k}: {v:.2f}")
        with open(results_root / f"{domain}_latency.json", "w") as f:
            json.dump(lat, f, indent=2)

    # Ablation
    if run_ablation_flag:
        logger.info(f"[eval:{domain}] Running ablation conditions ...")
        ablation_df = run_ablation(
            base_cfg=cfg,
            pipeline_factory=load_pipeline,
            domain=domain,
            queries_df=queries_df,
            qrels_df=qrels_df,
        )
        ablation_path = results_root / f"{domain}_ablation.csv"
        ablation_df.to_csv(ablation_path, index=False)
        print(f"\n=== {domain.upper()} Ablation ===")
        print(ablation_df.to_string(index=False))
        logger.info(f"[eval:{domain}] Ablation saved to {ablation_path}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multi-domain retrieval.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", default=None)
    parser.add_argument("--domain", default=None, help="Single domain (default: all enabled)")
    parser.add_argument("--ablation", action="store_true", help="Run ablation conditions")
    parser.add_argument("--latency", action="store_true", help="Measure query latency")
    parser.add_argument("--output", default=None, help="Results directory (overrides config)")
    parser.add_argument("--max-queries", type=int, default=None, help="Cap queries per domain (default: all)")
    parser.add_argument("--domain-filter", action="store_true",
                        help="Filter results to target domain only (BEIR-comparable single-domain scores)")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    import torch
    if not torch.cuda.is_available():
        cfg["system"]["device"] = "cpu"
        cfg["system"]["fp16"] = False
        cfg["reranking"]["enabled"] = False  # ColBERT requires CUDA
    results_root = Path(args.output or cfg["paths"]["results_root"])

    domains = (
        [args.domain]
        if args.domain
        else [d for d, dc in cfg["domains"].items() if dc.get("enabled", False)]
    )

    logger.info("Loading pipeline ...")
    pipeline = load_pipeline(cfg)
    logger.info("Pipeline ready.")

    all_metrics: dict[str, dict] = {}
    for domain in domains:
        m = evaluate_single_domain(
            domain=domain,
            cfg=cfg,
            pipeline=pipeline,
            results_root=results_root,
            run_ablation_flag=args.ablation,
            measure_lat=args.latency,
            max_queries=args.max_queries,
            domain_filter=args.domain_filter,
        )
        if m:
            all_metrics[domain] = m

    # Macro-average across domains
    if len(all_metrics) > 1:
        import numpy as np
        all_keys = set(k for m in all_metrics.values() for k in m)
        macro = {}
        for k in all_keys:
            vals = [m[k] for m in all_metrics.values() if k in m]
            macro[k] = float(np.mean(vals))
        print("\n=== MACRO-AVERAGE ===")
        for k, v in macro.items():
            print(f"  {k}: {v:.4f}")
        with open(results_root / "macro_average.json", "w") as f:
            json.dump(macro, f, indent=2)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
