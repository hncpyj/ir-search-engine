"""
OPT-B: Evaluate RRF k values (30, 60, 100) with BM25 enabled.

Runs the full evaluation for each k value, saves per-domain and macro metrics,
then prints a comparison table.

Usage:
    python3 scripts/eval_rrf_k.py --max-queries 300 --domains all
    python3 scripts/eval_rrf_k.py --max-queries 300 --domains science finance
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.pipeline.online_pipeline import SearchPipeline
from src.evaluation.metrics import evaluate_domain

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("eval_rrf_k")

K_VALUES = [30, 60, 100]
ALL_DOMAINS = ["general", "scidocs", "science", "finance", "medical", "biomedical"]


def load_domain_data(cfg: dict, domain: str, max_queries: int | None):
    data_root = Path(cfg["paths"]["data_root"])
    queries_df = pd.read_parquet(data_root / domain / "queries.parquet")
    qrels_df = pd.read_parquet(data_root / domain / "qrels.parquet")

    annotated_qids = set(qrels_df["query_id"].astype(str))
    queries_df = queries_df[queries_df["query_id"].astype(str).isin(annotated_qids)].reset_index(drop=True)

    if max_queries and len(queries_df) > max_queries:
        queries_df = queries_df.sample(n=max_queries, random_state=42).reset_index(drop=True)
        sampled_qids = set(queries_df["query_id"].astype(str))
        qrels_df = qrels_df[qrels_df["query_id"].astype(str).isin(sampled_qids)]

    return queries_df, qrels_df


def run_domain(pipeline: SearchPipeline, queries_df, qrels_df, domain: str) -> dict:
    results_by_qid = {}
    for _, row in queries_df.iterrows():
        qid = str(row["query_id"])
        try:
            r = pipeline.search(str(row["text"]))
            results_by_qid[qid] = r["results"]
        except Exception as e:
            logger.warning(f"Query error {qid}: {e}")
            results_by_qid[qid] = []
    return evaluate_domain(qrels_df, results_by_qid, domain=domain)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", default="configs/full.yaml")
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--domains", nargs="+", default=["all"])
    parser.add_argument("--output", default="results/rrf_k_comparison")
    parser.add_argument("--k-values", nargs="+", type=int, default=K_VALUES)
    args = parser.parse_args()

    domains = ALL_DOMAINS if args.domains == ["all"] else args.domains
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(args.config, args.override)

    # Disable CUDA (use CPU/MPS)
    import torch
    if not torch.cuda.is_available():
        base_cfg["system"]["device"] = "cpu"
        base_cfg["system"]["fp16"] = False

    # Preload domain data once
    logger.info("Loading domain query/qrel data ...")
    domain_data = {}
    for domain in domains:
        try:
            q, qr = load_domain_data(base_cfg, domain, args.max_queries)
            domain_data[domain] = (q, qr)
            logger.info(f"  {domain}: {len(q)} queries, {len(qr)} qrels")
        except Exception as e:
            logger.warning(f"  {domain}: skipped — {e}")

    # Test each k value
    all_results: dict[int, dict[str, dict]] = {}  # k -> domain -> metrics

    for k in args.k_values:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing k={k} ...")
        cfg = copy.deepcopy(base_cfg)
        cfg["retrieval"]["rrf_k"] = k

        pipeline = SearchPipeline(cfg)
        pipeline.load()

        k_results = {}
        for domain, (queries_df, qrels_df) in domain_data.items():
            logger.info(f"  [{domain}] {len(queries_df)} queries ...")
            metrics = run_domain(pipeline, queries_df, qrels_df, domain)
            k_results[domain] = metrics
            logger.info(f"  [{domain}] nDCG@10={metrics.get('nDCG@10', 0):.4f}")

        # Macro average
        import numpy as np
        all_keys = set(k for m in k_results.values() for k in m)
        macro = {k2: float(np.mean([m[k2] for m in k_results.values() if k2 in m])) for k2 in all_keys}
        k_results["macro"] = macro
        logger.info(f"  [macro] nDCG@10={macro.get('nDCG@10', 0):.4f}")

        all_results[k] = k_results

        # Save per-k results
        k_dir = out_dir / f"k{k}"
        k_dir.mkdir(exist_ok=True)
        for domain, metrics in k_results.items():
            with open(k_dir / f"{domain}_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

        # Free pipeline memory
        del pipeline

    # Print comparison table
    print("\n" + "="*70)
    print("OPT-B: RRF k Comparison (nDCG@10)")
    print("="*70)
    header = f"{'Domain':<15}" + "".join(f"  k={k:<8}" for k in args.k_values)
    print(header)
    print("-"*70)

    for domain in list(domains) + ["macro"]:
        row = f"{domain:<15}"
        for k in args.k_values:
            val = all_results.get(k, {}).get(domain, {}).get("nDCG@10", 0)
            row += f"  {val:.4f}   "
        print(row)

    # Save comparison JSON
    comparison = {
        "k_values": args.k_values,
        "domains": domains,
        "nDCG@10": {
            str(k): {d: all_results[k][d].get("nDCG@10", 0) for d in list(domains) + ["macro"]}
            for k in args.k_values
        },
        "R@100": {
            str(k): {d: all_results[k][d].get("R@100", 0) for d in list(domains) + ["macro"]}
            for k in args.k_values
        },
    }
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"\nComparison saved to {out_dir}/comparison.json")

    # Best k
    macro_ndcg = {k: all_results[k]["macro"].get("nDCG@10", 0) for k in args.k_values}
    best_k = max(macro_ndcg, key=macro_ndcg.get)
    print(f"\nBest k={best_k} (macro nDCG@10={macro_ndcg[best_k]:.4f})")


if __name__ == "__main__":
    main()
