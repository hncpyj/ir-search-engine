"""
evaluate_routing.py — FeB4RAG Tier 2: Resource selection accuracy.

Evaluates the domain classifier as a retrieval resource selector, following
the RouterRetriever analysis (Lee et al., AAAI 2025) which identifies routing
error as the primary gap between actual and oracle retrieval performance.

Ground truth: each per-domain BEIR query has a known correct domain label.
We sample up to --max-per-domain queries per domain, run the classifier,
and report:

  top1_accuracy   — correct domain is the top-1 prediction
  top2_accuracy   — correct domain is in top-2 predictions
  macro_f1        — macro-averaged F1 across domains
  per_domain_acc  — per-domain top-1 accuracy breakdown
  confidence_dist — mean top-1 confidence per domain

Methodology reference:
  Lee et al. (2025). RouterRetriever. AAAI 2025.
  — Oracle routing ceiling: upper bound when ground-truth domain is used.
  — ~30% of retrieval error attributable to routing mistakes.

Usage:
    python scripts/evaluate_routing.py --config configs/default.yaml
    python scripts/evaluate_routing.py --config configs/default.yaml \\
        --max-per-domain 300 --output results/routing_eval.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.classification.domain_classifier import DomainClassifier
from src.classification.query_split import is_eval_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("eval_routing")


def load_domain_queries(
    data_root: Path,
    domains: list[str],
    max_per_domain: int,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """
    Load (query_text, true_domain) pairs from per-domain BEIR parquet files.
    Only queries that appear in qrels are used (test set queries).
    """
    rng = np.random.default_rng(seed)
    samples: list[tuple[str, str]] = []

    for domain in domains:
        q_path  = data_root / domain / "queries.parquet"
        qr_path = data_root / domain / "qrels.parquet"
        if not q_path.exists() or not qr_path.exists():
            logger.warning(f"[routing] Missing data for domain '{domain}', skipping.")
            continue

        queries_df = pd.read_parquet(q_path)
        qrels_df   = pd.read_parquet(qr_path)

        # Keep only qrel-annotated queries that fall in the held-out eval split.
        # The same split is enforced in train_classifier.py so there is no leakage.
        qrel_qids = set(qrels_df["query_id"].astype(str).unique())
        qid_str = queries_df["query_id"].astype(str)
        is_qrel = qid_str.isin(qrel_qids)
        is_eval_split = qid_str.map(is_eval_query)
        eval_df = queries_df[is_qrel & is_eval_split].copy()

        if len(eval_df) == 0:
            logger.warning(f"[routing] No eval queries for domain '{domain}'.")
            continue

        # Sample up to max_per_domain
        n = min(max_per_domain, len(eval_df))
        idx = rng.choice(len(eval_df), size=n, replace=False)
        sampled = eval_df.iloc[idx]

        for _, row in sampled.iterrows():
            samples.append((str(row["text"]), domain))

        logger.info(f"[routing] {domain}: {n} queries sampled (total eval={len(eval_df)})")

    return samples


def evaluate_routing(
    classifier: DomainClassifier,
    samples: list[tuple[str, str]],
) -> dict:
    """
    Run classifier on all samples and compute routing accuracy metrics.
    Returns dict of metrics + per-domain breakdown.
    """
    from sklearn.metrics import f1_score

    true_labels: list[str] = []
    pred_top1: list[str]   = []
    pred_top2: list[str]   = []
    top1_confs: dict[str, list[float]] = {}

    for query_text, true_domain in samples:
        result = classifier.classify(query_text)
        true_labels.append(true_domain)
        pred_top1.append(result.top1_domain)
        pred_top2.append(result.top2_domain)
        top1_confs.setdefault(true_domain, []).append(result.top1_prob)

    # Top-1 accuracy
    top1_correct = [t == p for t, p in zip(true_labels, pred_top1)]
    top1_acc = float(np.mean(top1_correct))

    # Top-2 accuracy (correct domain is top-1 OR top-2)
    top2_correct = [
        t == p1 or t == p2
        for t, p1, p2 in zip(true_labels, pred_top1, pred_top2)
    ]
    top2_acc = float(np.mean(top2_correct))

    # Macro F1
    domains = sorted(set(true_labels))
    macro_f1 = float(f1_score(true_labels, pred_top1, labels=domains, average="macro", zero_division=0))

    # Per-domain accuracy
    per_domain: dict[str, dict] = {}
    for domain in domains:
        mask = [t == domain for t in true_labels]
        domain_true = [true_labels[i] for i, m in enumerate(mask) if m]
        domain_pred = [pred_top1[i]   for i, m in enumerate(mask) if m]
        acc = float(np.mean([t == p for t, p in zip(domain_true, domain_pred)])) if domain_true else 0.0
        per_domain[domain] = {
            "top1_accuracy":    round(acc, 4),
            "n_queries":        len(domain_true),
            "mean_confidence":  round(float(np.mean(top1_confs.get(domain, [0.0]))), 4),
        }

    return {
        "top1_accuracy":  round(top1_acc,  4),
        "top2_accuracy":  round(top2_acc,  4),
        "macro_f1":       round(macro_f1,  4),
        "n_total":        len(samples),
        "per_domain":     per_domain,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FeB4RAG Tier 2: routing evaluation.")
    parser.add_argument("--config",          default="configs/default.yaml")
    parser.add_argument("--override",        default=None)
    parser.add_argument("--max-per-domain",  type=int, default=200,
                        help="Max queries per domain to evaluate (default: 200).")
    parser.add_argument("--output",          default=None)
    args = parser.parse_args()

    default = "configs/default.yaml"
    cfg = load_config(default, args.override if args.override else
                      (args.config if args.config != default else None))

    data_root = Path(cfg["paths"]["data_root"])
    enabled_domains = [d for d, dc in cfg["domains"].items() if dc.get("enabled", False)]
    logger.info(f"Enabled domains: {enabled_domains}")

    # Load classifier
    clf_cfg = cfg["classification"]
    classifier = DomainClassifier(
        model_path=clf_cfg["model_path"],
        device=cfg["system"]["device"],
        max_length=clf_cfg["max_length"],
    )
    classifier.load()

    # Load evaluation queries
    samples = load_domain_queries(
        data_root=data_root,
        domains=enabled_domains,
        max_per_domain=args.max_per_domain,
        seed=cfg["system"]["seed"],
    )
    logger.info(f"Total evaluation pairs: {len(samples)}")

    # Evaluate
    metrics = evaluate_routing(classifier, samples)

    print("\n=== FeB4RAG Tier 2 — Resource Selection (Routing) Accuracy ===")
    print(f"  Methodology: RouterRetriever oracle analysis (Lee et al., AAAI 2025)")
    print(f"  Classifier:  DistilBERT (Sanh et al., 2019), fine-tuned on domain queries")
    print(f"  n_total:     {metrics['n_total']} queries across {len(enabled_domains)} domains")
    print()
    print(f"  top1_accuracy : {metrics['top1_accuracy']:.4f}")
    print(f"  top2_accuracy : {metrics['top2_accuracy']:.4f}")
    print(f"  macro_f1      : {metrics['macro_f1']:.4f}")
    print()
    print("  Per-domain breakdown:")
    for domain, dm in metrics["per_domain"].items():
        print(f"    {domain:12s}: acc={dm['top1_accuracy']:.3f}  "
              f"conf={dm['mean_confidence']:.3f}  n={dm['n_queries']}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
