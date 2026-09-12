"""
eval_grounding.py — Run the grounding intervention suite.

Measures whether generated answers depend on the retrieved context or on
the model's parametric memory, using five intervention conditions:

  normal           baseline — full pipeline, top-k passages
  no_retrieval     retrieval blocking — generate with no context
  swapped_context  opposite-memory patching — passages from a different query
  random_in_domain weaker swap control — random corpus passages, same domain
  shuffled_order   position-sensitivity control — same passages, permuted

Usage:
    # Smoke test — 5 queries, 1 seed
    python scripts/eval_grounding.py --config configs/default.yaml \\
        --domain general --n-queries 5 --n-seeds 1

    # Full run — 100 queries, 3 seeds
    python scripts/eval_grounding.py --config configs/default.yaml \\
        --domain general --n-queries 100 --n-seeds 3

    # Both primary domains
    python scripts/eval_grounding.py --config configs/default.yaml \\
        --domain general science --n-queries 100 --n-seeds 3

    # Custom conditions only
    python scripts/eval_grounding.py --config configs/default.yaml \\
        --domain general --conditions normal no_retrieval swapped_context
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.pipeline.online_pipeline import SearchPipeline
from src.generation.ollama_backend import make_generator
from src.evaluation.grounding import (
    run_grounding_eval,
    grounding_summary,
    format_summary_markdown,
    compute_domain_accuracy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("eval_grounding")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grounding intervention evaluation")
    p.add_argument("--config", required=True, help="Base config YAML")
    p.add_argument("--override", default=None, help="Optional override YAML (e.g. configs/full.yaml)")
    p.add_argument(
        "--domain", nargs="+", default=["general"],
        help="Domain(s) to evaluate. Default: general",
    )
    p.add_argument(
        "--conditions", nargs="+", default=None,
        help=(
            "Grounding conditions to run. "
            "Default: reads from grounding.conditions in config "
            "(includes corrupted_context when listed there)."
        ),
    )
    p.add_argument("--n-queries", type=int, default=None, help="Max queries per domain (overrides config)")
    p.add_argument("--n-seeds", type=int, default=None, help="Repeat seeds per query (overrides config)")
    p.add_argument(
        "--prompt-mode", default=None,
        choices=["strict", "partial", "claim_verify"],
        help=(
            "Override generation.prompt_mode for all domains. "
            "Per-domain overrides in grounding.domain_overrides take precedence when this flag is absent."
        ),
    )
    p.add_argument("--output", default="results/grounding", help="Output directory")
    return p.parse_args()


def load_domain_queries(cfg: dict, domain: str, max_queries: int | None) -> pd.DataFrame:
    data_root = Path(cfg["paths"]["data_root"])
    queries_path = data_root / domain / "queries.parquet"
    qrels_path = data_root / domain / "qrels.parquet"

    if not queries_path.exists():
        raise FileNotFoundError(
            f"queries.parquet not found for domain '{domain}'. "
            "Run: python scripts/build_corpus.py --config configs/default.yaml"
        )
    if not qrels_path.exists():
        raise FileNotFoundError(f"qrels.parquet not found for domain '{domain}'.")

    queries_df = pd.read_parquet(queries_path)
    qrels_df   = pd.read_parquet(qrels_path)

    # Keep only queries with at least one qrel
    annotated_qids = set(qrels_df["query_id"].astype(str))
    queries_df = queries_df[
        queries_df["query_id"].astype(str).isin(annotated_qids)
    ].reset_index(drop=True)

    if max_queries and len(queries_df) > max_queries:
        queries_df = queries_df.sample(
            n=max_queries,
            random_state=cfg.get("system", {}).get("seed", 42),
        ).reset_index(drop=True)
        logger.info(f"[{domain}] Sampled {max_queries} queries from {len(annotated_qids)} annotated")

    return queries_df


def run_domain(
    domain: str,
    cfg: dict,
    conditions: list[str],
    output_dir: Path,
    n_queries_override: int | None,
    n_seeds_override: int | None,
    prompt_mode_override: str | None = None,
) -> None:
    import copy
    cfg = copy.deepcopy(cfg)  # isolate per-domain config mutations
    grounding_cfg = cfg.get("grounding", {})

    # Apply per-domain overrides from config (domain_overrides section)
    domain_cfg_overrides = grounding_cfg.get("domain_overrides", {}).get(domain, {})
    if domain_cfg_overrides:
        logger.info(f"[{domain}] Applying domain_overrides: {domain_cfg_overrides}")

    # Resolve prompt_mode: CLI flag > domain_overrides > global config
    pm = (
        prompt_mode_override
        or domain_cfg_overrides.get("prompt_mode")
        or cfg.get("generation", {}).get("prompt_mode", "strict")
    )
    cfg.setdefault("generation", {})["prompt_mode"] = pm
    logger.info(f"[{domain}] prompt_mode = {pm}")

    n_queries = n_queries_override or int(grounding_cfg.get("n_queries", 100))
    # Per-domain n_seeds override has lower priority than explicit CLI flag
    n_seeds = (
        n_seeds_override
        or int(domain_cfg_overrides.get("n_seeds", grounding_cfg.get("n_seeds", 3)))
    )

    cfg["grounding"]["n_queries"] = n_queries
    cfg["grounding"]["n_seeds"] = n_seeds

    logger.info(f"[{domain}] Loading queries …")
    queries_df = load_domain_queries(cfg, domain, n_queries)
    logger.info(f"[{domain}] {len(queries_df)} queries to evaluate")

    # Load qrels for end-to-end accuracy (science domain uses gold labels)
    qrels_path = Path(cfg["paths"]["data_root"]) / domain / "qrels.parquet"
    qrels_df = pd.read_parquet(qrels_path) if qrels_path.exists() else None

    logger.info(f"[{domain}] Loading pipeline …")
    pipeline = SearchPipeline(cfg)
    pipeline.load()

    logger.info(f"[{domain}] Initialising generator (prompt_mode={pm}) …")
    generator = make_generator(cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{domain}_checkpoint.parquet"

    logger.info(f"[{domain}] Running grounding eval ({len(conditions)} conditions × {n_seeds} seeds) …")
    raw_df = run_grounding_eval(
        pipeline=pipeline,
        generator=generator,
        queries_df=queries_df,
        domain=domain,
        cfg=cfg,
        conditions=conditions,
        checkpoint_path=checkpoint_path,
    )

    # Save raw results (checkpoint can be removed now)
    raw_path = output_dir / f"{domain}_raw.parquet"
    raw_df.to_parquet(raw_path, index=False)
    logger.info(f"[{domain}] Saved raw results → {raw_path}")
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info(f"[{domain}] Removed checkpoint file")

    # ── Experiment metadata (reproducibility) ────────────────────────────────
    metadata = {
        "domain":              domain,
        "model":               cfg.get("generation", {}).get("model"),
        "prompt_mode":         pm,
        "temperature":         cfg.get("generation", {}).get("temperature"),
        "max_tokens":          cfg.get("generation", {}).get("max_tokens"),
        "topk_passages":       cfg.get("generation", {}).get("topk_passages"),
        "n_queries":           len(queries_df),
        "n_seeds":             n_seeds,
        "conditions":          conditions,
        "similarity_threshold": float(grounding_cfg.get("similarity_threshold", 0.90)),
        "timestamp_utc":       datetime.now(timezone.utc).isoformat(),
    }
    meta_path = output_dir / f"{domain}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"[{domain}] Saved experiment metadata → {meta_path}")

    # ── Summary + domain accuracy ─────────────────────────────────────────────
    threshold = float(grounding_cfg.get("similarity_threshold", 0.90))
    summary = grounding_summary(raw_df, threshold)

    # End-to-end accuracy (science: claim_verify vs qrels gold labels)
    if qrels_df is not None:
        acc = compute_domain_accuracy(raw_df, qrels_df, domain)
        if acc is not None:
            summary["domain_accuracy"] = acc
            logger.info(
                f"[{domain}] Verdict accuracy: {acc.get('accuracy', float('nan')):.3f} "
                f"(n={acc.get('n_evaluated', 0)})"
            )

    summary_path = output_dir / f"{domain}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"[{domain}] Saved summary → {summary_path}")

    # Markdown report
    md = format_summary_markdown(summary, domain, len(queries_df), n_seeds)
    md_path = output_dir / f"{domain}_summary.md"
    md_path.write_text(md, encoding="utf-8")
    logger.info(f"[{domain}] Saved markdown report → {md_path}")

    # Print headline to console
    gs_v1 = summary.get("grounding_score_v1", float("nan"))
    gs_v2 = summary.get("grounding_score", float("nan"))
    fmt = lambda v: f"{v:.3f}" if v == v else "N/A"  # noqa: E731

    print(f"\n{'='*72}")
    print(f"Domain: {domain}  ({len(queries_df)} queries × {n_seeds} seeds)")
    print(f"Grounding score v2 (conditional): {fmt(gs_v2)}"
          f"   v1 (raw change rate): {fmt(gs_v1)}")
    print()
    header = (
        f"  {'Condition':<24}  {'content_chg':>11}  {'both_same':>9}"
        f"  {'abst_shift':>10}  {'both_abst':>9}"
        f"  {'cond_gs (n)':>12}  {'faith':>5}  {'cit_sup':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    ordered = [
        "normal", "no_retrieval", "swapped_context",
        "random_in_domain", "shuffled_order", "corrupted_context",
    ]
    for cond in ordered:
        s = summary.get("per_condition", {}).get(cond)
        if s is None:
            continue
        cgs   = s["conditional_grounding"]
        cgs_n = s.get("conditional_grounding_n", 0)
        faith = s["sentence_faithfulness"]
        csup  = s["citation_support"]

        if cgs != cgs:  # nan
            cgs_str = "N/A        "
        else:
            warn = "⚠" if cgs_n < 30 else " "
            cgs_str = f"{cgs:.3f}{warn}({cgs_n:>3})"

        print(
            f"  {cond:<24}"
            f"  {s['content_change_rate']:>11.3f}"
            f"  {s['both_answer_same_rate']:>9.3f}"
            f"  {s['abstention_shift_rate']:>10.3f}"
            f"  {s['both_abstain_rate']:>9.3f}"
            f"  {cgs_str:>12}"
            f"  {fmt(faith):>5}"
            f"  {fmt(csup):>7}"
        )
    print(f"{'='*72}\n")


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config, args.override)
    output_dir = Path(args.output)

    # Resolve conditions: CLI flag > config > hardcoded fallback
    conditions = args.conditions or cfg.get("grounding", {}).get("conditions", [
        "normal", "no_retrieval", "swapped_context", "random_in_domain", "shuffled_order",
    ])
    logger.info(f"Conditions: {conditions}")

    for domain in args.domain:
        if not cfg.get("domains", {}).get(domain, {}).get("enabled", False):
            logger.warning(f"Domain '{domain}' is not enabled in config — skipping")
            continue
        try:
            run_domain(
                domain=domain,
                cfg=cfg,
                conditions=conditions,
                output_dir=output_dir,
                n_queries_override=args.n_queries,
                n_seeds_override=args.n_seeds,
                prompt_mode_override=args.prompt_mode,
            )
        except Exception as e:
            logger.error(f"[{domain}] Failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
