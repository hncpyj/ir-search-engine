"""
eval_ab.py — Statistical A/B test for grounding evaluation results.

Compares two grounding evaluation result files (control vs treatment)
and produces a Benjamini-Hochberg FDR-corrected statistical comparison report with
per-metric effect sizes, p-values, and win/loss/tie summary.

Workflow
--------
Step 1 — Run control (already done if using existing grounding results):
    python scripts/eval_grounding.py --config configs/default.yaml \\
        --domain finance --output results/grounding_v3

Step 2 — Run treatment with a different variable (e.g. prompt_mode):
    python scripts/eval_grounding.py --config configs/default.yaml \\
        --domain finance --prompt-mode partial --output results/grounding_ab

Step 3 — Compare:
    python scripts/eval_ab.py \\
        --control   results/grounding_v3/finance_raw.parquet \\
        --treatment results/grounding_ab/finance_raw.parquet \\
        --control-name strict --treatment-name partial \\
        --domain finance \\
        --output results/grounding_ab/

The script saves:
  {output}/ab_{domain}_{control_name}_vs_{treatment_name}.md   — Markdown report
  {output}/ab_{domain}_{control_name}_vs_{treatment_name}.json — machine-readable
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.ab_test import compare_grounding, format_ab_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("eval_ab")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Statistical A/B comparison of two grounding evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--control", required=True,
        help="Path to control raw parquet (e.g. results/grounding_v3/finance_raw.parquet)",
    )
    p.add_argument(
        "--treatment", required=True,
        help="Path to treatment raw parquet (e.g. results/grounding_ab/finance_raw.parquet)",
    )
    p.add_argument("--control-name",   default="control",   help="Short label for control arm")
    p.add_argument("--treatment-name", default="treatment", help="Short label for treatment arm")
    p.add_argument(
        "--domain", default=None,
        help="Domain name for report header (inferred from filename if omitted)",
    )
    p.add_argument("--alpha", type=float, default=0.05, help="FDR threshold α for Benjamini-Hochberg correction (default 0.05)")
    p.add_argument(
        "--output", default=None,
        help="Output directory for report files (default: parent of --treatment)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ctrl_path = Path(args.control)
    trt_path  = Path(args.treatment)

    for p in (ctrl_path, trt_path):
        if not p.exists():
            logger.error(f"File not found: {p}")
            sys.exit(1)

    logger.info(f"Loading control:   {ctrl_path}")
    ctrl_df = pd.read_parquet(ctrl_path)
    logger.info(f"Loading treatment: {trt_path}")
    trt_df  = pd.read_parquet(trt_path)

    domain = args.domain or trt_path.stem.replace("_raw", "").split("_")[0]
    output_dir = Path(args.output) if args.output else trt_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Comparing '{args.control_name}' vs '{args.treatment_name}' on domain '{domain}' …"
    )
    result = compare_grounding(
        ctrl_df, trt_df,
        control_name=args.control_name,
        treatment_name=args.treatment_name,
        domain=domain,
        alpha=args.alpha,
    )

    slug = f"ab_{domain}_{args.control_name}_vs_{args.treatment_name}"

    # Markdown report
    md = format_ab_report(result)
    md_path = output_dir / f"{slug}.md"
    md_path.write_text(md, encoding="utf-8")
    logger.info(f"Saved Markdown report → {md_path}")

    # JSON (machine-readable)
    json_data = {
        "control_name":   result.control_name,
        "treatment_name": result.treatment_name,
        "domain":         result.domain,
        "n_queries":      result.n_queries,
        "n_seeds":        result.n_seeds,
        "alpha":          result.alpha,
        "n_tests":        result.n_tests,
        "wins":           result.wins,
        "losses":         result.losses,
        "ties":           result.ties,
        "metrics": [
            {
                "metric":             m.metric,
                "condition":          m.condition,
                "better_is":          m.better_is,
                "control_mean":       round(m.control_mean, 6),
                "treatment_mean":     round(m.treatment_mean, 6),
                "delta":              round(m.delta, 6),
                "relative_pct":       round(m.relative_pct, 2),
                "p_value":            round(m.p_value, 6),
                "p_value_corrected":  round(m.p_value_corrected, 6),
                "effect_size":        round(m.effect_size, 4),
                "n_control":          m.n_control,
                "n_treatment":        m.n_treatment,
                "test_type":          m.test_type,
                "significant":        m.significant,
                "direction":          m.direction,
            }
            for m in result.metrics
        ],
    }
    json_path = output_dir / f"{slug}.json"
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    logger.info(f"Saved JSON report   → {json_path}")

    # Console summary
    correction_label = (
        "Benjamini-Hochberg FDR" if result.correction == "bh" else "Bonferroni FWER"
    )
    print(f"\n{'='*60}")
    print(f"A/B Test: {result.control_name!r} (ctrl) vs {result.treatment_name!r} (trt)")
    print(f"Domain: {result.domain}  |  Queries: {result.n_queries}  |  Seeds: {result.n_seeds}")
    print(f"Tests: {result.n_tests}  |  Correction: {correction_label} (α = {result.alpha})")
    print(f"{'='*60}")
    print(f"  ✅ Improvements: {result.wins}")
    print(f"  ❌ Regressions:  {result.losses}")
    print(f"  —  Ties (n.s.):  {result.ties}")
    print(f"{'='*60}")

    if result.wins or result.losses:
        print("\nSignificant findings:")
        for m in result.metrics:
            if m.direction in ("improvement", "regression"):
                arrow = "↑" if m.delta > 0 else "↓"
                mark  = "✅" if m.direction == "improvement" else "❌"
                print(
                    f"  {mark} [{m.condition}] {m.metric}: "
                    f"{m.control_mean:.3f} {arrow} {m.treatment_mean:.3f} "
                    f"(Δ={m.delta:+.3f}, p={m.p_value_corrected:.4f})"
                )
    print()


if __name__ == "__main__":
    main()
