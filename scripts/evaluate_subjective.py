"""
evaluate_subjective.py — Subjective retrieval quality analysis.

Runs the pipeline on a selected set of queries and outputs a Markdown report
with ranked result samples for qualitative analysis.  Intended for inclusion
in the coursework report as per the examiner's guidance to "consolidate the
findings from the three tiers into an overall evaluation."

The script selects queries across different themes (general, single-domain
specialist, cross-domain, expected-failure) to give a diverse picture of
system behaviour.  Analysis placeholders are included for manual completion.

Usage:
    python scripts/evaluate_subjective.py --config configs/default.yaml
    python scripts/evaluate_subjective.py --config configs/default.yaml \\
        --output results/subjective_eval.md --topk 5
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.config import load_config
from src.pipeline.online_pipeline import SearchPipeline
from src.evaluation.cross_domain_eval import load_cross_domain_queries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("eval_subjective")

# Query IDs selected for subjective analysis — one per evaluation theme:
#   xd_01 — biomedical+medical (strong domain signal, expected to do well)
#   xd_05 — finance+science (interdisciplinary, quantitative)
#   xd_10 — science+scidocs (paper recommendation, tests scidocs routing)
#   xd_12 — medical+science+biomedical (three-domain cross)
#   xd_22 — science+general (CS query, expected difficulty: general domain coverage)
DEFAULT_QUERY_IDS = ["xd_01", "xd_05", "xd_10", "xd_12", "xd_22"]


def _truncate(text: str, max_len: int = 200) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:max_len] + "…" if len(text) > max_len else text


def format_result_table(results: list, topk: int) -> str:
    header = "| Rank | Domain | Score | Title | Snippet |\n"
    sep    = "|------|--------|-------|-------|---------|\n"
    rows = []
    for i, r in enumerate(results[:topk], 1):
        title   = _truncate(r.title or "(no title)", 60)
        snippet = _truncate(r.text, 150)
        score   = r.colbert_score if hasattr(r, "colbert_score") and r.colbert_score else getattr(r, "dense_score", 0.0)
        rows.append(f"| {i} | {r.domain} | {score:.4f} | {title} | {snippet} |")
    return header + sep + "\n".join(rows)


def run_query(pipeline: SearchPipeline, query_text: str, topk: int) -> dict:
    out = pipeline.search(query_text)
    clf = out.get("classification")
    return {
        "results": out["results"],
        "top1_domain": clf.top1_domain if clf else "n/a",
        "top1_prob": clf.top1_prob if clf else 0.0,
        "top2_domain": clf.top2_domain if clf else "n/a",
        "top2_prob": clf.top2_prob if clf else 0.0,
        "active_domains": out.get("active_domains", []),
        "latency_ms": out.get("latency", {}).get("total_ms", 0.0),
    }


def build_report(
    selected_queries: list[dict],
    pipeline: SearchPipeline,
    topk: int,
) -> str:
    lines = [
        "# Subjective Retrieval Quality Analysis\n",
        "This report presents sample retrieval results for qualitative analysis, "
        "consolidating the findings from the three-tier evaluation framework "
        "(Tier 1: per-domain BEIR, Tier 2: routing accuracy, Tier 3: cross-domain fusion).\n",
        "Queries are selected to cover a range of scenarios: single-domain specialist "
        "queries, interdisciplinary cross-domain queries, and expected-failure cases.\n",
        "---\n",
    ]

    for q in selected_queries:
        qid  = q["query_id"]
        text = q["text"]
        expected = q.get("expected_domains", [])

        logger.info(f"Running query {qid}: {text}")
        res = run_query(pipeline, text, topk)

        domains_in_top = sorted({r.domain for r in res["results"][:topk]})
        active = res["active_domains"]

        lines += [
            f"## {qid}: \"{text}\"\n",
            f"**Expected domains:** {', '.join(expected)}  ",
            f"**Routing decision:** {res['top1_domain']} (conf: {res['top1_prob']:.2f})"
            + (f", {res['top2_domain']} (conf: {res['top2_prob']:.2f})" if res['top2_prob'] > 0.1 else ""),
            f"  \n**Active domains queried:** {', '.join(active)}  ",
            f"**Domains in top-{topk}:** {', '.join(domains_in_top)}  ",
            f"**Latency:** {res['latency_ms']:.1f} ms\n",
            format_result_table(res["results"], topk),
            "\n**Analysis:**",
            "> TODO: Write 2–3 sentences analysing: "
            "(1) correctness of the routing decision, "
            "(2) relevance of top results, "
            "(3) any notable failures or surprises, "
            "(4) one actionable observation.\n",
            "---\n",
        ]

    lines += [
        "## Overall Observations\n",
        "> TODO: Write a brief paragraph (4–6 sentences) summarising system "
        "strengths and limitations observed across all five queries. "
        "Reference specific examples from the queries above.\n",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Subjective retrieval quality analysis.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", default=None)
    parser.add_argument("--topk", type=int, default=5, help="Results per query (default: 5)")
    parser.add_argument(
        "--query-ids", nargs="+", default=DEFAULT_QUERY_IDS,
        help="Query IDs to evaluate (from data/cross_domain/queries.jsonl)",
    )
    parser.add_argument(
        "--output", default="results/subjective_eval.md",
        help="Output Markdown file path (default: results/subjective_eval.md)",
    )
    args = parser.parse_args()

    default = "configs/default.yaml"
    cfg = load_config(
        default,
        args.override if args.override else (args.config if args.config != default else None),
    )
    if not torch.cuda.is_available():
        cfg["system"]["device"] = "cpu"
        cfg["system"]["fp16"] = False

    logger.info("Loading pipeline…")
    pipeline = SearchPipeline(cfg)
    pipeline.load()

    all_queries = load_cross_domain_queries()
    query_map = {q["query_id"]: q for q in all_queries}

    selected = []
    for qid in args.query_ids:
        if qid not in query_map:
            logger.warning(f"Query ID '{qid}' not found in queries.jsonl — skipping.")
            continue
        selected.append(query_map[qid])

    if not selected:
        logger.error("No valid query IDs found. Exiting.")
        sys.exit(1)

    logger.info(f"Analysing {len(selected)} queries, top-{args.topk} results each…")
    report = build_report(selected, pipeline, args.topk)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info(f"Report saved to {out_path}")
    print(f"\nDone. Open {out_path} and fill in the TODO analysis sections.")


if __name__ == "__main__":
    main()
