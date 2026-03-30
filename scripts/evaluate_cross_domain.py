"""
evaluate_cross_domain.py — FeB4RAG Tier 3 cross-domain evaluation.

Evaluates the multi-domain pipeline on cross-domain queries using the FeB4RAG
framework (Wang et al., 2024, arXiv:2402.11891):

  Tier 1: Per-domain BEIR nDCG@10          → scripts/evaluate.py
  Tier 2: Resource selection accuracy       → scripts/evaluate_routing.py
  Tier 3: Cross-domain fusion quality       → this script

Tier 3 uses an LLM (via local Ollama) as a relevance judge to produce weak
labels, following the LLM-as-judge methodology (Fridman et al., Elastic
Research, 2024; TREC 2024 LLMJudge track).  Judgements are cached so the LLM
is only called once per (query, document) pair.

Usage:
    # LLM judge (default, requires Ollama running locally)
    python scripts/evaluate_cross_domain.py --config configs/default.yaml

    # Cross-encoder fallback (no Ollama required)
    python scripts/evaluate_cross_domain.py --judge cross_encoder

    # Domain metrics only, no judge
    python scripts/evaluate_cross_domain.py --judge none

    # Save results
    python scripts/evaluate_cross_domain.py --output results/cross_domain.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.config import load_config
from src.pipeline.online_pipeline import SearchPipeline
from src.evaluation.cross_domain_eval import (
    load_cross_domain_queries,
    evaluate_cross_domain,
    OllamaJudge,
    CrossEncoderJudge,
    OLLAMA_DEFAULT_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_CACHE_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("eval_cross_domain")


def load_cfg(config_path: str, override: str | None) -> dict:
    default = "configs/default.yaml"
    cfg = load_config(default, override if override else (config_path if config_path != default else None))
    if not torch.cuda.is_available():
        cfg["system"]["device"] = "cpu"
        cfg["system"]["fp16"] = False
    cfg["reranking"]["enabled"] = False  # Tier 3 is retrieval-only
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="FeB4RAG Tier 3 cross-domain evaluation.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--judge", choices=["llm", "cross_encoder", "none"], default="llm",
        help="Relevance judge: llm (Ollama, default), cross_encoder, or none (domain metrics only).",
    )
    parser.add_argument("--ollama-model", default=OLLAMA_DEFAULT_MODEL,
                        help=f"Ollama model name (default: {OLLAMA_DEFAULT_MODEL})")
    parser.add_argument("--ollama-url", default=OLLAMA_BASE_URL,
                        help=f"Ollama base URL (default: {OLLAMA_BASE_URL})")
    parser.add_argument("--ollama-cache", default=str(OLLAMA_CACHE_PATH),
                        help="Path to LLM judgement cache JSONL file.")
    parser.add_argument("--output", default=None, help="Save JSON results to path.")
    args = parser.parse_args()

    cfg = load_cfg(args.config, args.override)
    device = cfg["system"]["device"]

    logger.info("Loading retrieval pipeline…")
    pipeline = SearchPipeline(cfg)
    pipeline.load()

    # --- Build judge ---
    judge = None
    judge_desc = "none (domain metrics only)"

    if args.judge == "llm":
        judge = OllamaJudge(
            model=args.ollama_model,
            base_url=args.ollama_url,
            cache_path=Path(args.ollama_cache),
        )
        judge.load()
        judge_desc = f"LLM via Ollama ({args.ollama_model})"

    elif args.judge == "cross_encoder":
        judge = CrossEncoderJudge(device=device)
        judge.load()
        judge_desc = f"cross-encoder ({judge.model_name})"

    queries = load_cross_domain_queries()
    logger.info(f"Running {len(queries)} cross-domain queries (top-{args.topk})")

    results_by_qid: dict[str, list] = {}
    for q in queries:
        out = pipeline.search(q["text"])
        results_by_qid[q["query_id"]] = out["results"]
        domains = {r.domain for r in out["results"][:args.topk]}
        logger.info(f"  {q['query_id']}: '{q['text'][:55]}' → {sorted(domains)}")

    logger.info("Computing Tier 3 metrics…")
    metrics = evaluate_cross_domain(
        results_by_qid, queries, judge=judge, top_n=args.topk
    )

    print("\n=== FeB4RAG Tier 3 — Cross-Domain Evaluation ===")
    print(f"  Methodology: FeB4RAG (Wang et al., 2024)")
    print(f"  Judge:       {judge_desc}")
    print(f"  Queries:     {len(queries)}")
    print()
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"judge": judge_desc, "metrics": metrics}, f, indent=2)
        logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
