"""
search.py — CLI for single-query and batch search.

Usage:
    # Single query
    python scripts/search.py --query "what causes alzheimer's disease"

    # Custom config
    python scripts/search.py --query "ECHR fair trial rights" --config configs/default.yaml

    # Batch from file (one query per line)
    python scripts/search.py --batch queries.txt --topk 10 --output results.jsonl

    # Full config
    python scripts/search.py --query "..." --override configs/full.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.pipeline.online_pipeline import SearchPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("search")


def format_result(i: int, r, show_text: bool = True) -> str:
    text_preview = r.text[:200].replace("\n", " ") + "…" if len(r.text) > 200 else r.text
    score_parts = [f"dense={r.score:.4f}"]
    if r.fused_score:
        score_parts.append(f"fused={r.fused_score:.4f}")
    if r.colbert_score:
        score_parts.append(f"colbert={r.colbert_score:.4f}")
    lines = [
        f"[{i+1}] {r.doc_id}  [{r.domain}]  ({', '.join(score_parts)})",
    ]
    if r.title:
        lines.append(f"     Title: {r.title}")
    if show_text:
        lines.append(f"     {text_preview}")
    return "\n".join(lines)


def run_query(
    pipeline: SearchPipeline,
    query: str,
    topk: int = 10,
    verbose: bool = True,
) -> dict:
    result = pipeline.search(query)
    clf = result["classification"]
    lat = result["latency"]

    if verbose:
        print(f"\nQuery: {result['query']}")
        print(
            f"Domain: {clf.top1_domain} ({clf.top1_prob:.3f})  "
            f"| 2nd: {clf.top2_domain} ({clf.top2_prob:.3f})"
        )
        print(f"Active domains: {result['active_domains']}")
        print(f"Latency: total={lat['total_ms']:.1f}ms  "
              f"[classify={lat['classify_ms']:.1f}  "
              f"retrieve={lat['retrieve_ms']:.1f}  "
              f"fuse={lat['fuse_ms']:.1f}  "
              f"rerank={lat['rerank_ms']:.1f}]ms")
        print(f"\nTop-{topk} results:")
        for i, r in enumerate(result["results"][:topk]):
            print(format_result(i, r))
        print()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-domain retrieval search.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", default=None)
    parser.add_argument("--query", default=None, help="Single query string")
    parser.add_argument("--batch", default=None, help="File with one query per line")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--output", default=None, help="Save results to JSONL file")
    args = parser.parse_args()

    if not args.query and not args.batch:
        parser.error("Provide --query or --batch")

    cfg = load_config(args.config, args.override)
    import torch
    if not torch.cuda.is_available():
        cfg["system"]["device"] = "cpu"
        cfg["system"]["fp16"] = False
        cfg["reranking"]["enabled"] = False  # ColBERT requires CUDA
    pipeline = SearchPipeline(cfg)
    logger.info("Loading pipeline ...")
    pipeline.load()
    logger.info("Pipeline ready.")

    queries = []
    if args.query:
        queries = [args.query]
    elif args.batch:
        with open(args.batch) as f:
            queries = [line.strip() for line in f if line.strip()]

    out_file = open(args.output, "w") if args.output else None

    try:
        for q in queries:
            result = run_query(pipeline, q, topk=args.topk, verbose=True)
            if out_file:
                # Serialise results (replace dataclass with dict)
                serialisable = {
                    "query": result["query"],
                    "active_domains": result["active_domains"],
                    "top1_domain": result["classification"].top1_domain,
                    "top1_prob": result["classification"].top1_prob,
                    "latency": result["latency"],
                    "results": [
                        {
                            "rank": i + 1,
                            "doc_id": r.doc_id,
                            "domain": r.domain,
                            "score": r.score,
                            "fused_score": r.fused_score,
                            "colbert_score": r.colbert_score,
                            "title": r.title,
                            "text": r.text[:500],
                        }
                        for i, r in enumerate(result["results"][:args.topk])
                    ],
                }
                out_file.write(json.dumps(serialisable) + "\n")
    finally:
        if out_file:
            out_file.close()


if __name__ == "__main__":
    main()
