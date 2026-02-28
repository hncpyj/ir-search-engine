"""
build_bm25.py — Offline Phase Step 3 (optional)

Builds Pyserini BM25 Lucene indexes for domains where bm25_enabled=true.

Prerequisites:
    java -version  # must show Java 11+
    pip install pyserini

Usage:
    python scripts/build_bm25.py --config configs/default.yaml --domain science
    python scripts/build_bm25.py --config configs/full.yaml   # all bm25-enabled domains
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.indexing.bm25_builder import BM25IndexBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("build_bm25")


def build_domain_bm25(domain: str, cfg: dict, force: bool = False) -> None:
    data_root = Path(cfg["paths"]["data_root"])
    index_root = Path(cfg["paths"]["index_root"])

    corpus_path = data_root / domain / "corpus.parquet"
    output_dir = index_root / "bm25" / domain

    if not corpus_path.exists():
        logger.error(
            f"[{domain}] corpus.parquet not found. Run build_corpus.py first."
        )
        return

    lucene_dir = output_dir / "lucene_index"
    if lucene_dir.exists() and not force:
        logger.info(
            f"[{domain}] Lucene index exists — skipping (use --force to rebuild)."
        )
        return

    builder = BM25IndexBuilder(domain=domain, cfg=cfg)
    builder.build(corpus_path, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 Lucene indexes.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)

    if args.domain:
        domains = [args.domain]
    else:
        domains = [
            d for d, dc in cfg["domains"].items()
            if dc.get("enabled", False) and dc.get("bm25_enabled", False)
        ]

    if not domains:
        logger.info("No domains have bm25_enabled=true. Nothing to do.")
        return

    logger.info(f"Building BM25 indexes for: {domains}")
    for domain in domains:
        build_domain_bm25(domain, cfg, force=args.force)

    logger.info("Done.")


if __name__ == "__main__":
    main()
