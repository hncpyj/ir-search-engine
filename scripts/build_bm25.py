"""
build_bm25.py — Offline Phase Step 3 (optional)

Builds rank-bm25 BM25Okapi indexes (bm25.pkl) for domains where bm25_enabled=true.
No Java required.

Usage:
    python scripts/build_bm25.py --config configs/full.yaml          # all bm25-enabled domains
    python scripts/build_bm25.py --config configs/full.yaml --domain science
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

    if (output_dir / "bm25.pkl").exists() and not force:
        logger.info(
            f"[{domain}] BM25 index exists — skipping (use --force to rebuild)."
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

    # Always load default.yaml as base; treat --config as override if it differs
    default_cfg = "configs/default.yaml"
    if args.config == default_cfg:
        cfg = load_config(default_cfg, args.override)
    else:
        cfg = load_config(default_cfg, args.config)

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
