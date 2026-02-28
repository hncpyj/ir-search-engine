"""
build_faiss.py — Offline Phase Step 2

Encodes chunked corpus for one (or all) domain(s) and builds FAISS indexes.

Usage:
    python scripts/build_faiss.py --config configs/default.yaml --domain science
    python scripts/build_faiss.py --config configs/default.yaml   # all domains
    python scripts/build_faiss.py --config configs/default.yaml --override configs/full.yaml --domain general
"""
from __future__ import annotations

import multiprocessing
import os

# macOS: sentence-transformers 내부 DataLoader worker가 fork() 충돌을 일으킴
# → tokenizer 병렬화 비활성화 + spawn 방식 강제
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("spawn")

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.indexing.faiss_builder import FAISSIndexBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("build_faiss")


def build_domain(domain: str, cfg: dict, force: bool = False) -> None:
    data_root = Path(cfg["paths"]["data_root"])
    index_root = Path(cfg["paths"]["index_root"])

    corpus_path = data_root / domain / "corpus.parquet"
    output_dir = index_root / "faiss" / domain

    if not corpus_path.exists():
        logger.error(
            f"[{domain}] corpus.parquet not found at {corpus_path}. "
            "Run build_corpus.py first."
        )
        return

    index_path = output_dir / "faiss.index"
    if index_path.exists() and not force:
        logger.info(
            f"[{domain}] faiss.index already exists — skipping "
            "(use --force to rebuild)."
        )
        return

    builder = FAISSIndexBuilder(domain=domain, cfg=cfg)
    builder.build(corpus_path, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-domain FAISS indexes.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", default=None)
    parser.add_argument("--domain", default=None, help="Single domain to build")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)

    domains = (
        [args.domain]
        if args.domain
        else [d for d, dc in cfg["domains"].items() if dc.get("enabled", False)]
    )

    logger.info(f"Building FAISS indexes for: {domains}")
    for domain in domains:
        build_domain(domain, cfg, force=args.force)

    logger.info("Done.")


if __name__ == "__main__":
    main()
