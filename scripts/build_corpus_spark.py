"""
build_corpus_spark.py — Distributed corpus preprocessing (PySpark path).

Drop-in parallel to build_corpus.py. Uses SparkCorpusPipeline instead of
the single-process pandas path. Output layout is identical — one
corpus.parquet per domain under data/processed/{domain}/ — so all
downstream scripts (build_faiss.py, build_bm25.py, evaluate.py) work
without modification.

Usage:
    # All enabled domains, local mode
    python scripts/build_corpus_spark.py --config configs/default.yaml

    # Single domain
    python scripts/build_corpus_spark.py --config configs/default.yaml --domain science

    # Full corpus on a Spark cluster
    python scripts/build_corpus_spark.py --config configs/full.yaml \\
        --master spark://host:7077

    # Force rebuild
    python scripts/build_corpus_spark.py --config configs/default.yaml --force
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Must be set before pyspark imports so the JVM worker picks up the same Python.
os.environ.setdefault("PYSPARK_PYTHON",        sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters import ADAPTER_MAP, TOKENIZER_MAP
from src.config import load_config
from src.spark.corpus_pipeline import SparkCorpusPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("build_corpus_spark")


def make_spark(master: str, app_name: str = "ir-corpus-pipeline") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.pyspark.python",        sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .getOrCreate()
    )


def build_domain_spark(
    domain: str,
    cfg: dict,
    data_root: Path,
    pipeline: SparkCorpusPipeline,
    force: bool = False,
) -> None:
    out_dir = data_root / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.parquet"

    if corpus_path.exists() and not force:
        logger.info(f"[{domain}] corpus.parquet exists — skipping (use --force to rebuild).")
        return

    domain_cfg = cfg["domains"][domain]
    adapter = ADAPTER_MAP[domain](domain_cfg)
    corpus_df, queries_df, qrels_df = adapter.to_dataframes(split="test")

    logger.info(f"[{domain}] {len(corpus_df):,} docs loaded — starting Spark pipeline...")
    stats = pipeline.run(corpus_df, TOKENIZER_MAP[domain], corpus_path)

    queries_df.to_parquet(out_dir / "queries.parquet",  index=False)
    qrels_df.to_parquet(  out_dir / "qrels.parquet",    index=False)

    logger.info(
        f"[{domain}] {stats['n_docs_in']:,} docs → {stats['n_chunks_out']:,} chunks "
        f"in {stats['elapsed_s']:.1f}s"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Distributed corpus build (PySpark).")
    p.add_argument("--config",   default="configs/default.yaml")
    p.add_argument("--override", default=None)
    p.add_argument("--domain",   default=None, help="Process only this domain (default: all enabled)")
    p.add_argument("--master",   default="local[*]", help="Spark master URL (default: local[*])")
    p.add_argument("--force",    action="store_true", help="Rebuild even if output exists")
    args = p.parse_args()

    cfg = load_config(args.config, args.override)
    data_root = Path(cfg["paths"]["data_root"])

    domains = (
        [args.domain]
        if args.domain
        else [d for d, dcfg in cfg["domains"].items() if dcfg.get("enabled", False)]
    )

    spark = make_spark(args.master)
    spark.sparkContext.setLogLevel("WARN")
    pipeline = SparkCorpusPipeline(spark, cfg)

    for domain in domains:
        if domain not in ADAPTER_MAP:
            logger.warning(f"No adapter for '{domain}', skipping.")
            continue
        build_domain_spark(domain, cfg, data_root, pipeline, force=args.force)

    spark.stop()
    logger.info("Done.")


if __name__ == "__main__":
    main()
