"""
benchmark_corpus_pipeline.py — pandas vs PySpark corpus pipeline benchmark.

Measures wall-clock time for the full preprocess-and-chunk pipeline at
increasing document counts to identify the crossover point where Spark
becomes faster than the single-process pandas path.

Results are written to results/benchmarks/corpus_pipeline.json and
printed as a Markdown table.

Usage:
    # All enabled domains (auto-sized per domain)
    python scripts/benchmark_corpus_pipeline.py

    # Single domain, explicit sizes
    python scripts/benchmark_corpus_pipeline.py \\
        --domain science --sizes 1000 3000 5000

    # Subset of domains
    python scripts/benchmark_corpus_pipeline.py \\
        --domain general finance medical
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Must be set before pyspark imports so the JVM worker picks up the same Python.
os.environ.setdefault("PYSPARK_PYTHON",        sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import pandas as pd
from pyspark.sql import SparkSession

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters import ADAPTER_MAP, TOKENIZER_MAP
from src.config import load_config
from src.preprocessing.chunker import SlidingWindowChunker
from src.spark.corpus_pipeline import SparkCorpusPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] — %(message)s")
logger = logging.getLogger("benchmark")

DEFAULT_SIZES = [50_000, 500_000, 1_000_000]
# Fractions of corpus used when --sizes is not explicitly provided
AUTO_SIZE_FRACTIONS = [0.25, 0.50, 0.75, 1.0]
TMP_DIR = Path("/tmp/ir_corpus_benchmark")


def _run_pandas(corpus_df: pd.DataFrame, chunker: SlidingWindowChunker) -> tuple[int, float]:
    t0 = time.perf_counter()
    rows = list(chunker.chunk_documents(corpus_df.to_dict(orient="records")))
    return len(rows), round(time.perf_counter() - t0, 2)


def _run_spark(
    corpus_df: pd.DataFrame,
    tokenizer_name: str,
    cfg: dict,
    spark: SparkSession,
    domain: str,
    n: int,
) -> tuple[int, float]:
    pipeline = SparkCorpusPipeline(spark, cfg)
    out_path = TMP_DIR / f"spark_out_{domain}_{n}"
    stats = pipeline.run(corpus_df, tokenizer_name, out_path)
    return stats["n_chunks_out"], stats["elapsed_s"]


def _auto_sizes(n_total: int) -> list[int]:
    """Compute 4 evenly-spaced sizes from 25% to 100% of corpus, minimum 100 docs each."""
    return sorted({max(100, int(n_total * f)) for f in AUTO_SIZE_FRACTIONS})


def _bench_domain(
    domain: str,
    cfg: dict,
    explicit_sizes: list[int] | None,
    spark: SparkSession,
) -> list[dict]:
    tokenizer_name = TOKENIZER_MAP[domain]

    logger.info(f"\n{'='*60}")
    logger.info(f"  DOMAIN: {domain.upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"Loading {domain} corpus via adapter...")
    adapter = ADAPTER_MAP[domain](cfg["domains"][domain])
    full_df, _, _ = adapter.to_dataframes(split="test")
    logger.info(f"Loaded {len(full_df):,} docs total.")

    sizes = explicit_sizes if explicit_sizes is not None else _auto_sizes(len(full_df))

    chunk_cfg = cfg["chunking"]
    chunker = SlidingWindowChunker(tokenizer_name, chunk_cfg["max_tokens"], chunk_cfg["stride"])
    logger.info(f"[{domain}] Tokenizer loaded: {tokenizer_name}")

    header = f"{'N docs':>12} | {'pandas (s)':>12} | {'Spark (s)':>12} | {'faster':>10} | {'speedup':>8}"
    divider = "-" * len(header)
    print(f"\n[{domain}] {header}\n{divider}")

    rows = []
    for n in sorted(sizes):
        if n > len(full_df):
            logger.warning(f"[{domain}] Skipping n={n:,}: corpus only has {len(full_df):,} docs.")
            continue

        sample = full_df.sample(n=n, random_state=42).reset_index(drop=True)
        logger.info(f"[{domain}] n={n:,} ...")

        pandas_chunks, pandas_t = _run_pandas(sample.copy(), chunker)
        spark_chunks,  spark_t  = _run_spark(sample.copy(), tokenizer_name, cfg, spark, domain, n)

        speedup = round(pandas_t / spark_t, 2) if spark_t > 0 else float("inf")
        faster  = "spark" if spark_t < pandas_t else "pandas"

        rows.append({
            "domain":                  domain,
            "n_docs":                  n,
            "pandas_chunks":           pandas_chunks,
            "pandas_time_s":           pandas_t,
            "spark_chunks":            spark_chunks,
            "spark_time_s":            spark_t,
            "speedup_spark_vs_pandas": speedup,
            "faster":                  faster,
        })
        speedup_label = f"{speedup:.2f}x" if faster == "spark" else f"1/{speedup:.2f}x"
        print(f"[{domain}] {n:>12,} | {pandas_t:>12.1f} | {spark_t:>12.1f} | {faster:>10} | {speedup_label:>8}")

    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="pandas vs PySpark corpus pipeline benchmark")
    p.add_argument("--config",  default="configs/default.yaml")
    p.add_argument("--domain",  nargs="*", default=None,
                   help="Domain(s) to benchmark (default: all enabled in config)")
    p.add_argument("--sizes",   type=int, nargs="+", default=None,
                   help="Explicit document counts to test per domain (default: auto 25%%/50%%/75%%/100%% of corpus)")
    p.add_argument("--master",  default="local[*]",
                   help="Spark master URL (default: local[*])")
    p.add_argument("--output",  default="results/benchmarks/corpus_pipeline.json",
                   help="Output path for benchmark results JSON")
    args = p.parse_args()

    cfg = load_config(args.config)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    domains = (
        args.domain
        if args.domain
        else [d for d, dcfg in cfg["domains"].items() if dcfg.get("enabled", False)]
    )
    logger.info(f"Benchmarking domains: {domains}")

    spark = (
        SparkSession.builder
        .appName("corpus-pipeline-benchmark")
        .master(args.master)
        .config("spark.pyspark.python",        sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    all_results = []
    for domain in domains:
        if domain not in ADAPTER_MAP:
            logger.warning(f"No adapter for '{domain}', skipping.")
            continue
        rows = _bench_domain(domain, cfg, args.sizes, spark)
        all_results.extend(rows)

    spark.stop()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    logger.info(f"\nResults saved → {out_path} ({len(all_results)} rows across {len(domains)} domains)")


if __name__ == "__main__":
    main()
