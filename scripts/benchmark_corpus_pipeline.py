"""
benchmark_corpus_pipeline.py — pandas vs PySpark corpus pipeline benchmark.

Measures wall-clock time for the full preprocess-and-chunk pipeline at
increasing document counts to identify the crossover point where Spark
becomes faster than the single-process pandas path.

Results are written to results/benchmarks/corpus_pipeline.json and
printed as a Markdown table.

Usage:
    # Default: general domain, sizes 50k / 500k / 1M
    python scripts/benchmark_corpus_pipeline.py

    # Custom domain and sizes
    python scripts/benchmark_corpus_pipeline.py \\
        --domain science --sizes 5000 20000 50000

    # Larger run on all available docs
    python scripts/benchmark_corpus_pipeline.py \\
        --domain general --sizes 50000 500000 1000000
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
TMP_DIR = Path("/tmp/ir_corpus_benchmark")


def _run_pandas(corpus_df: pd.DataFrame, tokenizer_name: str, cfg: dict) -> tuple[int, float]:
    chunk_cfg = cfg["chunking"]
    chunker = SlidingWindowChunker(tokenizer_name, chunk_cfg["max_tokens"], chunk_cfg["stride"])
    t0 = time.perf_counter()
    rows = list(chunker.chunk_documents(corpus_df.to_dict(orient="records")))
    return len(rows), round(time.perf_counter() - t0, 2)


def _run_spark(
    corpus_df: pd.DataFrame,
    tokenizer_name: str,
    cfg: dict,
    spark: SparkSession,
    n: int,
) -> tuple[int, float]:
    pipeline = SparkCorpusPipeline(spark, cfg)
    out_path = TMP_DIR / f"spark_out_{n}"
    stats = pipeline.run(corpus_df, tokenizer_name, out_path)
    return stats["n_chunks_out"], stats["elapsed_s"]


def main() -> None:
    p = argparse.ArgumentParser(description="pandas vs PySpark corpus pipeline benchmark")
    p.add_argument("--config",  default="configs/default.yaml")
    p.add_argument("--domain",  default="general",
                   help="Domain to benchmark (default: general)")
    p.add_argument("--sizes",   type=int, nargs="+", default=DEFAULT_SIZES,
                   help="Document counts to test (default: 50000 500000 1000000)")
    p.add_argument("--master",  default="local[*]",
                   help="Spark master URL (default: local[*])")
    p.add_argument("--output",  default="results/benchmarks/corpus_pipeline.json",
                   help="Output path for benchmark results JSON")
    args = p.parse_args()

    cfg = load_config(args.config)
    tokenizer_name = TOKENIZER_MAP[args.domain]
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {args.domain} corpus via adapter...")
    adapter = ADAPTER_MAP[args.domain](cfg["domains"][args.domain])
    full_df, _, _ = adapter.to_dataframes(split="test")
    logger.info(f"Loaded {len(full_df):,} docs total.")

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

    results = []
    header = f"{'N docs':>12} | {'pandas (s)':>12} | {'Spark (s)':>12} | {'faster':>10} | {'speedup':>8}"
    divider = "-" * len(header)
    print(f"\n{header}\n{divider}")

    for n in sorted(args.sizes):
        if n > len(full_df):
            logger.warning(f"Skipping n={n:,}: corpus only has {len(full_df):,} docs.")
            continue

        sample = full_df.sample(n=n, random_state=42).reset_index(drop=True)
        logger.info(f"\n--- n={n:,} ---")

        pandas_chunks, pandas_t = _run_pandas(sample.copy(), tokenizer_name, cfg)
        logger.info(f"  pandas → {pandas_t:.1f}s, {pandas_chunks:,} chunks")

        spark_chunks, spark_t = _run_spark(sample.copy(), tokenizer_name, cfg, spark, n)
        logger.info(f"  spark  → {spark_t:.1f}s, {spark_chunks:,} chunks")

        speedup = round(pandas_t / spark_t, 2) if spark_t > 0 else float("inf")
        faster  = "spark" if spark_t < pandas_t else "pandas"

        row = {
            "domain":                    args.domain,
            "n_docs":                    n,
            "pandas_chunks":             pandas_chunks,
            "pandas_time_s":             pandas_t,
            "spark_chunks":              spark_chunks,
            "spark_time_s":              spark_t,
            "speedup_spark_vs_pandas":   speedup,
            "faster":                    faster,
        }
        results.append(row)
        speedup_label = f"{speedup:.2f}x" if faster == "spark" else f"1/{speedup:.2f}x"
        print(f"{n:>12,} | {pandas_t:>12.1f} | {spark_t:>12.1f} | {faster:>10} | {speedup_label:>8}")

    spark.stop()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
