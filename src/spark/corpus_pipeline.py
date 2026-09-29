"""
PySpark distributed corpus preprocessing pipeline.

Replaces the pandas build_corpus.py path for large-scale ingestion.
Uses mapInPandas to apply the exact same SlidingWindowChunker logic
across distributed partitions — the tokenizer is initialised once per
executor partition, not once per document.

Flow: ingest → clean → deduplicate → chunk → write
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    LongType,
    StringType,
    StructField,
    StructType,
)

logger = logging.getLogger(__name__)

CHUNK_SCHEMA = StructType([
    StructField("chunk_id",    StringType(), nullable=False),
    StructField("doc_id",      StringType(), nullable=False),
    StructField("orig_id",     StringType(), nullable=True),
    StructField("domain",      StringType(), nullable=False),
    StructField("title",       StringType(), nullable=True),
    StructField("text",        StringType(), nullable=False),
    StructField("start_token", LongType(),   nullable=False),
    StructField("end_token",   LongType(),   nullable=False),
])

_CHUNK_COLS = [f.name for f in CHUNK_SCHEMA]


def _make_chunk_fn(tokenizer_name: str, max_tokens: int, stride: int):
    """
    Returns a mapInPandas-compatible function.
    SlidingWindowChunker is constructed once per executor partition,
    not once per document — this amortises the tokenizer load cost.
    """
    def _chunk_partition(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        import os
        # Tokenizer is already cached by the driver; skip hub version-check on workers.
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from src.preprocessing.chunker import SlidingWindowChunker  # lazy import on executor
        chunker = SlidingWindowChunker(tokenizer_name, max_tokens, stride)
        for pdf in iterator:
            rows: list[dict] = []
            for _, row in pdf.iterrows():
                for chunk in chunker.chunk_document(row.to_dict()):
                    rows.append(chunk)
            if rows:
                out = pd.DataFrame(rows, columns=_CHUNK_COLS)
                out["start_token"] = out["start_token"].astype("int64")
                out["end_token"]   = out["end_token"].astype("int64")
                yield out
            else:
                yield pd.DataFrame(columns=_CHUNK_COLS)

    return _chunk_partition


class SparkCorpusPipeline:
    """
    Distributed corpus preprocessing pipeline.
    Output layout matches build_corpus.py exactly so downstream
    FAISS and BM25 builders require no changes.
    """

    def __init__(self, spark: SparkSession, cfg: dict):
        self.spark = spark
        self.cfg = cfg
        chunk_cfg = cfg["chunking"]
        self.max_tokens: int = chunk_cfg["max_tokens"]
        self.stride: int = chunk_cfg["stride"]

    # ── Ingestion ─────────────────────────────────────────────────────────

    def ingest(self, corpus_df: pd.DataFrame, n_partitions: int | None = None) -> DataFrame:
        """Convert pandas corpus DataFrame to Spark DataFrame."""
        required = {"doc_id", "text", "domain"}
        missing = required - set(corpus_df.columns)
        if missing:
            raise ValueError(f"corpus_df missing required columns: {missing}")
        for col in ("title", "orig_id"):
            if col not in corpus_df.columns:
                corpus_df[col] = ""

        partitions = n_partitions or self.spark.sparkContext.defaultParallelism * 2
        df = self.spark.createDataFrame(corpus_df).repartition(partitions)
        logger.info(f"Ingested {len(corpus_df):,} docs → {partitions} Spark partitions")
        return df

    # ── Cleaning ──────────────────────────────────────────────────────────

    def clean(self, df: DataFrame) -> DataFrame:
        """Strip whitespace, fill nulls, drop empty-text rows."""
        df = df.withColumn("text",  F.trim(F.coalesce(F.col("text"),  F.lit(""))))
        df = df.withColumn("title", F.trim(F.coalesce(F.col("title"), F.lit(""))))
        return df.filter(F.length("text") > 0)

    # ── Deduplication ─────────────────────────────────────────────────────

    def deduplicate(self, df: DataFrame) -> DataFrame:
        """Keep first occurrence of each doc_id."""
        return df.dropDuplicates(["doc_id"])

    # ── Chunking ──────────────────────────────────────────────────────────

    def chunk(self, df: DataFrame, tokenizer_name: str) -> DataFrame:
        """
        Sliding-window chunking via mapInPandas.
        Tokenizer is instantiated once per Spark partition (not per document).
        """
        fn = _make_chunk_fn(tokenizer_name, self.max_tokens, self.stride)
        return df.mapInPandas(fn, schema=CHUNK_SCHEMA)

    # ── Write ─────────────────────────────────────────────────────────────

    def write_parquet(self, df: DataFrame, output_path: str | Path) -> None:
        """Write corpus parquet — single file to match existing layout."""
        (
            df.coalesce(1)
            .write
            .mode("overwrite")
            .parquet(str(output_path))
        )
        logger.info(f"Written → {output_path}")

    # ── Full pipeline ─────────────────────────────────────────────────────

    def run(
        self,
        corpus_df: pd.DataFrame,
        tokenizer_name: str,
        output_path: str | Path,
        n_partitions: int | None = None,
    ) -> dict:
        """
        Run full pipeline. Returns timing and chunk-count stats.
        The action (write + count) triggers full Spark execution.
        """
        t0 = time.perf_counter()

        df = self.ingest(corpus_df, n_partitions=n_partitions)
        df = self.clean(df)
        df = self.deduplicate(df)
        df = self.chunk(df, tokenizer_name)
        self.write_parquet(df, output_path)

        elapsed = time.perf_counter() - t0
        n_chunks = self.spark.read.parquet(str(output_path)).count()

        return {
            "n_docs_in":   len(corpus_df),
            "n_chunks_out": n_chunks,
            "elapsed_s":   round(elapsed, 2),
        }
