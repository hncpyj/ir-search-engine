"""
BM25 index builder — Offline Phase Step 3 (optional).

Builds a rank-bm25 BM25Okapi index from a corpus.parquet file and
saves it as bm25.pkl in the output directory. No Java required.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25IndexBuilder:
    """Builds a rank-bm25 BM25Okapi index from a corpus.parquet file."""

    def __init__(self, domain: str, cfg: dict):
        self.domain = domain

    def build(self, corpus_parquet: Path, output_dir: Path) -> None:
        """Load parquet → tokenise → fit BM25Okapi → save pickle."""
        df = pd.read_parquet(corpus_parquet)

        texts: list[list[str]] = []
        chunk_ids: list[str] = []

        for _, row in df.iterrows():
            title = (row.get("title") or "").strip()
            text = (row.get("text") or "").strip()
            combined = f"{title} {text}".strip() if title else text
            texts.append(combined.lower().split())
            chunk_ids.append(str(row["chunk_id"]))

        logger.info(f"[{self.domain}] Fitting BM25 on {len(texts)} passages…")
        bm25 = BM25Okapi(texts)

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "bm25.pkl"
        with open(out_path, "wb") as f:
            pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)

        logger.info(f"[{self.domain}] BM25 index saved to {out_path}")
