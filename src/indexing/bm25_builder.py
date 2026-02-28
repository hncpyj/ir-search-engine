"""
BM25 index builder — Offline Phase Step 3 (optional).

Converts a chunked corpus.parquet into Pyserini JSONL format, then
calls the Pyserini Lucene indexer via subprocess.

Requires:
  - Java 11+  (verify: java -version)
  - pyserini  (pip install pyserini)
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class BM25IndexBuilder:
    """Builds a Pyserini/Lucene BM25 index from a corpus.parquet file."""

    def __init__(self, domain: str, cfg: dict):
        self.domain = domain
        self.threads: int = cfg.get("bm25", {}).get("threads", 4)

    def corpus_to_jsonl(self, df: pd.DataFrame, jsonl_dir: Path) -> Path:
        """
        Write Pyserini-compatible JSONL: {id, contents} per line.
        Uses chunk_id as id; concatenates title + text as contents.
        Returns the path to the written JSONL file.
        """
        jsonl_dir.mkdir(parents=True, exist_ok=True)
        out_path = jsonl_dir / "corpus.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                title = (row.get("title") or "").strip()
                text = (row.get("text") or "").strip()
                contents = f"{title} {text}".strip() if title else text
                record = {"id": str(row["chunk_id"]), "contents": contents}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"[{self.domain}] Wrote {len(df)} JSONL records to {out_path}")
        return out_path

    def build_lucene_index(self, jsonl_dir: Path, index_dir: Path) -> None:
        """Invoke Pyserini Lucene indexer via subprocess."""
        index_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", str(jsonl_dir),
            "--index", str(index_dir),
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(self.threads),
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw",
        ]
        logger.info(f"[{self.domain}] Building Lucene index: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"[{self.domain}] BM25 indexing failed:\n{result.stderr}"
            )
        logger.info(f"[{self.domain}] Lucene index built at {index_dir}")

    def build(self, corpus_parquet: Path, output_dir: Path) -> None:
        """Full pipeline: load parquet → JSONL → Lucene index."""
        df = pd.read_parquet(corpus_parquet)
        jsonl_dir = output_dir / "jsonl"
        self.corpus_to_jsonl(df, jsonl_dir)
        self.build_lucene_index(jsonl_dir, output_dir / "lucene_index")
