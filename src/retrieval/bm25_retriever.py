"""
BM25 retriever — wraps rank-bm25 BM25Okapi for one domain.

Loads bm25.pkl produced by BM25IndexBuilder and the docstore.parquet
from the FAISS build step to look up metadata.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .dense_retriever import RetrievalResult

logger = logging.getLogger(__name__)


class BM25Retriever:
    """rank-bm25 BM25Okapi retriever for a single domain."""

    def __init__(
        self,
        domain: str,
        bm25_index_dir: Path,
        docstore_path: Path,
    ):
        self.domain = domain
        self.bm25_index_dir = Path(bm25_index_dir)
        self.docstore_path = Path(docstore_path)

        self._bm25 = None
        self._chunk_ids: list[str] = []
        self._docstore: Optional[pd.DataFrame] = None
        self._chunk_to_row: dict[str, int] = {}

    def load(self) -> None:
        with open(self.bm25_index_dir / "bm25.pkl", "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunk_ids = data["chunk_ids"]
        self._docstore = pd.read_parquet(self.docstore_path).reset_index(drop=True)
        self._chunk_to_row = {
            str(cid): int(i)
            for i, cid in enumerate(self._docstore["chunk_id"])
        }
        logger.info(
            f"[{self.domain}] BM25Retriever loaded "
            f"({len(self._chunk_ids)} passages) from {self.bm25_index_dir / 'bm25.pkl'}"
        )

    def search(self, query: str, topk: int = 100) -> list[RetrievalResult]:
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_idxs = np.argpartition(scores, -min(topk, len(scores)))[-topk:]
        top_idxs = top_idxs[np.argsort(scores[top_idxs])[::-1]]

        results = []
        for idx in top_idxs:
            if scores[idx] <= 0:
                continue
            chunk_id = self._chunk_ids[idx]
            row_idx = self._chunk_to_row.get(chunk_id)
            if row_idx is None:
                continue
            row = self._docstore.iloc[row_idx]
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                doc_id=str(row["doc_id"]),
                domain=self.domain,
                score=0.0,
                text=str(row.get("text", "")),
                title=str(row.get("title", "")),
                bm25_score=float(scores[idx]),
            ))
        return results
