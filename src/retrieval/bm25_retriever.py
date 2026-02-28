"""
BM25 retriever — wraps Pyserini LuceneSearcher for one domain.

Requires the docstore.parquet from the FAISS build step to look up
metadata (text, title, doc_id) for hit docids.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .dense_retriever import RetrievalResult

logger = logging.getLogger(__name__)


class BM25Retriever:
    """Pyserini BM25 retriever for a single domain."""

    def __init__(
        self,
        domain: str,
        lucene_index_dir: Path,
        docstore_path: Path,
    ):
        self.domain = domain
        self.lucene_index_dir = Path(lucene_index_dir)
        self.docstore_path = Path(docstore_path)

        self._searcher = None
        self._docstore: Optional[pd.DataFrame] = None
        self._chunk_to_row: dict[str, int] = {}

    def load(self) -> None:
        from pyserini.search.lucene import LuceneSearcher
        self._searcher = LuceneSearcher(str(self.lucene_index_dir))
        self._docstore = pd.read_parquet(self.docstore_path)
        self._chunk_to_row = {
            str(cid): int(i)
            for i, cid in self._docstore["chunk_id"].items()
        }
        logger.info(f"[{self.domain}] BM25Retriever loaded from {self.lucene_index_dir}")

    def search(self, query: str, topk: int = 1000) -> list[RetrievalResult]:
        hits = self._searcher.search(query, k=topk)
        results = []
        for hit in hits:
            chunk_id = str(hit.docid)
            row_idx = self._chunk_to_row.get(chunk_id)
            if row_idx is None:
                continue
            row = self._docstore.iloc[row_idx]
            r = RetrievalResult(
                chunk_id=chunk_id,
                doc_id=str(row["doc_id"]),
                domain=self.domain,
                score=0.0,
                text=str(row.get("text", "")),
                title=str(row.get("title", "")),
                bm25_score=float(hit.score),
            )
            r.bm25_score = float(hit.score)
            results.append(r)
        return results
