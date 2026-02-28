"""
nfcorpus adapter — biomedical domain.

Corpus/queries: BeIR/nfcorpus  ← this one still works (no deprecated script)
QRels: BeIR/nfcorpus-qrels

BeIR/nfcorpus schema:
  corpus split (config="corpus") : _id, title, text
  queries split (config="queries"): _id, text
"""
from __future__ import annotations

import logging
from typing import Iterator

from datasets import load_dataset

from .base_adapter import BaseAdapter, Document, Query, QRel

logger = logging.getLogger(__name__)

DOMAIN = "biomedical"


class NfcorpusAdapter(BaseAdapter):
    def __init__(self, config: dict):
        super().__init__(DOMAIN, config)

    def iter_documents(self) -> Iterator[Document]:
        # BeIR/nfcorpus still works with config="corpus"
        corpus = load_dataset("BeIR/nfcorpus", "corpus", split="corpus", streaming=True)
        for i, row in enumerate(corpus):
            if self.max_docs and i >= self.max_docs:
                break
            yield Document(
                doc_id=f"{DOMAIN}:{row['_id']}",
                title=row.get("title", "") or "",
                text=row["text"],
                domain=DOMAIN,
                orig_id=str(row["_id"]),
            )

    def iter_queries(self, split: str = "test") -> Iterator[Query]:
        queries = load_dataset("BeIR/nfcorpus", "queries", split="queries", streaming=True)
        for row in queries:
            yield Query(
                query_id=f"{DOMAIN}:{row['_id']}",
                text=row["text"],
                domain=DOMAIN,
            )

    def iter_qrels(self, split: str = "test") -> Iterator[QRel]:
        qrels = load_dataset("BeIR/nfcorpus-qrels", split=split, streaming=True)
        for row in qrels:
            yield QRel(
                query_id=f"{DOMAIN}:{row['query-id']}",
                doc_id=f"{DOMAIN}:{row['corpus-id']}",
                relevance=int(row["score"]),
            )
