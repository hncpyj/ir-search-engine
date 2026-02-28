"""
fiqa adapter — finance domain.

Dataset: mteb/fiqa
  - config="corpus",  split="corpus"  → _id, title, text  (57 638 docs)
  - config="queries", split="queries" → _id, text          (6 648 queries)
  - config="default", split="train"/"dev"/"test" → query-id, corpus-id, score  (qrels)

Note: BeIR/fiqa uses deprecated loading scripts → broken.
      mteb/fiqa is the canonical replacement.
"""
from __future__ import annotations

import logging
from typing import Iterator

from datasets import load_dataset

from .base_adapter import BaseAdapter, Document, Query, QRel

logger = logging.getLogger(__name__)

DOMAIN = "finance"


class FiqaAdapter(BaseAdapter):
    def __init__(self, config: dict):
        super().__init__(DOMAIN, config)

    def iter_documents(self) -> Iterator[Document]:
        corpus = load_dataset(
            "mteb/fiqa",
            name="corpus",
            split="corpus",
            streaming=True,
        )
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

    def iter_queries(self, split: str = "queries") -> Iterator[Query]:
        ds = load_dataset(
            "mteb/fiqa",
            name="queries",
            split="queries",
            streaming=True,
        )
        for row in ds:
            yield Query(
                query_id=f"{DOMAIN}:{row['_id']}",
                text=row["text"],
                domain=DOMAIN,
            )

    def iter_qrels(self, split: str = "test") -> Iterator[QRel]:
        # qrels live in the "default" config; splits: train / dev / test
        qrels = load_dataset(
            "mteb/fiqa",
            name="default",
            split=split,
            streaming=True,
        )
        for row in qrels:
            yield QRel(
                query_id=f"{DOMAIN}:{row['query-id']}",
                doc_id=f"{DOMAIN}:{row['corpus-id']}",
                relevance=int(row["score"]),
            )
