"""
SciDocs adapter — scidocs domain.

Dataset: mteb/scidocs
  - config="corpus",  split="corpus"  → _id, title, text  (25,657 docs)
  - config="queries", split="queries" → _id, text          (1,000 queries)
  - config="default", split="test"    → query-id, corpus-id, score  (qrels)
"""
from __future__ import annotations

import logging
from typing import Iterator

from datasets import load_dataset

from .base_adapter import BaseAdapter, Document, Query, QRel

logger = logging.getLogger(__name__)

DOMAIN = "scidocs"


class ScidocsAdapter(BaseAdapter):
    def __init__(self, config: dict):
        super().__init__(DOMAIN, config)

    def iter_documents(self) -> Iterator[Document]:
        corpus = load_dataset(
            "mteb/scidocs",
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
            "mteb/scidocs",
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
        qrels = load_dataset(
            "mteb/scidocs",
            name="default",
            split="test",
            streaming=True,
        )
        for row in qrels:
            yield QRel(
                query_id=f"{DOMAIN}:{row['query-id']}",
                doc_id=f"{DOMAIN}:{row['corpus-id']}",
                relevance=int(row["score"]),
            )
