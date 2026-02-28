"""
isaacus/echr-retrieval adapter — legal domain.

Three configs (confirmed by inspection):
  "corpus"  split="corpus"  → _id, title (empty), text
  "queries" split="queries" → _id, text
  "default" split="test"    → query-id, corpus-id, score  (qrels)

IDs use string prefixes: "query_NNN" / "passage_NNN".
"""
from __future__ import annotations

import logging
from typing import Iterator

from datasets import load_dataset

from .base_adapter import BaseAdapter, Document, Query, QRel

logger = logging.getLogger(__name__)

DOMAIN = "legal"


class EchrAdapter(BaseAdapter):
    def __init__(self, config: dict):
        super().__init__(DOMAIN, config)

    def iter_documents(self) -> Iterator[Document]:
        corpus = load_dataset(
            "isaacus/echr-retrieval", "corpus", split="corpus", streaming=True
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

    def iter_queries(self, split: str = "test") -> Iterator[Query]:
        queries = load_dataset(
            "isaacus/echr-retrieval", "queries", split="queries", streaming=True
        )
        for row in queries:
            yield Query(
                query_id=f"{DOMAIN}:{row['_id']}",
                text=row["text"],
                domain=DOMAIN,
            )

    def iter_qrels(self, split: str = "test") -> Iterator[QRel]:
        # "default" config holds the qrels in split="test"
        qrels = load_dataset(
            "isaacus/echr-retrieval", "default", split=split, streaming=True
        )
        for row in qrels:
            yield QRel(
                query_id=f"{DOMAIN}:{row['query-id']}",
                doc_id=f"{DOMAIN}:{row['corpus-id']}",
                relevance=int(float(row["score"])),
            )
