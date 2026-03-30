"""
MS MARCO adapter — general domain.

Uses mteb/msmarco which mirrors the standard BEIR benchmark format,
enabling direct comparison with published baselines (MRR@10 on dev set).

mteb/msmarco schema:
  corpus  (config="corpus",  split="corpus") : _id, title, text  (~8.8M passages)
  queries (config="queries", split="queries"): _id, text         (~6,980 dev queries)
  qrels   (config="default", split="dev")   : query-id, corpus-id, score
"""
from __future__ import annotations

import logging
from typing import Iterator

from datasets import load_dataset

from .base_adapter import BaseAdapter, Document, Query, QRel

logger = logging.getLogger(__name__)

DOMAIN = "general"


class MSMarcoAdapter(BaseAdapter):
    def __init__(self, config: dict):
        super().__init__(DOMAIN, config)

    def iter_documents(self) -> Iterator[Document]:
        corpus = load_dataset("mteb/msmarco", "corpus", split="corpus", streaming=True)
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

    def iter_queries(self, split: str = "dev") -> Iterator[Query]:
        queries = load_dataset("mteb/msmarco", "queries", split="queries", streaming=True)
        for row in queries:
            yield Query(
                query_id=f"{DOMAIN}:{row['_id']}",
                text=row["text"],
                domain=DOMAIN,
            )

    def iter_qrels(self, split: str = "dev") -> Iterator[QRel]:
        # Standard MS MARCO dev set — 6,980 queries, MRR@10 benchmark
        qrels = load_dataset("mteb/msmarco", "default", split="dev", streaming=True)
        for row in qrels:
            yield QRel(
                query_id=f"{DOMAIN}:{row['query-id']}",
                doc_id=f"{DOMAIN}:{row['corpus-id']}",
                relevance=int(row["score"]),
            )
