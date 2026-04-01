"""
MS MARCO adapter — general domain.

Uses mteb/msmarco which mirrors the standard BEIR benchmark format,
enabling direct comparison with published baselines (MRR@10 on dev set).

mteb/msmarco schema:
  corpus  (config="corpus",  split="corpus") : _id, title, text  (~8.8M passages)
  queries (config="queries", split="queries"): _id, text         (~6,980 dev queries)
  qrels   (config="default", split="dev")   : query-id, corpus-id, score

Corpus loading strategy (when max_docs is set):
  Step 1 — Pre-collect ALL dev-relevant passages first (guarantees qrel coverage).
  Step 2 — Stream corpus to fill the remainder up to max_docs.
  This ensures evaluation metrics are valid even with a capped corpus.
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

    def _relevant_ids(self) -> set[str]:
        """Collect all corpus-ids referenced in the dev qrels (small set, ~7k)."""
        logger.info("[MSMarco] Pre-collecting dev-relevant passage IDs from qrels ...")
        qrels = load_dataset("mteb/msmarco", "default", split="dev", streaming=True)
        ids = {row["corpus-id"] for row in qrels}
        logger.info(f"[MSMarco] Found {len(ids):,} relevant passage IDs.")
        return ids

    def iter_documents(self) -> Iterator[Document]:
        relevant_ids = self._relevant_ids()

        def _make_doc(row) -> Document:
            orig_id = str(row["_id"])
            return Document(
                doc_id=f"{DOMAIN}:{orig_id}",
                title=row.get("title", "") or "",
                text=row["text"],
                domain=DOMAIN,
                orig_id=orig_id,
            )

        if not self.max_docs:
            # No cap: single pass
            corpus = load_dataset("mteb/msmarco", "corpus", split="corpus", streaming=True)
            for row in corpus:
                yield _make_doc(row)
            return

        # Pass 1: collect all relevant passages (guaranteed qrel coverage)
        logger.info("[MSMarco] Pass 1 — collecting relevant passages ...")
        relevant_docs: dict[str, Document] = {}
        corpus = load_dataset("mteb/msmarco", "corpus", split="corpus", streaming=True)
        for row in corpus:
            orig_id = str(row["_id"])
            if orig_id in relevant_ids and orig_id not in relevant_docs:
                relevant_docs[orig_id] = _make_doc(row)
            if len(relevant_docs) == len(relevant_ids):
                break  # found them all
        logger.info(f"[MSMarco] Pass 1 done — {len(relevant_docs):,} relevant passages collected.")

        for doc in relevant_docs.values():
            yield doc

        # Pass 2: fill remainder with random passages
        remainder = self.max_docs - len(relevant_docs)
        if remainder <= 0:
            return
        logger.info(f"[MSMarco] Pass 2 — filling {remainder:,} additional passages ...")
        corpus = load_dataset("mteb/msmarco", "corpus", split="corpus", streaming=True)
        count = 0
        for row in corpus:
            if count >= remainder:
                break
            orig_id = str(row["_id"])
            if orig_id in relevant_docs:
                continue  # already yielded
            yield _make_doc(row)
            count += 1
        logger.info(f"[MSMarco] Pass 2 done — {count:,} additional passages added.")

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
