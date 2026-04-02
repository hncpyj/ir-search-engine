"""
trec-covid adapter — medical domain.

Dataset: mteb/trec-covid
  - config="corpus",  split="corpus"  → _id, title, text  (171 332 docs)
  - config="queries", split="queries" → _id, text          (50 queries)
  - config="default", split="test"    → query-id, corpus-id, score  (66 336 qrels)

Note: BeIR/trec-covid uses deprecated loading scripts → broken.
      mteb/trec-covid is the canonical replacement.

Two-pass loading (BUG-2 fix):
  TREC-COVID has 171k documents but evaluates against 35k+ relevance-judged docs.
  Naive top-N streaming excludes most relevant documents.
  Pass 1: collect all qrel-referenced corpus-ids (tiny, streaming qrels only).
  Pass 2: yield relevant docs first, then fill remainder up to max_docs.
"""
from __future__ import annotations

import logging
from typing import Iterator

from datasets import load_dataset

from .base_adapter import BaseAdapter, Document, Query, QRel

logger = logging.getLogger(__name__)

DOMAIN = "medical"


class TrecCovidAdapter(BaseAdapter):
    def __init__(self, config: dict):
        super().__init__(DOMAIN, config)

    def _relevant_ids(self) -> set[str]:
        """Stream qrels once to collect all corpus-ids that appear in relevance judgements."""
        qrels = load_dataset(
            "mteb/trec-covid", name="default", split="test", streaming=True
        )
        return {str(row["corpus-id"]) for row in qrels}

    def iter_documents(self) -> Iterator[Document]:
        def _make_doc(row: dict) -> Document:
            return Document(
                doc_id=f"{DOMAIN}:{row['_id']}",
                title=row.get("title", "") or "",
                text=row["text"],
                domain=DOMAIN,
                orig_id=str(row["_id"]),
            )

        if not self.max_docs:
            # No cap — stream everything
            corpus = load_dataset(
                "mteb/trec-covid", name="corpus", split="corpus", streaming=True
            )
            for row in corpus:
                yield _make_doc(row)
            return

        # Two-pass: guarantee all qrel-relevant docs are included within the cap.
        logger.info(f"[{DOMAIN}] Two-pass loading (max_docs={self.max_docs}): "
                    "collecting qrel-relevant IDs first ...")
        relevant_ids = self._relevant_ids()
        logger.info(f"[{DOMAIN}] {len(relevant_ids)} qrel-relevant corpus IDs found.")

        corpus_stream = load_dataset(
            "mteb/trec-covid", name="corpus", split="corpus", streaming=True
        )

        # Pass 1: collect all relevant docs (may be < max_docs)
        relevant_docs: dict[str, Document] = {}
        for row in corpus_stream:
            if str(row["_id"]) in relevant_ids:
                relevant_docs[str(row["_id"])] = _make_doc(row)

        logger.info(f"[{DOMAIN}] Pass 1 complete: {len(relevant_docs)} relevant docs collected.")
        for doc in relevant_docs.values():
            yield doc

        # Pass 2: fill remaining quota with non-relevant docs
        remaining = self.max_docs - len(relevant_docs)
        if remaining <= 0:
            return

        corpus_stream2 = load_dataset(
            "mteb/trec-covid", name="corpus", split="corpus", streaming=True
        )
        count = 0
        for row in corpus_stream2:
            if count >= remaining:
                break
            if str(row["_id"]) not in relevant_docs:
                yield _make_doc(row)
                count += 1

        logger.info(f"[{DOMAIN}] Pass 2 complete: {count} additional docs yielded "
                    f"(total={len(relevant_docs) + count}).")

    def iter_queries(self, split: str = "queries") -> Iterator[Query]:
        ds = load_dataset(
            "mteb/trec-covid",
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
        # qrels live in the "default" config, "test" split
        qrels = load_dataset(
            "mteb/trec-covid",
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
