"""
MS MARCO v2.1 passage ranking adapter.

Corpus: passages are embedded inside query rows as 'passages' dict.
Queries/QRels: from the 'validation' split (dev set, ~6,980 queries).

Actual v2.1 parquet schema (no 'passage_id' field!):
  passages.passage_text : list[str]
  passages.is_selected  : list[int]   (1 = relevant, 0 = not relevant)
  passages.url          : list[str]

We use (query_id, passage_index) as a stable surrogate passage ID,
and deduplicate by (url, text) hash across queries.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Iterator

from datasets import load_dataset

from .base_adapter import BaseAdapter, Document, Query, QRel

logger = logging.getLogger(__name__)

DOMAIN = "general"


def _passage_id(url: str, text: str) -> str:
    """Stable 16-char hex ID from URL + text content."""
    key = f"{url}\x00{text}"
    return hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()[:16]


class MSMarcoAdapter(BaseAdapter):
    def __init__(self, config: dict):
        super().__init__(DOMAIN, config)
        self.subset = config.get("subset", "v2.1")

    def _load(self, split: str):
        return load_dataset(
            "microsoft/ms_marco",
            self.subset,
            split=split,
            streaming=True,
        )

    def _validation_relevant_passages(self) -> dict[str, str]:
        """
        Pre-collect all relevant (is_selected=1) passages from the validation
        split so we can ensure they are included in the corpus.
        Returns {orig_id: text}.
        """
        logger.info("[MSMarco] Pre-collecting validation relevant passages ...")
        relevant: dict[str, str] = {}
        for row in self._load("validation"):
            passages = row["passages"]
            texts = passages["passage_text"]
            urls  = passages.get("url", [""] * len(texts))
            for text, url, is_sel in zip(texts, urls, passages["is_selected"]):
                if int(is_sel) > 0:
                    oid = _passage_id(url, text)
                    relevant[oid] = text
        logger.info(f"[MSMarco] Found {len(relevant)} validation-relevant passages.")
        return relevant

    def iter_documents(self) -> Iterator[Document]:
        """
        Yield corpus documents.
        Step 1: yield all validation-relevant passages first (ensures qrel coverage).
        Step 2: stream train split to fill up to max_docs.
        Deduplicates by (url, text) hash throughout.
        """
        seen_ids: set[str] = set()
        count = 0

        # ── Step 1: validation relevant passages ──────────────────────────
        for orig_id, text in self._validation_relevant_passages().items():
            if orig_id in seen_ids:
                continue
            seen_ids.add(orig_id)
            yield Document(
                doc_id=f"{DOMAIN}:{orig_id}",
                title="",
                text=text,
                domain=DOMAIN,
                orig_id=orig_id,
            )
            count += 1

        # ── Step 2: fill remainder from train split ────────────────────────
        for row in self._load("train"):
            if self.max_docs and count >= self.max_docs:
                logger.info(f"[MSMarco] Reached max_docs={self.max_docs}, stopping.")
                return
            passages = row["passages"]
            texts = passages["passage_text"]
            urls  = passages.get("url", [""] * len(texts))
            for text, url in zip(texts, urls):
                orig_id = _passage_id(url, text)
                if orig_id in seen_ids:
                    continue
                seen_ids.add(orig_id)
                yield Document(
                    doc_id=f"{DOMAIN}:{orig_id}",
                    title="",
                    text=text,
                    domain=DOMAIN,
                    orig_id=orig_id,
                )
                count += 1
                if self.max_docs and count >= self.max_docs:
                    logger.info(
                        f"[MSMarco] Reached max_docs={self.max_docs}, stopping."
                    )
                    return

    def iter_queries(self, split: str = "validation") -> Iterator[Query]:
        for row in self._load(split):
            yield Query(
                query_id=f"{DOMAIN}:{row['query_id']}",
                text=row["query"],
                domain=DOMAIN,
            )

    def iter_qrels(self, split: str = "validation") -> Iterator[QRel]:
        for row in self._load(split):
            passages = row["passages"]
            texts = passages["passage_text"]
            urls  = passages.get("url", [""] * len(texts))
            for text, url, is_selected in zip(
                texts, urls, passages["is_selected"]
            ):
                if int(is_selected) > 0:
                    orig_id = _passage_id(url, text)
                    yield QRel(
                        query_id=f"{DOMAIN}:{row['query_id']}",
                        doc_id=f"{DOMAIN}:{orig_id}",
                        relevance=int(is_selected),
                    )
