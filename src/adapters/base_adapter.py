"""
Base adapter: shared data model and abstract interface for all dataset adapters.
All doc_ids and query_ids are namespaced: "{domain}:{original_id}".
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional

import pandas as pd


@dataclass
class Document:
    doc_id: str      # "{domain}:{orig_id}"
    title: str
    text: str
    domain: str
    orig_id: str


@dataclass
class Query:
    query_id: str    # "{domain}:{orig_qid}"
    text: str
    domain: str


@dataclass
class QRel:
    query_id: str    # "{domain}:{orig_qid}"
    doc_id: str      # "{domain}:{orig_doc_id}"
    relevance: int


class BaseAdapter(ABC):
    """
    Abstract base for dataset adapters.
    Subclasses implement iter_documents, iter_queries, iter_qrels.
    """

    def __init__(self, domain: str, config: dict):
        self.domain = domain
        self.config = config
        self.max_docs: Optional[int] = config.get("max_docs", None)

    @abstractmethod
    def iter_documents(self) -> Iterator[Document]:
        """Yield Document objects from the corpus."""
        ...

    @abstractmethod
    def iter_queries(self, split: str = "test") -> Iterator[Query]:
        """Yield Query objects. split: 'train'|'dev'|'validation'|'test'."""
        ...

    @abstractmethod
    def iter_qrels(self, split: str = "test") -> Iterator[QRel]:
        """Yield QRel objects."""
        ...

    def to_dataframes(
        self, split: str = "test"
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Returns (corpus_df, queries_df, qrels_df).
          corpus_df  — columns: doc_id, title, text, domain, orig_id
          queries_df — columns: query_id, text, domain
          qrels_df   — columns: query_id, doc_id, relevance
        """
        docs = pd.DataFrame(
            [{"doc_id": d.doc_id, "title": d.title, "text": d.text,
              "domain": d.domain, "orig_id": d.orig_id}
             for d in self.iter_documents()]
        )
        queries = pd.DataFrame(
            [{"query_id": q.query_id, "text": q.text, "domain": q.domain}
             for q in self.iter_queries(split)]
        )
        qrels = pd.DataFrame(
            [{"query_id": r.query_id, "doc_id": r.doc_id, "relevance": r.relevance}
             for r in self.iter_qrels(split)]
        )
        return docs, queries, qrels
