"""
Online serving pipeline — orchestrates all components for a single query.

Flow:
  raw_query
    → QueryNormalizer
    → DomainClassifier          (→ routing decision)
    → DenseRetriever per domain [+ BM25Retriever if enabled]
    → RRFFusion
    → ColBERTReranker           (if enabled)
    → {results, classification, active_domains, latency}
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from ..preprocessing.normalizer import QueryNormalizer
from ..classification.domain_classifier import DomainClassifier, ClassificationResult
from ..retrieval.dense_retriever import DenseRetriever, RetrievalResult
from ..retrieval.bm25_retriever import BM25Retriever
from ..retrieval.hybrid_fusion import RRFFusion

logger = logging.getLogger(__name__)

ALL_DOMAINS = ["general", "scidocs", "science", "finance", "medical", "biomedical"]


class SearchPipeline:
    """
    End-to-end query pipeline.

    Routing modes (routing.mode config key):
      broadcast            — always query all enabled domain indexes.
      routed_with_general  — top-1 domain + general always;
                             add top-2 if confidence < threshold.
      routed_only          — top-1 domain only;
                             add top-2 if confidence < threshold.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.normalizer = QueryNormalizer()
        self.classifier: Optional[DomainClassifier] = None
        self.dense_retrievers: dict[str, DenseRetriever] = {}
        self.bm25_retrievers: dict[str, BM25Retriever] = {}
        self.fuser = RRFFusion(rrf_k=cfg["retrieval"].get("rrf_k", 60))
        self.reranker = None

        self._routing_mode: str = cfg["routing"]["mode"]
        self._confidence_threshold: float = cfg["routing"]["confidence_threshold"]
        self._topk_dense: int = cfg["retrieval"]["topk_dense"]
        self._topk_bm25: int = cfg["retrieval"]["topk_bm25"]
        self._k_fusion: int = cfg["retrieval"]["k_fusion"]

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all models and indexes. Call once at startup."""
        cfg = self.cfg
        paths = cfg["paths"]
        sys_cfg = cfg["system"]

        # Domain classifier
        clf_model_path = str(Path(paths["model_root"]) / "domain_classifier")
        self.classifier = DomainClassifier(
            model_path=clf_model_path,
            device=sys_cfg["device"],
            max_length=cfg["classification"].get("max_length", 64),
        )
        self.classifier.load()

        # Dense retrievers + optional BM25
        for domain, domain_cfg in cfg["domains"].items():
            if not domain_cfg.get("enabled", False):
                continue

            index_dir = Path(paths["index_root"]) / "faiss" / domain
            if not (index_dir / "faiss.index").exists():
                logger.warning(
                    f"[pipeline] FAISS index not found for {domain}: {index_dir}. "
                    "Skipping this domain."
                )
                continue

            retriever = DenseRetriever(
                domain=domain,
                index_dir=index_dir,
                encoder_model=domain_cfg["encoder_model"],
                device=sys_cfg["device"],
                fp16=sys_cfg.get("fp16", False),
            )
            retriever.load()
            self.dense_retrievers[domain] = retriever

            if domain_cfg.get("bm25_enabled", False):
                bm25_dir = Path(paths["index_root"]) / "bm25" / domain
                docstore_path = index_dir / "docstore.parquet"
                if (bm25_dir / "bm25.pkl").exists() and docstore_path.exists():
                    bm25 = BM25Retriever(domain, bm25_dir, docstore_path)
                    bm25.load()
                    self.bm25_retrievers[domain] = bm25
                else:
                    logger.warning(
                        f"[pipeline] BM25 index not found for {domain}, skipping BM25."
                    )

        # Reranker (ColBERT if CUDA + ragatouille available, else cross-encoder)
        if cfg["reranking"].get("enabled", False):
            backend = cfg["reranking"].get("backend", "auto")
            model_name = cfg["reranking"]["model"]
            topk_rerank = cfg["reranking"]["topk_rerank"]

            use_colbert = False
            if backend in ("auto", "colbert"):
                try:
                    import torch  # noqa: F401
                    import ragatouille  # noqa: F401
                    use_colbert = True
                except ImportError:
                    use_colbert = False

            if use_colbert and "colbert" in model_name.lower():
                from ..reranking.colbert_reranker import ColBERTReranker
                self.reranker = ColBERTReranker(
                    model_name=model_name,
                    topk=topk_rerank,
                )
            else:
                from ..reranking.cross_encoder_reranker import CrossEncoderReranker
                ce_model = cfg["reranking"].get(
                    "cross_encoder_model",
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                )
                self.reranker = CrossEncoderReranker(
                    model_name=ce_model,
                    topk=min(topk_rerank, 50),  # CPU-friendly cap
                )
            self.reranker.load()

        logger.info(
            f"[pipeline] Loaded {len(self.dense_retrievers)} domain retrievers: "
            f"{list(self.dense_retrievers)}"
        )

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _get_active_domains(self, clf: ClassificationResult) -> list[str]:
        """Select which domain indexes to query based on routing mode."""
        enabled = list(self.dense_retrievers.keys())

        if self._routing_mode == "broadcast":
            return enabled

        elif self._routing_mode == "routed_with_general":
            selected = {clf.top1_domain, "general"}
            if clf.top1_prob < self._confidence_threshold:
                selected.add(clf.top2_domain)
            return [d for d in enabled if d in selected]

        elif self._routing_mode == "routed_only":
            selected = {clf.top1_domain}
            if clf.top1_prob < self._confidence_threshold:
                selected.add(clf.top2_domain)
            return [d for d in enabled if d in selected]

        else:
            raise ValueError(f"Unknown routing mode: {self._routing_mode!r}")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, raw_query: str) -> dict:
        """
        Run the full pipeline for a single query.

        Returns:
            {
              "query": str,
              "normalized_query": str,
              "classification": ClassificationResult,
              "active_domains": list[str],
              "results": list[RetrievalResult],
              "latency": {normalize_ms, classify_ms, retrieve_ms,
                          fuse_ms, rerank_ms, total_ms}
            }
        """
        t0 = time.perf_counter()
        lat: dict[str, float] = {}

        # 1. Normalize
        t = time.perf_counter()
        norm_q = self.normalizer.normalize(raw_query)
        lat["normalize_ms"] = (time.perf_counter() - t) * 1000

        # 2. Classify
        t = time.perf_counter()
        clf = self.classifier.classify(norm_q)
        lat["classify_ms"] = (time.perf_counter() - t) * 1000

        # 3. Route and retrieve
        active_domains = self._get_active_domains(clf)
        t = time.perf_counter()

        # Stage 1: within-domain fusion (BM25 + Dense → one list per domain)
        domain_fused: list[list[RetrievalResult]] = []
        for domain in active_domains:
            domain_lists: list[list[RetrievalResult]] = []
            if domain in self.dense_retrievers and self._topk_dense > 0:
                domain_lists.append(
                    self.dense_retrievers[domain].search(norm_q, self._topk_dense)
                )
            if domain in self.bm25_retrievers:
                domain_lists.append(
                    self.bm25_retrievers[domain].search(norm_q, self._topk_bm25)
                )
            if domain_lists:
                within = (
                    self.fuser.fuse(domain_lists, topk=self._k_fusion)
                    if len(domain_lists) > 1
                    else domain_lists[0]
                )
                domain_fused.append(within)

        lat["retrieve_ms"] = (time.perf_counter() - t) * 1000

        # 4. Stage 2: cross-domain fusion
        t = time.perf_counter()
        fused = self.fuser.fuse(domain_fused, topk=self._k_fusion) if domain_fused else []
        lat["fuse_ms"] = (time.perf_counter() - t) * 1000

        # 5. Rerank
        t = time.perf_counter()
        if self.reranker is not None and fused:
            final = self.reranker.rerank(norm_q, fused)
        else:
            final = fused
        lat["rerank_ms"] = (time.perf_counter() - t) * 1000

        lat["total_ms"] = (time.perf_counter() - t0) * 1000

        return {
            "query": raw_query,
            "normalized_query": norm_q,
            "classification": clf,
            "active_domains": active_domains,
            "results": final,
            "latency": lat,
        }
