"""
Dense retriever — wraps a per-domain FAISS index + docstore.

Encodes queries with the domain's bi-encoder (raw transformers, no ST workers)
and searches FAISS.  Thread-safe after load() (index is read-only).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Single retrieved passage with metadata and scores."""
    chunk_id: str
    doc_id: str
    domain: str
    score: float           # dense score (inner product / cosine)
    text: str
    title: str
    bm25_score: float = 0.0
    fused_score: float = 0.0
    colbert_score: float = 0.0


def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_emb = (token_embeddings * mask_expanded).sum(dim=1)
    count   = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_emb / count


class DenseRetriever:
    """
    Loads a FAISS index, docstore and bi-encoder for one domain.
    Provides single-query and batch search.
    """

    def __init__(
        self,
        domain: str,
        index_dir: Path,
        encoder_model: str,
        device: str = "cpu",
        fp16: bool = False,
    ):
        self.domain = domain
        self.index_dir = Path(index_dir)
        self.encoder_model = encoder_model
        self.device = device
        self.fp16 = fp16

        self._index: Optional[faiss.Index] = None
        self._docstore: Optional[pd.DataFrame] = None
        self._id_mapping: Optional[dict[int, str]] = None
        self._chunk_to_row: Optional[dict[str, int]] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModel] = None

    def load(self) -> None:
        """Load all artifacts. Call once before searching."""
        self._index = faiss.read_index(str(self.index_dir / "faiss.index"))
        self._docstore = pd.read_parquet(self.index_dir / "docstore.parquet")
        with open(self.index_dir / "id_mapping.json") as f:
            raw = json.load(f)
        self._id_mapping = {int(k): v for k, v in raw.items()}
        self._chunk_to_row = {
            str(cid): int(i)
            for i, cid in self._docstore["chunk_id"].items()
        }
        self._tokenizer = AutoTokenizer.from_pretrained(self.encoder_model)
        self._model = AutoModel.from_pretrained(self.encoder_model)
        self._model.to(self.device)
        self._model.eval()
        if self.fp16 and self.device == "cuda":
            self._model = self._model.half()
        logger.info(
            f"[{self.domain}] DenseRetriever loaded: "
            f"{self._index.ntotal} vectors, dim={self._index.d}"
        )

    def _encode(self, texts: list[str], max_length: int = 512) -> np.ndarray:
        """Encode texts → float32 L2-normalised array (N, D)."""
        enc = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self._model(**enc)
        emb = _mean_pool(out.last_hidden_state, enc["attention_mask"])
        emb = F.normalize(emb, p=2, dim=-1)
        return emb.cpu().float().numpy()

    def _lookup(self, idx: int, score: float) -> Optional[RetrievalResult]:
        """Map FAISS internal int ID to a RetrievalResult."""
        if idx == -1:
            return None
        chunk_id = self._id_mapping.get(idx)
        if chunk_id is None:
            return None
        row_idx = self._chunk_to_row.get(chunk_id)
        if row_idx is None:
            return None
        row = self._docstore.iloc[row_idx]
        return RetrievalResult(
            chunk_id=chunk_id,
            doc_id=str(row["doc_id"]),
            domain=self.domain,
            score=float(score),
            text=str(row.get("text", "")),
            title=str(row.get("title", "")),
        )

    def search(self, query: str, topk: int = 1000) -> list[RetrievalResult]:
        """Encode query and return top-k results."""
        vec = self._encode([query]).astype(np.float32)
        scores, indices = self._index.search(vec, topk)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            r = self._lookup(int(idx), float(score))
            if r is not None:
                results.append(r)
        return results

    def search_batch(
        self, queries: list[str], topk: int = 1000, encode_batch_size: int = 32
    ) -> list[list[RetrievalResult]]:
        """Batch search — more efficient for evaluation loops."""
        all_vecs = []
        for start in range(0, len(queries), encode_batch_size):
            batch = queries[start : start + encode_batch_size]
            all_vecs.append(self._encode(batch))
        vecs = np.vstack(all_vecs).astype(np.float32)

        scores_batch, indices_batch = self._index.search(vecs, topk)
        all_results = []
        for scores, indices in zip(scores_batch, indices_batch):
            results = []
            for score, idx in zip(scores, indices):
                r = self._lookup(int(idx), float(score))
                if r is not None:
                    results.append(r)
            all_results.append(results)
        return all_results
