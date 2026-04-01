"""
FAISS index builder — Offline Phase Step 2.

For a given domain:
  1. Load chunked corpus.parquet.
  2. Encode all chunks with the domain's bi-encoder (fp16 on GPU if available).
  3. Build a FAISS index (Flat | HNSW | IVF_PQ based on config).
  4. Save: faiss.index, docstore.parquet, id_mapping.json.

NOTE: We use raw transformers (AutoTokenizer + AutoModel) instead of
SentenceTransformer.encode() to avoid the macOS multiprocessing/semaphore
crash that sentence-transformers 5.x triggers when spawning DataLoader
workers on Python 3.9.

NOTE 2: FAISS index building runs in a FRESH subprocess (via _faiss_index_worker.py)
that has never imported torch or transformers.  This completely sidesteps the
macOS Apple Silicon segfault where PyTorch's Accelerate BLAS and FAISS's
Accelerate BLAS conflict at index.add() time (Python 3.12+, ARM64).
Embeddings are passed via a temporary .npy file on disk.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pooling over token dimension."""
    mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_emb = (token_embeddings * mask_expanded).sum(dim=1)
    count   = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_emb / count


def encode_texts(
    texts: list[str],
    model_name: str,
    device: str,
    batch_size: int,
    fp16: bool,
    max_length: int = 512,
) -> np.ndarray:
    """
    Encode a list of texts with mean-pooled CLS representation.
    Returns float32 L2-normalised array (N, D).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    if fp16 and device == "cuda":
        model = model.half()

    all_embs: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Batches"):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        emb = _mean_pool(out.last_hidden_state, enc["attention_mask"])
        emb = F.normalize(emb, p=2, dim=-1)
        all_embs.append(emb.cpu().float().numpy())

    return np.vstack(all_embs).astype(np.float32)


# ── main class ────────────────────────────────────────────────────────────────

class FAISSIndexBuilder:
    """
    Encodes a chunked corpus and builds a FAISS index.

    Supports three index types (configured via faiss_type):
      - "Flat"    — exact IP search; best for tiny corpora (<10k).
      - "HNSW"   — approximate, no training; good up to ~1M vectors.
      - "IVF_PQ" — approximate, requires training; good for >500k vectors.

    All embeddings are L2-normalised so that inner product == cosine similarity.
    """

    def __init__(self, domain: str, cfg: dict):
        self.domain = domain
        dcfg = cfg["domains"][domain]
        scfg = cfg["system"]

        self.encoder_model: str = dcfg["encoder_model"]
        self.batch_size: int    = dcfg.get("batch_size", 32)
        self.faiss_type: str    = dcfg.get("faiss_type", "HNSW")
        self.faiss_m: int       = dcfg.get("faiss_m", 32)
        self.faiss_nlist: int   = dcfg.get("faiss_nlist", 128)
        self.ef_construction: int = dcfg.get("faiss_ef_construction", 200)
        self.ef_search: int       = dcfg.get("faiss_ef_search", 128)

        # Auto-detect best device for PyTorch inference.
        # MPS (Apple Silicon) is preferred over CPU; CUDA over MPS.
        # Note: FAISS does not support MPS — index operations always run on CPU.
        configured = scfg.get("device", "cpu")
        if configured == "cpu":
            import torch
            if torch.cuda.is_available():
                configured = "cuda"
            elif torch.backends.mps.is_available():
                configured = "mps"
        self.device: str = configured
        self.fp16: bool  = scfg.get("fp16", False) and self.device == "cuda"

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_corpus(self, df: pd.DataFrame) -> np.ndarray:
        """
        Encode df["text"] in batches via raw transformers (no ST workers).
        Returns float32 array (N, D) with L2-normalised vectors.
        """
        texts = df["text"].tolist()
        logger.info(
            f"[{self.domain}] Encoding {len(texts)} chunks "
            f"with {self.encoder_model} on {self.device} ..."
        )
        return encode_texts(
            texts,
            model_name=self.encoder_model,
            device=self.device,
            batch_size=self.batch_size,
            fp16=self.fp16,
        )

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, embeddings: np.ndarray):
        import faiss  # lazy import — must come AFTER all torch model inference
        D  = embeddings.shape[1]
        ft = self.faiss_type

        if ft == "Flat":
            index = faiss.IndexFlatIP(D)

        elif ft == "HNSW":
            index = faiss.IndexHNSWFlat(D, self.faiss_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch       = self.ef_search

        elif ft == "IVF_PQ":
            m_pq      = min(self.faiss_m, D // 8)
            quantizer = faiss.IndexFlatIP(D)
            index     = faiss.IndexIVFPQ(
                quantizer, D, self.faiss_nlist, m_pq, 8,
                faiss.METRIC_INNER_PRODUCT,
            )
            logger.info(f"[{self.domain}] Training IVF_PQ index ...")
            n_train    = max(self.faiss_nlist * 39, len(embeddings))
            train_vecs = embeddings[:n_train]
            index.train(train_vecs)

        else:
            raise ValueError(f"Unknown faiss_type: {ft!r}")

        # Optional GPU acceleration for building (HNSW不支援 GPU)
        if self.device == "cuda" and ft != "HNSW" and faiss.get_num_gpus() > 0:
            res   = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        logger.info(f"[{self.domain}] Adding {len(embeddings)} vectors ...")
        index.add(embeddings)

        # Move back to CPU before saving
        if self.device == "cuda" and faiss.get_num_gpus() > 0:
            try:
                index = faiss.index_gpu_to_cpu(index)
            except Exception:
                pass

        return index

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        index,
        corpus_df: pd.DataFrame,
        output_dir: Path,
        embeddings: np.ndarray = None,
    ) -> None:
        import faiss  # lazy import — must come AFTER all torch model inference
        output_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(output_dir / "faiss.index"))
        logger.info(f"[{self.domain}] Saved faiss.index ({index.ntotal} vectors)")

        corpus_df.reset_index(drop=True).to_parquet(
            output_dir / "docstore.parquet", index=True
        )

        id_mapping: dict[str, str] = {
            str(i): str(cid)
            for i, cid in enumerate(corpus_df["chunk_id"])
        }
        with open(output_dir / "id_mapping.json", "w") as f:
            json.dump(id_mapping, f)

        metadata = {
            "encoder_model": self.encoder_model,
            "dimension": int(embeddings.shape[1]) if embeddings is not None else index.d,
            "num_vectors": int(index.ntotal),
            "faiss_type": self.faiss_type,
            "normalized": True,
        }
        with open(output_dir / "index_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"[{self.domain}] Index artifacts saved to {output_dir}")

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def build(self, corpus_parquet: Path, output_dir: Path) -> None:
        """Full pipeline: load → encode → (subprocess) index → save."""
        df         = pd.read_parquet(corpus_parquet)
        embeddings = self.encode_corpus(df)

        # ── Flush PyTorch / GPU state before handing off to subprocess ──────
        import gc
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Write embeddings to a temp file, run FAISS in a clean subprocess ─
        # On macOS Apple Silicon (Python 3.12+) both PyTorch-CPU and FAISS use
        # Apple Accelerate/BLAS.  Loading both in the same process causes a
        # silent crash at index.add().  Spawning a fresh process that imports
        # ONLY numpy + faiss (never torch) completely avoids the conflict.
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            np.save(str(tmp_path), embeddings)
            logger.info(
                f"[{self.domain}] Embeddings saved to {tmp_path} "
                f"({embeddings.nbytes / 1e6:.1f} MB) — spawning FAISS worker ..."
            )

            worker = Path(__file__).parent / "_faiss_index_worker.py"
            worker_args = json.dumps({
                "emb_path":       str(tmp_path),
                "corpus_path":    str(corpus_parquet),
                "output_dir":     str(output_dir),
                "faiss_type":     self.faiss_type,
                "faiss_m":        self.faiss_m,
                "faiss_nlist":    self.faiss_nlist,
                "ef_construction": self.ef_construction,
                "ef_search":      self.ef_search,
                "encoder_model":  self.encoder_model,
                "domain":         self.domain,
            })

            result = subprocess.run(
                [sys.executable, str(worker), worker_args],
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"[{self.domain}] FAISS worker exited with code {result.returncode}"
                )
            logger.info(f"[{self.domain}] FAISS worker completed successfully.")

        finally:
            if tmp_path.exists():
                tmp_path.unlink()
                logger.debug(f"[{self.domain}] Temp embeddings file removed.")
