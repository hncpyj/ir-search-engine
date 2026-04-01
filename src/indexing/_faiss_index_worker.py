"""
FAISS index worker — subprocess entry point.

Runs in a FRESH Python process that has never imported torch or transformers.
This sidesteps the macOS Apple Silicon segfault where PyTorch's Accelerate BLAS
framework conflicts with FAISS's own Accelerate usage at index.add() time.

Called by FAISSIndexBuilder.build() via subprocess.run():
    python _faiss_index_worker.py '<json_args>'

JSON args (single CLI argument, JSON-encoded dict):
    emb_path        : str  — path to float32 .npy embeddings file
    corpus_path     : str  — path to corpus.parquet (for docstore + id_mapping)
    output_dir      : str  — directory to write index artifacts
    faiss_type      : str  — "Flat" | "HNSW" | "IVF_PQ"
    faiss_m         : int  — HNSW M / IVF_PQ sub-quantizers
    faiss_nlist     : int  — IVF nlist
    ef_construction : int  — HNSW efConstruction
    ef_search       : int  — HNSW efSearch
    encoder_model   : str  — stored in index_metadata.json
    domain          : str  — used for log prefixes
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd


def _build_index(embeddings: np.ndarray, args: dict):
    import faiss  # only import here — this process has NO torch

    D          = embeddings.shape[1]
    faiss_type = args["faiss_type"]
    faiss_m    = args["faiss_m"]

    if faiss_type == "Flat":
        index = faiss.IndexFlatIP(D)

    elif faiss_type == "HNSW":
        index = faiss.IndexHNSWFlat(D, faiss_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = args["ef_construction"]
        index.hnsw.efSearch       = args["ef_search"]

    elif faiss_type == "IVF_PQ":
        m_pq      = min(faiss_m, D // 8)
        quantizer = faiss.IndexFlatIP(D)
        index     = faiss.IndexIVFPQ(
            quantizer, D, args["faiss_nlist"], m_pq, 8,
            faiss.METRIC_INNER_PRODUCT,
        )
        n_train    = max(args["faiss_nlist"] * 39, len(embeddings))
        train_vecs = embeddings[:n_train]
        print(f"[{args['domain']}] Training IVF_PQ index ...", flush=True)
        index.train(train_vecs)

    else:
        raise ValueError(f"Unknown faiss_type: {faiss_type!r}")

    print(f"[{args['domain']}] Adding {len(embeddings)} vectors ...", flush=True)
    index.add(embeddings)
    return index, faiss


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: _faiss_index_worker.py '<json_args>'", file=sys.stderr)
        sys.exit(1)

    args       = json.loads(sys.argv[1])
    emb_path   = args["emb_path"]
    output_dir = pathlib.Path(args["output_dir"])
    domain     = args["domain"]

    # ── Load embeddings ────────────────────────────────────────────────────────
    print(f"[{domain}] Worker: loading embeddings from {emb_path} ...", flush=True)
    embeddings = np.load(emb_path).astype(np.float32)

    # ── Build index ────────────────────────────────────────────────────────────
    index, faiss = _build_index(embeddings, args)

    # ── Save artifacts ─────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(output_dir / "faiss.index"))
    print(f"[{domain}] Saved faiss.index ({index.ntotal} vectors)", flush=True)

    # docstore + id_mapping from corpus parquet
    corpus_df = pd.read_parquet(args["corpus_path"]).reset_index(drop=True)
    corpus_df.to_parquet(output_dir / "docstore.parquet", index=True)

    id_mapping: dict[str, str] = {
        str(i): str(cid)
        for i, cid in enumerate(corpus_df["chunk_id"])
    }
    with open(output_dir / "id_mapping.json", "w") as f:
        json.dump(id_mapping, f)

    metadata = {
        "encoder_model": args["encoder_model"],
        "dimension":    int(embeddings.shape[1]),
        "num_vectors":  int(index.ntotal),
        "faiss_type":   args["faiss_type"],
        "normalized":   True,
    }
    with open(output_dir / "index_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[{domain}] All index artifacts saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
