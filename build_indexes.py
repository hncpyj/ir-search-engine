"""
build_indexes.py — 도메인별 FAISS 인덱스 빌드 (재시작 안전)

작은 도메인부터 순서대로 빌드. 이미 faiss.index가 있으면 스킵.
완료마다 즉시 저장 → 중간에 꺼도 완료된 것은 보존.

Usage:
    python build_indexes.py          # 전체
    python build_indexes.py science  # 특정 도메인만
"""
import os, sys, json, logging
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"   # prevent HNSW deadlock on macOS

ROOT = Path(__file__).parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("build_indexes")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import faiss
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from src.config import load_config

# 작은 도메인부터 → 빠르게 완료되는 것 먼저
DOMAIN_ORDER = ["science", "biomedical", "legal", "finance", "medical", "general"]

ENCODER_MAP = {
    "general":    "sentence-transformers/msmarco-bert-base-dot-v5",
    "science":    "allenai/scibert_scivocab_uncased",
    "finance":    "ProsusAI/finbert",
    "medical":    "emilyalsentzer/Bio_ClinicalBERT",
    "legal":      "nlpaueb/legal-bert-base-uncased",
    "biomedical": "dmis-lab/biobert-base-cased-v1.1",
}

FAISS_TYPE = {
    "general":    "HNSW",
    "science":    "Flat",
    "finance":    "HNSW",
    "medical":    "HNSW",
    "legal":      "HNSW",
    "biomedical": "Flat",
}


def mean_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    mask = attn_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def encode_texts(texts, model_name, batch_size=32, max_length=256):
    logger.info(f"  Loading encoder: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    all_embs = []
    total = (len(texts) + batch_size - 1) // batch_size
    for i, start in enumerate(tqdm(range(0, len(texts), batch_size), desc="  Encoding")):
        batch = texts[start:start+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
        emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
        emb = F.normalize(emb, p=2, dim=-1)
        all_embs.append(emb.float().numpy())

    return np.vstack(all_embs).astype(np.float32)


def build_faiss_index(embeddings, faiss_type, m=32, nlist=128,
                      ef_construction=200, ef_search=128):
    D = embeddings.shape[1]
    if faiss_type == "Flat":
        index = faiss.IndexFlatIP(D)
    elif faiss_type == "HNSW":
        index = faiss.IndexHNSWFlat(D, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
    else:
        raise ValueError(f"Unknown faiss_type: {faiss_type}")
    index.add(embeddings)
    return index


def build_domain(domain, data_root, index_root, force=False, max_chunks=None):
    corpus_path = data_root / domain / "corpus.parquet"
    out_dir = index_root / "faiss" / domain
    index_path = out_dir / "faiss.index"

    if index_path.exists() and not force:
        logger.info(f"[{domain}] ✅ faiss.index already exists — SKIP")
        return True

    if not corpus_path.exists():
        logger.error(f"[{domain}] ❌ corpus.parquet not found — run build_corpus.py first")
        return False

    logger.info(f"\n{'='*55}")
    logger.info(f"[{domain}] Building FAISS index ...")

    df = pd.read_parquet(corpus_path)
    if max_chunks and len(df) > max_chunks:
        df = df.iloc[:max_chunks].reset_index(drop=True)
        logger.info(f"[{domain}] Corpus capped to {max_chunks} chunks (was {len(df)} before cap)")
    logger.info(f"[{domain}] Corpus: {len(df)} chunks")

    model_name = ENCODER_MAP[domain]
    embeddings = encode_texts(df["text"].tolist(), model_name)
    logger.info(f"[{domain}] Embeddings shape: {embeddings.shape}")

    ft = FAISS_TYPE[domain]
    logger.info(f"[{domain}] Building {ft} index ...")
    index = build_faiss_index(embeddings, ft)
    logger.info(f"[{domain}] Index ntotal={index.ntotal}")

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    df.reset_index(drop=True).to_parquet(out_dir / "docstore.parquet", index=True)
    id_mapping = {str(i): str(cid) for i, cid in enumerate(df["chunk_id"])}
    with open(out_dir / "id_mapping.json", "w") as f:
        json.dump(id_mapping, f)

    logger.info(f"[{domain}] ✅ Saved to {out_dir}")
    return True


def main():
    cfg = load_config("configs/default.yaml")
    data_root  = Path(cfg["paths"]["data_root"])
    index_root = Path(cfg["paths"]["index_root"])

    # 커맨드라인으로 특정 도메인만 지정 가능
    if len(sys.argv) > 1:
        domains = sys.argv[1:]
    else:
        domains = DOMAIN_ORDER

    # Per-domain chunk cap: derived from max_docs in config
    # medical: 171k docs → 377k chunks; 50k docs cap → ~110k chunks
    # general: 50k docs cap → ~150k chunks
    MAX_CHUNKS = {}
    for domain, dcfg in cfg["domains"].items():
        md = dcfg.get("max_docs")
        if md:
            # Estimate chunks per doc from actual ratio if known, else 3x
            RATIO = {"medical": 2.2, "general": 3.0}
            ratio = RATIO.get(domain, 3.0)
            MAX_CHUNKS[domain] = int(md * ratio)

    logger.info(f"Domains to build: {domains}")
    results = {}
    for domain in domains:
        ok = build_domain(domain, data_root, index_root,
                          max_chunks=MAX_CHUNKS.get(domain))
        results[domain] = "✅ OK" if ok else "❌ FAIL"

    logger.info("\n" + "="*55)
    logger.info("Summary:")
    for d, r in results.items():
        logger.info(f"  {d:12s} {r}")


if __name__ == "__main__":
    main()
