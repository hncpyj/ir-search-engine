"""
build_corpus.py — Offline Phase Step 1

For each enabled domain:
  1. Run the dataset adapter to load corpus, queries, qrels.
  2. Chunk the corpus with SlidingWindowChunker.
  3. Write chunked corpus, queries, and qrels to data/processed/{domain}/.

Skips domains where corpus.parquet already exists (re-run safe).

Usage:
    python scripts/build_corpus.py --config configs/default.yaml [--domain science]
    python scripts/build_corpus.py --config configs/default.yaml --force
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.msmarco_adapter import MSMarcoAdapter
from src.adapters.scifact_adapter import ScifactAdapter
from src.adapters.fiqa_adapter import FiqaAdapter
from src.adapters.treccovid_adapter import TrecCovidAdapter
from src.adapters.nfcorpus_adapter import NfcorpusAdapter
from src.adapters.echr_adapter import EchrAdapter
from src.preprocessing.chunker import SlidingWindowChunker
from src.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("build_corpus")

ADAPTER_MAP = {
    "general": MSMarcoAdapter,
    "science": ScifactAdapter,
    "finance": FiqaAdapter,
    "medical": TrecCovidAdapter,
    "biomedical": NfcorpusAdapter,
    "legal": EchrAdapter,
}

# Use the domain's own encoder tokenizer for chunking (consistent token counts).
TOKENIZER_MAP = {
    "general": "sentence-transformers/msmarco-bert-base-dot-v5",
    "science": "allenai/scibert_scivocab_uncased",
    "finance": "ProsusAI/finbert",
    "medical": "emilyalsentzer/Bio_ClinicalBERT",
    "legal": "nlpaueb/legal-bert-base-uncased",
    "biomedical": "dmis-lab/biobert-base-cased-v1.1",
}


def build_domain(
    domain: str,
    cfg: dict,
    data_root: Path,
    force: bool = False,
) -> None:
    domain_cfg = cfg["domains"][domain]
    out_dir = data_root / domain
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = out_dir / "corpus.parquet"
    queries_path = out_dir / "queries.parquet"
    qrels_path = out_dir / "qrels.parquet"

    if corpus_path.exists() and not force:
        logger.info(f"[{domain}] corpus.parquet exists — skipping (use --force to rebuild).")
    else:
        logger.info(f"[{domain}] Loading corpus from {domain_cfg['dataset']} ...")
        AdapterClass = ADAPTER_MAP[domain]
        adapter = AdapterClass(domain_cfg)

        # Determine eval split (MS MARCO uses 'validation', BEIR uses 'test')
        eval_split = "validation" if domain == "general" else "test"

        logger.info(f"[{domain}] Loading queries and qrels (split={eval_split}) ...")
        corpus_df, queries_df, qrels_df = adapter.to_dataframes(split=eval_split)

        logger.info(
            f"[{domain}] Corpus: {len(corpus_df)} docs, "
            f"Queries: {len(queries_df)}, QRels: {len(qrels_df)}"
        )

        # Chunk corpus
        chunk_cfg = cfg["chunking"]
        tokenizer_name = TOKENIZER_MAP[domain]
        logger.info(
            f"[{domain}] Chunking with {tokenizer_name} "
            f"(max_tokens={chunk_cfg['max_tokens']}, stride={chunk_cfg['stride']}) ..."
        )
        chunker = SlidingWindowChunker(
            tokenizer_name=tokenizer_name,
            max_tokens=chunk_cfg["max_tokens"],
            stride=chunk_cfg["stride"],
        )
        chunk_rows = list(
            chunker.chunk_documents(corpus_df.to_dict(orient="records"))
        )
        chunked_df = pd.DataFrame(chunk_rows)
        logger.info(
            f"[{domain}] Chunked: {len(corpus_df)} docs → {len(chunked_df)} chunks."
        )

        chunked_df.to_parquet(corpus_path, index=False)
        queries_df.to_parquet(queries_path, index=False)
        qrels_df.to_parquet(qrels_path, index=False)
        logger.info(f"[{domain}] Saved to {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chunked corpus parquets per domain.")
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Base config file (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--override", default=None,
        help="Optional override config (e.g. configs/full.yaml)"
    )
    parser.add_argument(
        "--domain", default=None,
        help="Process only this domain (default: all enabled domains)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild even if output already exists"
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    data_root = Path(cfg["paths"]["data_root"])

    domains_to_build = (
        [args.domain]
        if args.domain
        else [d for d, dcfg in cfg["domains"].items() if dcfg.get("enabled", False)]
    )

    logger.info(f"Building corpus for domains: {domains_to_build}")
    for domain in domains_to_build:
        if domain not in ADAPTER_MAP:
            logger.warning(f"No adapter for domain '{domain}', skipping.")
            continue
        build_domain(domain, cfg, data_root, force=args.force)

    logger.info("Done.")


if __name__ == "__main__":
    main()
