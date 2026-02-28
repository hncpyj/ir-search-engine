"""
train_domain_classifier.py — Train the 6-class domain classifier.

Reads query texts from data/processed/{domain}/queries.parquet,
augments small domains with seed queries, fine-tunes DistilBERT,
and saves the model to models/domain_classifier/.

Usage:
    python scripts/train_domain_classifier.py --config configs/default.yaml
    python scripts/train_domain_classifier.py --config configs/default.yaml --epochs 5
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.classification.train_classifier import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train domain classifier.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    clf_cfg = cfg["classification"]
    data_root = cfg["paths"]["data_root"]
    output_dir = str(Path(cfg["paths"]["model_root"]) / "domain_classifier")

    train(
        data_root=data_root,
        output_dir=output_dir,
        base_model=clf_cfg.get("base_model", "distilbert-base-uncased"),
        num_epochs=args.epochs or 3,
        batch_size=args.batch_size or 32,
        lr=2e-5,
        max_length=clf_cfg.get("max_length", 64),
        min_per_domain=clf_cfg.get("min_queries_per_domain", 200),
        max_per_domain=clf_cfg.get("max_queries_per_domain", 5000),
        seed=cfg["system"].get("seed", 42),
    )


if __name__ == "__main__":
    main()
