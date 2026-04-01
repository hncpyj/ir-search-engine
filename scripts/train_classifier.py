"""
train_classifier.py — Train the domain routing classifier.

Fine-tunes DistilBERT as a 6-class domain classifier using query texts from
each domain's queries.parquet. Saves the trained model to models/domain_classifier/.

Usage:
    python scripts/train_classifier.py
    python scripts/train_classifier.py --config configs/default.yaml
    python scripts/train_classifier.py --epochs 5 --max-per-domain 3000
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
    parser = argparse.ArgumentParser(description="Train domain routing classifier.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-per-domain", type=int, default=None,
                        help="Override max queries per domain (default: from config)")
    parser.add_argument("--min-per-domain", type=int, default=None,
                        help="Override min queries per domain (default: from config)")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    clf_cfg = cfg["classification"]
    paths   = cfg["paths"]

    train(
        data_root=paths["data_root"],
        output_dir=f"{paths['model_root']}/domain_classifier",
        base_model=clf_cfg["base_model"],
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=clf_cfg["max_length"],
        min_per_domain=args.min_per_domain or clf_cfg["min_queries_per_domain"],
        max_per_domain=args.max_per_domain or clf_cfg["max_queries_per_domain"],
    )


if __name__ == "__main__":
    main()
