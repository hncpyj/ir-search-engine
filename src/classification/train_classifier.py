"""
Domain classifier training.

Loads query texts from each domain's queries.parquet, optionally augments
small domains with seed queries, balances classes, then fine-tunes
distilbert-base-uncased as a 6-class sequence classifier.

Saves the trained model + tokenizer + label_map.json to models/domain_classifier/.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .seed_queries import SEEDS

logger = logging.getLogger(__name__)

DOMAIN_LABELS = ["general", "science", "scidocs", "finance", "medical", "biomedical"]
LABEL2ID = {d: i for i, d in enumerate(DOMAIN_LABELS)}
ID2LABEL = {i: d for d, i in LABEL2ID.items()}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class QueryDataset(Dataset):
    def __init__(self, encodings, labels: list[int]):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_domain_queries(
    data_root: Path,
    min_per_domain: int = 200,
    max_per_domain: int = 5000,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """
    Load query texts from queries.parquet for each domain.
    - Augments with seed queries if domain has < min_per_domain.
    - Caps at max_per_domain to prevent class imbalance.
    - BUG-3 fix: excludes qrel-annotated queries (BEIR evaluation queries) from
      training data to prevent train/eval leakage in routing evaluation.
      evaluate_routing.py samples from qrel-annotated queries only, so excluding
      them here ensures the routing eval is on genuinely held-out data.
    Returns (texts, int_labels).
    """
    all_texts: list[str] = []
    all_labels: list[int] = []
    rng = random.Random(seed)

    for domain in DOMAIN_LABELS:
        label = LABEL2ID[domain]
        qpath  = data_root / domain / "queries.parquet"
        qrpath = data_root / domain / "qrels.parquet"

        texts: list[str] = []
        if qpath.exists():
            df = pd.read_parquet(qpath)
            if "text" in df.columns:
                # Exclude qrel-annotated (evaluation) queries to prevent leakage
                if qrpath.exists():
                    qrels_df = pd.read_parquet(qrpath)
                    eval_qids = set(qrels_df["query_id"].astype(str).unique())
                    n_before = len(df)
                    df = df[~df["query_id"].astype(str).isin(eval_qids)]
                    logger.info(
                        f"[classifier] {domain}: excluded {n_before - len(df)} "
                        f"eval queries (BUG-3 fix), {len(df)} training queries remain"
                    )
                texts = df["text"].dropna().tolist()
            logger.info(f"[classifier] {domain}: loaded {len(texts)} queries from parquet")
        else:
            logger.warning(f"[classifier] {domain}: queries.parquet not found at {qpath}")

        # Augment with seeds if below minimum
        seeds = SEEDS.get(domain, [])
        if len(texts) < min_per_domain and seeds:
            # Repeat seeds until we have enough
            extra = []
            while len(texts) + len(extra) < min_per_domain:
                extra.extend(seeds)
            texts = texts + extra[:max(0, min_per_domain - len(texts))]
            logger.info(
                f"[classifier] {domain}: augmented to {len(texts)} with seed queries"
            )

        # Cap to max_per_domain
        if len(texts) > max_per_domain:
            rng.shuffle(texts)
            texts = texts[:max_per_domain]
            logger.info(f"[classifier] {domain}: capped to {max_per_domain}")

        all_texts.extend(texts)
        all_labels.extend([label] * len(texts))

    logger.info(
        f"[classifier] Total: {len(all_texts)} queries across {len(DOMAIN_LABELS)} domains"
    )
    return all_texts, all_labels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    top1_acc = float((preds == labels).mean())

    # Top-2 accuracy
    top2 = np.argsort(logits, axis=-1)[:, -2:]
    top2_acc = float(
        sum(labels[i] in top2[i] for i in range(len(labels))) / len(labels)
    )
    return {"top1_accuracy": top1_acc, "top2_accuracy": top2_acc}


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(
    data_root: str,
    output_dir: str,
    base_model: str = "distilbert-base-uncased",
    num_epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_length: int = 64,
    min_per_domain: int = 200,
    max_per_domain: int = 5000,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    texts, labels = load_domain_queries(
        Path(data_root), min_per_domain, max_per_domain, seed
    )

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=seed
    )
    logger.info(
        f"[classifier] Train: {len(train_texts)}, Val: {len(val_texts)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(DOMAIN_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def _encode(t: list[str]) -> dict:
        return tokenizer(
            t,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

    train_dataset = QueryDataset(_encode(train_texts), train_labels)
    val_dataset = QueryDataset(_encode(val_texts), val_labels)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=128,
        learning_rate=lr,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="top1_accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("[classifier] Starting training ...")
    trainer.train()

    # Save model + tokenizer + label map
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(Path(output_dir) / "label_map.json", "w") as f:
        json.dump(ID2LABEL, f, indent=2)

    # Final eval
    results = trainer.evaluate()
    logger.info(f"[classifier] Final val metrics: {results}")
    print(
        f"\nDomain classifier trained.\n"
        f"  Top-1 accuracy: {results.get('eval_top1_accuracy', 0):.4f}\n"
        f"  Top-2 accuracy: {results.get('eval_top2_accuracy', 0):.4f}\n"
        f"  Saved to: {output_dir}"
    )
