"""
Domain classifier inference wrapper.

Loads a fine-tuned DistilBERT/BERT model from disk and classifies
query strings into one of the 6 retrieval domains.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    top1_domain: str
    top2_domain: str
    top1_prob: float
    top2_prob: float
    all_probs: dict[str, float] = field(default_factory=dict)


class DomainClassifier:
    """
    Inference wrapper for the fine-tuned domain classifier.

    Args:
        model_path: Path to directory containing pytorch_model.bin,
                    config.json, tokenizer files, and label_map.json.
        device:     "cpu" or "cuda".
        max_length: Max tokenization length for queries (should match training).
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        max_length: int = 64,
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.max_length = max_length

        self._model: Optional[AutoModelForSequenceClassification] = None
        self._tokenizer = None
        self._label_map: dict[int, str] = {}

    def load(self) -> None:
        with open(self.model_path / "label_map.json") as f:
            raw = json.load(f)
        # Keys may be strings (from JSON) or ints
        self._label_map = {int(k): v for k, v in raw.items()}

        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path)
        ).to(self.device)
        self._model.eval()
        logger.info(
            f"DomainClassifier loaded from {self.model_path} "
            f"({len(self._label_map)} classes)"
        )

    def classify(self, query: str) -> ClassificationResult:
        inputs = self._tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze().cpu().float().numpy()
        sorted_idx = probs.argsort()[::-1]
        all_probs = {self._label_map[i]: float(probs[i]) for i in range(len(probs))}
        return ClassificationResult(
            top1_domain=self._label_map[int(sorted_idx[0])],
            top2_domain=self._label_map[int(sorted_idx[1])],
            top1_prob=float(probs[sorted_idx[0]]),
            top2_prob=float(probs[sorted_idx[1]]),
            all_probs=all_probs,
        )

    def classify_batch(self, queries: list[str]) -> list[ClassificationResult]:
        inputs = self._tokenizer(
            queries,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs_batch = F.softmax(logits, dim=-1).cpu().float().numpy()

        results = []
        for probs in probs_batch:
            sorted_idx = probs.argsort()[::-1]
            all_probs = {self._label_map[i]: float(probs[i]) for i in range(len(probs))}
            results.append(
                ClassificationResult(
                    top1_domain=self._label_map[int(sorted_idx[0])],
                    top2_domain=self._label_map[int(sorted_idx[1])],
                    top1_prob=float(probs[sorted_idx[0]]),
                    top2_prob=float(probs[sorted_idx[1]]),
                    all_probs=all_probs,
                )
            )
        return results
