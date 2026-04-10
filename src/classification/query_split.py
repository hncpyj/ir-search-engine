"""
Deterministic train/eval split for qrel-annotated queries.

Replaces the BUG-3 fix's "exclude all qrel queries from training" approach,
which left medical and scidocs with 0 real training queries. Instead, we
deterministically split each domain's qrel queries into a training half and
an evaluation half. Both train_classifier.py and evaluate_routing.py use
this same split, so there is no train/eval leakage.

The split is by hash(query_id) modulo 10:
  - hash % 10 < TRAIN_BUCKETS  -> training set
  - hash % 10 >= TRAIN_BUCKETS -> evaluation set
"""
from __future__ import annotations

import hashlib

TRAIN_BUCKETS = 7  # 70% train / 30% eval


def is_train_query(query_id: str) -> bool:
    """Return True if this query_id belongs to the training split."""
    h = hashlib.md5(str(query_id).encode("utf-8")).hexdigest()
    return int(h, 16) % 10 < TRAIN_BUCKETS


def is_eval_query(query_id: str) -> bool:
    """Return True if this query_id belongs to the evaluation split."""
    return not is_train_query(query_id)
