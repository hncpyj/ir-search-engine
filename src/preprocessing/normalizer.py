"""
Query normalizer: deterministic preprocessing for all query strings.

Applies unicode normalization, lowercasing, control-character removal,
and whitespace collapsing. Does NOT stem or remove stopwords — the BERT
encoders handle semantic understanding.
"""
from __future__ import annotations

import re
import unicodedata


class QueryNormalizer:
    """Lightweight, deterministic text normalization for queries."""

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        # Unicode normalization (handles accents, full-width chars, etc.)
        text = unicodedata.normalize("NFKC", text)
        # Lowercase
        text = text.lower()
        # Remove control characters (except tab/newline which become spaces)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
        # Collapse all whitespace (tab, newline, multiple spaces)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_batch(self, texts: list[str]) -> list[str]:
        return [self.normalize(t) for t in texts]
