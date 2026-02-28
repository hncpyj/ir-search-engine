"""
Token-aware sliding window chunker.

Documents longer than max_tokens are split into overlapping chunks.
Short documents yield exactly one chunk (no splitting needed).
Chunk IDs follow the pattern: "{doc_id}__chunk{i}".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from transformers import AutoTokenizer


@dataclass
class Chunk:
    chunk_id: str        # "{doc_id}__chunk{i}"
    doc_id: str
    text: str            # decoded text for this window
    start_token: int
    end_token: int


class SlidingWindowChunker:
    """
    Splits document text into overlapping token windows.
    Tokenises → windows → decode back to strings.

    Args:
        tokenizer_name: HuggingFace tokenizer to use for token counting.
        max_tokens: maximum tokens per chunk (default 180).
        stride: overlap between adjacent chunks in tokens (default 40).
    """

    def __init__(
        self,
        tokenizer_name: str,
        max_tokens: int = 180,
        stride: int = 40,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, use_fast=True
        )
        self.max_tokens = max_tokens
        self.stride = stride
        self._step = max_tokens - stride

    def chunk_text(self, doc_id: str, text: str) -> Iterator[Chunk]:
        """Yield Chunk objects for a single document text."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= self.max_tokens:
            yield Chunk(
                chunk_id=f"{doc_id}__chunk0",
                doc_id=doc_id,
                text=text,
                start_token=0,
                end_token=len(tokens),
            )
            return

        start = 0
        i = 0
        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(
                chunk_tokens, skip_special_tokens=True
            )
            yield Chunk(
                chunk_id=f"{doc_id}__chunk{i}",
                doc_id=doc_id,
                text=chunk_text,
                start_token=start,
                end_token=end,
            )
            if end == len(tokens):
                break
            start += self._step
            i += 1

    def chunk_document(self, doc: dict) -> Iterator[dict]:
        """
        Chunk a single document dict.
        Input dict keys: doc_id, title, text, domain, orig_id.
        Yields dicts with added: chunk_id, start_token, end_token.
        Title is prepended to text before chunking.
        """
        title = (doc.get("title") or "").strip()
        body = (doc.get("text") or "").strip()
        full_text = f"{title} {body}".strip() if title else body

        for chunk in self.chunk_text(doc["doc_id"], full_text):
            yield {
                "chunk_id": chunk.chunk_id,
                "doc_id": doc["doc_id"],
                "orig_id": doc["orig_id"],
                "domain": doc["domain"],
                "title": title,
                "text": chunk.text,
                "start_token": chunk.start_token,
                "end_token": chunk.end_token,
            }

    def chunk_documents(self, documents: list[dict]) -> Iterator[dict]:
        """Chunk a list of document dicts."""
        for doc in documents:
            yield from self.chunk_document(doc)
