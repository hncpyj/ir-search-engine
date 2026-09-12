"""
Generation layer — base interface.

Sits on top of the retrieval pipeline: takes a query and a list of
RetrievalResult passages, returns an answer with inline citations.

The interface is deliberately thin so backends (Ollama, OpenAI, …) are
swappable. Config-driven; no framework dependency.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.retrieval.dense_retriever import RetrievalResult


@dataclass
class GeneratorResult:
    answer: str
    cited_ids: list[str]      # passage chunk_ids cited as [P1], [P2], …
    model: str
    seed: int
    latency_ms: float
    abstained: bool = False   # True when model said it cannot answer


_CITATION_RE = re.compile(r"\[P(\d+)\]")


def extract_cited_ids(answer: str, passages: list[RetrievalResult]) -> list[str]:
    """Parse [P1], [P2] citations from the answer and resolve to chunk_ids."""
    cited: list[str] = []
    for m in _CITATION_RE.finditer(answer):
        idx = int(m.group(1)) - 1        # [P1] → index 0
        if 0 <= idx < len(passages):
            chunk_id = passages[idx].doc_id
            if chunk_id not in cited:
                cited.append(chunk_id)
    return cited


_ABSTENTION_PHRASES = (
    "i don't know",
    "i cannot answer",
    "i do not have",
    "no information",
    "cannot be determined",
    "not enough information",
    "the provided passages do not",
    "the passages do not",
    "no relevant",
)


def detect_abstention(answer: str) -> bool:
    low = answer.lower().strip()
    return any(phrase in low for phrase in _ABSTENTION_PHRASES)


class BaseGenerator(ABC):
    """Abstract generator — one backend per subclass."""

    @abstractmethod
    def generate(
        self,
        query: str,
        passages: list[RetrievalResult],
        seed: int = 42,
    ) -> GeneratorResult:
        """Generate an answer from query + passages.

        Args:
            query:    The user query (raw or normalized).
            passages: Retrieved passages to ground the answer in. Empty list
                      is a valid input (models the no_retrieval condition).
            seed:     Sampling seed for reproducibility.

        Returns:
            GeneratorResult with answer, citations, and metadata.
        """


def build_prompt(
    query: str,
    passages: list[RetrievalResult],
    prompt_mode: str = "strict",
) -> str:
    """Build the generation prompt.

    Args:
        query:       The user query.
        passages:    Retrieved passages to include as context.
        prompt_mode: "strict" (default) — answer ONLY from passages;
                     "partial" — passages are primary evidence, supplement
                     with knowledge when passages are insufficient.
                     "strict" is used for grounding evaluation so that
                     abstention truly reflects absence of relevant context.
                     "partial" is useful for science/medical domains where
                     the corpus slice may be sparse.
    """
    if not passages:
        context_block = "(No passages provided.)"
    else:
        parts = []
        for i, p in enumerate(passages, 1):
            title = f" — {p.title}" if p.title else ""
            parts.append(f"[P{i}]{title}\n{p.text.strip()}")
        context_block = "\n\n".join(parts)

    if prompt_mode == "claim_verify":
        instruction = (
            "You are a fact-checker. Using ONLY the provided passages, determine whether "
            "the following claim is SUPPORTED, REFUTED, or NOT ENOUGH INFO. "
            "Cite the relevant passage numbers inline as [P1], [P2], etc. "
            "Give a one-sentence verdict followed by a brief explanation."
        )
    elif prompt_mode == "partial":
        instruction = (
            "Answer the question using the provided passages as primary evidence. "
            "Cite passage numbers inline as [P1], [P2], etc. "
            "If the passages are insufficient, supplement with your knowledge "
            "but clearly indicate which parts come from the passages and which from general knowledge."
        )
    else:  # strict
        instruction = (
            "Answer the question using ONLY the provided passages below. "
            "Cite passage numbers inline as [P1], [P2], etc. "
            "If the passages do not contain enough information, say so explicitly."
        )

    return (
        f"{instruction}\n\n"
        f"Passages:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
