"""
FeB4RAG-inspired cross-domain evaluation — Tier 3.

Following Wang et al. (2024) "FeB4RAG: Evaluating Federated Search in the
Context of Retrieval Augmented Generation" (arXiv:2402.11891), each domain
is treated as an independent retrieval resource.  The three-tier framework:

  Tier 1 — Per-domain BEIR nDCG@10      → scripts/evaluate.py
  Tier 2 — Resource selection accuracy   → scripts/evaluate_routing.py
  Tier 3 — Cross-domain fusion quality   → this module (weak supervision)

For Tier 3 we use an LLM running locally via Ollama as the relevance judge,
replacing manual annotation.  This follows the LLM-as-judge methodology
validated by Fridman et al. (Elastic Research, 2024), who show ~80% agreement
with human assessors, and adopted by the TREC 2024 LLMJudge track.

Two judge backends are available:
  OllamaJudge        — local LLM via Ollama REST API (default, no API cost)
  CrossEncoderJudge  — neural cross-encoder (fallback, no Ollama required)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

QUERIES_PATH = Path(__file__).parent.parent.parent / "data" / "cross_domain" / "queries.jsonl"
OLLAMA_CACHE_PATH = Path(__file__).parent.parent.parent / "data" / "cross_domain" / "ollama_judgements.jsonl"
OPENAI_CACHE_PATH = Path(__file__).parent.parent.parent / "data" / "cross_domain" / "openai_judgements.jsonl"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"
OLLAMA_BASE_URL = "http://localhost:11434"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Query loading
# ---------------------------------------------------------------------------

def load_cross_domain_queries(path: Path = QUERIES_PATH) -> list[dict]:
    """Load queries.jsonl → list of {query_id, text, expected_domains}."""
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


# ---------------------------------------------------------------------------
# Tier 3: Ollama LLM relevance judge (FeB4RAG weak supervision)
# ---------------------------------------------------------------------------

class OllamaJudge:
    """
    Local LLM relevance judge via Ollama REST API.

    Scores (query, document) pairs on a 0-2 scale:
      0 = not relevant
      1 = partially relevant
      2 = highly relevant

    Results are cached to disk so repeated runs skip already-judged pairs.
    Following FeB4RAG (Wang et al., 2024) and TREC 2024 LLMJudge methodology.
    """

    judge_label = "llm"

    def __init__(
        self,
        model: str = OLLAMA_DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        cache_path: Path = OLLAMA_CACHE_PATH,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.cache_path = Path(cache_path)
        self._cache: dict[str, int] = {}

    def load(self) -> None:
        """Load cache from disk and verify Ollama is reachable."""
        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        self._cache[entry["key"]] = entry["score"]
        logger.info(
            f"[OllamaJudge] model={self.model}, "
            f"cache_entries={len(self._cache)}, url={self.base_url}"
        )

    def _cache_key(self, query_id: str, doc_id: str) -> str:
        return f"{query_id}:{doc_id}"

    def _save_entry(self, key: str, score: int) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "score": score}) + "\n")

    def _call_api(self, prompt: str) -> int:
        import requests
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 8},
        }
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        text = resp.json()["message"]["content"].strip()
        for char in text:
            if char in "012":
                return int(char)
        logger.warning(f"[OllamaJudge] Unexpected response '{text}', defaulting to 1")
        return 1

    def score(
        self,
        query: str,
        results: list,
        topk: int = 10,
        query_id: str = "",
    ) -> list[float]:
        """Score top-k results, using cache where available."""
        scores: list[float] = []
        for r in results[:topk]:
            key = self._cache_key(query_id or query[:30], r.doc_id)
            if key in self._cache:
                scores.append(float(self._cache[key]))
                continue

            snippet = f"{r.title or ''} {r.text or ''}"[:400]
            prompt = (
                "Rate the relevance of the following document to the query.\n"
                "Use this scale:\n"
                "  0 = not relevant\n"
                "  1 = partially relevant\n"
                "  2 = highly relevant\n\n"
                f"Query: {query}\n"
                f"Document: {snippet}\n\n"
                "Reply with only a single integer: 0, 1, or 2."
            )
            try:
                s = self._call_api(prompt)
                time.sleep(0.05)  # gentle rate limiting
            except Exception as e:
                logger.warning(f"[OllamaJudge] API error ({key}): {e} — defaulting to 1")
                s = 1

            self._cache[key] = s
            self._save_entry(key, s)
            scores.append(float(s))

        return scores


# ---------------------------------------------------------------------------
# Tier 3: OpenAI LLM relevance judge (API-based)
# ---------------------------------------------------------------------------

class OpenAIJudge:
    """
    LLM relevance judge via OpenAI API.

    Uses GPT-4o-mini (or any OpenAI chat model) to score (query, document)
    pairs on a 0-2 scale, following the same FeB4RAG / TREC 2024 LLMJudge
    methodology as OllamaJudge.

    Results are cached to disk so repeated runs skip already-judged pairs
    and avoid redundant API costs.
    """

    judge_label = "openai"

    def __init__(
        self,
        model: str = OPENAI_DEFAULT_MODEL,
        api_key: str | None = None,
        cache_path: Path = OPENAI_CACHE_PATH,
    ):
        self.model = model
        self.api_key = api_key
        self.cache_path = Path(cache_path)
        self._cache: dict[str, int] = {}
        self._client = None

    def load(self) -> None:
        """Load cache from disk and initialise OpenAI client."""
        import os
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass --openai-key."
            )
        from openai import OpenAI
        self._client = OpenAI(api_key=key)

        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        self._cache[entry["key"]] = entry["score"]
        logger.info(
            f"[OpenAIJudge] model={self.model}, "
            f"cache_entries={len(self._cache)}"
        )

    def _cache_key(self, query_id: str, doc_id: str) -> str:
        return f"openai:{query_id}:{doc_id}"

    def _save_entry(self, key: str, score: int) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "score": score}) + "\n")

    def _call_api(self, prompt: str) -> int:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=8,
        )
        text = resp.choices[0].message.content.strip()
        for char in text:
            if char in "012":
                return int(char)
        logger.warning(f"[OpenAIJudge] Unexpected response '{text}', defaulting to 1")
        return 1

    def score(
        self,
        query: str,
        results: list,
        topk: int = 10,
        query_id: str = "",
    ) -> list[float]:
        """Score top-k results, using cache where available."""
        scores: list[float] = []
        for r in results[:topk]:
            key = self._cache_key(query_id or query[:30], r.doc_id)
            if key in self._cache:
                scores.append(float(self._cache[key]))
                continue

            snippet = f"{r.title or ''} {r.text or ''}"[:400]
            prompt = (
                "Rate the relevance of the following document to the query.\n"
                "Use this scale:\n"
                "  0 = not relevant\n"
                "  1 = partially relevant\n"
                "  2 = highly relevant\n\n"
                f"Query: {query}\n"
                f"Document: {snippet}\n\n"
                "Reply with only a single integer: 0, 1, or 2."
            )
            try:
                s = self._call_api(prompt)
            except Exception as e:
                logger.warning(f"[OpenAIJudge] API error ({key}): {e} — defaulting to 1")
                s = 1

            self._cache[key] = s
            self._save_entry(key, s)
            scores.append(float(s))

        return scores


# ---------------------------------------------------------------------------
# Tier 3: Cross-encoder judge (fallback, no Ollama required)
# ---------------------------------------------------------------------------

class CrossEncoderJudge:
    """
    Cross-encoder relevance judge (fallback when Ollama is unavailable).

    Uses cross-encoder/ms-marco-MiniLM-L-6-v2 to score (query, document) pairs.
    Scores are min-max normalised to [0, 2] per query to match the graded
    relevance scale used by OllamaJudge.

    Note: this is a cross-encoder model, not an LLM. It is provided as a
    lightweight fallback and produces weaker relevance signals than an LLM
    judge, particularly for out-of-domain (non-MS MARCO) content.
    """

    judge_label = "cross_encoder"

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def load(self) -> None:
        from sentence_transformers import CrossEncoder
        logger.info(f"[CrossEncoderJudge] Loading {self.model_name} on {self.device}")
        self._model = CrossEncoder(self.model_name, max_length=512, device=self.device)
        logger.info("[CrossEncoderJudge] Ready.")

    def score(
        self,
        query: str,
        results: list,
        topk: int = 10,
        query_id: str = "",   # unused, kept for interface compatibility
    ) -> list[float]:
        if self._model is None:
            raise RuntimeError("Call load() before score()")
        pairs = [
            (query, f"{r.title or ''} {r.text or ''}"[:512])
            for r in results[:topk]
        ]
        if not pairs:
            return []
        raw = list(self._model.predict(pairs, show_progress_bar=False))
        lo, hi = min(raw), max(raw)
        if hi == lo:
            return [1.0] * len(raw)
        # Normalise to [0, 2] to match OllamaJudge scale
        return [2.0 * (s - lo) / (hi - lo) for s in raw]


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

AnyJudge = Union[OllamaJudge, OpenAIJudge, CrossEncoderJudge]


# ---------------------------------------------------------------------------
# nDCG@k with continuous relevance scores
# ---------------------------------------------------------------------------

def _dcg(relevances: list[float], k: int = 10) -> float:
    gains = [rel / np.log2(i + 2) for i, rel in enumerate(relevances[:k])]
    return float(np.sum(gains))


def ndcg_at_k(scores: list[float], k: int = 10) -> float:
    """nDCG@k given a list of relevance scores in rank order."""
    ideal = sorted(scores, reverse=True)
    dcg = _dcg(scores, k)
    idcg = _dcg(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Main evaluation function (Tier 3)
# ---------------------------------------------------------------------------

def evaluate_cross_domain(
    results_by_qid: dict[str, list],
    queries: list[dict],
    judge: Optional[AnyJudge] = None,
    top_n: int = 10,
) -> dict:
    """
    Tier 3: Cross-domain fusion quality evaluation (FeB4RAG methodology).

    Metrics:
      domain_coverage@N        — fraction of queries with ≥2 domains in top-N
      mean_domains_in_topN     — average distinct domains in top-N
      expected_domain_hit@N    — fraction of queries where ALL expected domains
                                  appear in top-N results
      llm_nDCG@N               — mean nDCG@N using OllamaJudge weak labels
      cross_encoder_nDCG@N     — mean nDCG@N using CrossEncoderJudge (fallback)

    Args:
        results_by_qid: {query_id: [RetrievalResult, ...]}
        queries:        list of {query_id, text, expected_domains}
        judge:          OllamaJudge or CrossEncoderJudge; if None, skips nDCG
        top_n:          rank cutoff
    """
    query_map = {q["query_id"]: q for q in queries}
    judge_label = judge.judge_label if judge is not None else None

    coverage_scores: list[float] = []
    domain_counts: list[int] = []
    expected_hits: list[float] = []
    ndcg_scores: list[float] = []

    for qid, results in results_by_qid.items():
        q = query_map.get(qid, {})
        query_text = q.get("text", "")
        expected = set(q.get("expected_domains", []))
        top_results = results[:top_n]

        # --- Domain coverage metrics (no annotation needed) ---
        domains_in_top = {r.domain for r in top_results}
        domain_counts.append(len(domains_in_top))
        coverage_scores.append(1.0 if len(domains_in_top) >= 2 else 0.0)

        if expected:
            expected_hits.append(1.0 if expected.issubset(domains_in_top) else 0.0)

        # --- Weak label nDCG ---
        if judge is not None and top_results:
            scores = judge.score(query_text, top_results, topk=top_n, query_id=qid)
            ndcg = ndcg_at_k(scores, k=top_n)
            ndcg_scores.append(ndcg)
        else:
            ndcg_scores.append(float("nan"))

        logger.info(
            f"  {qid}: domains={sorted(domains_in_top)}, "
            f"nDCG@{top_n}={ndcg_scores[-1]:.3f}"
        )

    valid_ndcg = [s for s in ndcg_scores if not np.isnan(s)]

    out: dict = {
        f"domain_coverage@{top_n}":     float(np.mean(coverage_scores)) if coverage_scores else 0.0,
        f"mean_domains_in_top{top_n}":  float(np.mean(domain_counts))   if domain_counts  else 0.0,
        f"expected_domain_hit@{top_n}": float(np.mean(expected_hits))   if expected_hits  else 0.0,
        "n_queries": len(results_by_qid),
    }
    if valid_ndcg and judge_label is not None:
        out[f"{judge_label}_nDCG@{top_n}"] = float(np.mean(valid_ndcg))

    return out
