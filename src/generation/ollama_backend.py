"""
Ollama generation backend.

Calls the local Ollama REST API (same pattern as OllamaJudge in
src/evaluation/cross_domain_eval.py). No pip package needed — uses
requests, which is already a transitive dependency.

Config keys consumed:
  generation.model        — e.g. "llama3.1:8b-instruct-q4_K_M"
  generation.base_url     — e.g. "http://localhost:11434"
  generation.temperature  — float, default 0.1
  generation.max_tokens   — int, default 300
"""
from __future__ import annotations

import logging
import time

from src.retrieval.dense_retriever import RetrievalResult
from .base import (
    BaseGenerator,
    GeneratorResult,
    build_prompt,
    detect_abstention,
    extract_cited_ids,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"
_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_TEMPERATURE = 0.1
_DEFAULT_MAX_TOKENS = 300


class OllamaGenerator(BaseGenerator):
    """Answer generator backed by a local Ollama model."""

    def __init__(self, cfg: dict):
        gen_cfg = cfg.get("generation", {})
        self.model = gen_cfg.get("model", _DEFAULT_MODEL)
        self.base_url = gen_cfg.get("base_url", _DEFAULT_BASE_URL).rstrip("/")
        self.temperature = float(gen_cfg.get("temperature", _DEFAULT_TEMPERATURE))
        self.max_tokens = int(gen_cfg.get("max_tokens", _DEFAULT_MAX_TOKENS))
        self.prompt_mode = str(gen_cfg.get("prompt_mode", "strict"))
        logger.info(
            f"[OllamaGenerator] model={self.model}, url={self.base_url}, "
            f"temperature={self.temperature}, max_tokens={self.max_tokens}, "
            f"prompt_mode={self.prompt_mode}"
        )

    def _check_ollama(self) -> None:
        """Verify Ollama is reachable; raise RuntimeError with a helpful message if not."""
        import requests
        try:
            requests.get(f"{self.base_url}/api/tags", timeout=5).raise_for_status()
        except Exception:
            raise RuntimeError(
                f"Ollama is not reachable at {self.base_url}. "
                "Start it with: ollama serve"
            )

    def generate(
        self,
        query: str,
        passages: list[RetrievalResult],
        seed: int = 42,
    ) -> GeneratorResult:
        import requests

        prompt = build_prompt(query, passages, prompt_mode=self.prompt_mode)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "seed": seed,
            },
        }
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"[OllamaGenerator] API error: {e}")
            return GeneratorResult(
                answer="",
                cited_ids=[],
                model=self.model,
                seed=seed,
                latency_ms=0.0,
                abstained=True,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        answer = resp.json()["message"]["content"].strip()
        cited_ids = extract_cited_ids(answer, passages)
        abstained = detect_abstention(answer)

        return GeneratorResult(
            answer=answer,
            cited_ids=cited_ids,
            model=self.model,
            seed=seed,
            latency_ms=latency_ms,
            abstained=abstained,
        )


def make_generator(cfg: dict) -> BaseGenerator:
    """Factory: pick the backend from cfg.generation.backend."""
    backend = cfg.get("generation", {}).get("backend", "ollama")
    if backend == "ollama":
        gen = OllamaGenerator(cfg)
        gen._check_ollama()
        return gen
    raise ValueError(
        f"Unknown generation backend '{backend}'. Supported: 'ollama'. "
        "To add OpenAI support, implement src/generation/openai_backend.py."
    )
