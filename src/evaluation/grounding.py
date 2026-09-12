"""
Grounding evaluation — causal intervention suite (v2).

Measures whether generated answers are grounded in retrieved passages
using a family of interventions that mirror the dissertation methodology.

Conditions:
  normal           — baseline (full pipeline, top-k passages)
  no_retrieval     — retrieval blocking: generate with no context at all
  swapped_context  — opposite-memory patching: passages from a different query
  random_in_domain — weaker swap control: random corpus passages, same domain
  shuffled_order   — position-sensitivity control: same passages, permuted
  corrupted_context — Faithfulness-QA style: key facts in passages negated/inverted

=== Stratified Response-Type Analysis (new in v2) ===

Raw answer_change_rate conflates three qualitatively different events:
  content_change   — both conditions produced an answer, content differs
                     → genuine grounding signal (model uses context)
  abstention_shift — one condition produced an answer, the other abstained
                     → calibration signal (model detects wrong context)
  both_abstain     — both conditions abstained
                     → context-independent: model cannot answer either way
  both_answer_same — both conditions produced an answer, content is the same
                     → parametric memory: model ignores context

This distinction follows Wallat et al. (2024) "Correctness is not Faithfulness
in RAG Attributions" (arXiv:2412.18004) and Yue et al. (2025) "Does RAG Know
When Retrieval Is Wrong?" (arXiv:2605.14473).

=== Conditional Grounding Score ===

  conditional_grounding_score = min over content-swap conditions of:
      content_change / (content_change + both_answer_same)

Denominator restricts to pairs where BOTH conditions produced actual answers,
eliminating the abstention confound. This is the direct analogue of
Smin = min(S_key, S_ball) from the dissertation, now applied at the level
of answered pairs only.

=== Sentence Faithfulness (new in v2) ===

For each answer, measures the fraction of non-trivial answer sentences whose
embedding is semantically close (cosine > threshold) to at least one retrieved
passage sentence. Approximates RAGAS faithfulness without an NLI model,
using the all-MiniLM-L6-v2 embedder already loaded for change detection.

=== Corrupted Context (new in v2) ===

Implements the counterfactual entity substitution approach from
Faithfulness-QA (arXiv:2604.25313): directional/causal keywords in passages
are lexically inverted and numbers are scaled by 10×, creating a controlled
context-vs-memory conflict. A well-grounded model should produce a different
(or contradicted) answer; a parametric-memory model ignores the corruption.
"""
from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.retrieval.dense_retriever import RetrievalResult
from src.generation.base import BaseGenerator, GeneratorResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _binomial_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _mean_ci(values: np.ndarray, z: float = 1.96) -> tuple[float, float]:
    """Normal-approximation 95 % CI for the mean of a continuous variable."""
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    se = float(np.std(values, ddof=1)) / np.sqrt(n)
    mean = float(np.mean(values))
    return (mean - z * se, mean + z * se)


# ---------------------------------------------------------------------------
# SciFact end-to-end accuracy helpers
# ---------------------------------------------------------------------------

def _extract_claim_verdict(answer: str) -> str:
    """Parse SUPPORTED / REFUTED / NOT ENOUGH INFO from a claim_verify answer.

    Inspects the first 100 characters where the model places its verdict.
    Returns 'UNKNOWN' when no recognisable verdict is present.
    """
    head = answer.strip().upper()[:100]
    if "SUPPORTED" in head:
        return "SUPPORTED"
    if "REFUTED" in head:
        return "REFUTED"
    if "NOT ENOUGH" in head or "INSUFFICIENT" in head:
        return "NOT ENOUGH INFO"
    return "UNKNOWN"


def compute_domain_accuracy(
    df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    domain: str,
) -> dict | None:
    """End-to-end verdict accuracy for domains with gold labels.

    Currently supports the *science* domain (SciFact) where claim_verify
    output (SUPPORTED / REFUTED / NOT ENOUGH INFO) can be compared to qrels.

    Gold-label derivation from BEIR qrels
    --------------------------------------
    - relevance ≥ 1 in qrels  →  SUPPORTED (a document supports the claim)
    - query absent from qrels →  NOT ENOUGH INFO

    Limitation: REFUTED claims cannot be distinguished from NOT Enough Info
    in the BEIR qrels format (no negative-evidence flag). Include this caveat
    when reporting.

    Returns None for domains without evaluable gold labels.
    """
    if domain != "science":
        return None

    supported_qids = set(
        qrels_df[qrels_df["relevance"] >= 1]["query_id"].astype(str)
    )

    norm = df[df["condition"] == "normal"].copy()
    norm["gold_verdict"] = norm["query_id"].apply(
        lambda qid: "SUPPORTED" if qid in supported_qids else "NOT ENOUGH INFO"
    )
    norm["predicted_verdict"] = norm["answer"].apply(_extract_claim_verdict)

    abstain_rate = float(norm["abstained"].mean())
    answered = norm[~norm["abstained"]]
    parse_fail_rate = float(
        (answered["predicted_verdict"] == "UNKNOWN").mean()
    ) if not answered.empty else float("nan")

    valid = answered[answered["predicted_verdict"] != "UNKNOWN"]
    if valid.empty:
        return {
            "accuracy": float("nan"),
            "n_evaluated": 0,
            "n_total": len(norm),
            "abstain_rate": abstain_rate,
            "parse_failure_rate": parse_fail_rate,
            "per_class": {},
            "limitation": (
                "REFUTED claims indistinguishable from NOT ENOUGH INFO "
                "in BEIR qrels format."
            ),
        }

    accuracy = float((valid["predicted_verdict"] == valid["gold_verdict"]).mean())
    lo, hi = _binomial_ci(
        int((valid["predicted_verdict"] == valid["gold_verdict"]).sum()),
        len(valid),
    )

    per_class: dict = {}
    for label in sorted(valid["gold_verdict"].unique()):
        subset = valid[valid["gold_verdict"] == label]
        correct = int((subset["predicted_verdict"] == label).sum())
        per_class[label] = {
            "n": len(subset),
            "accuracy": float(correct / len(subset)) if subset.empty is False else float("nan"),
            "ci_95": _binomial_ci(correct, len(subset)),
        }

    return {
        "accuracy": accuracy,
        "accuracy_ci_95": (lo, hi),
        "n_evaluated": len(valid),
        "n_total": len(norm),
        "abstain_rate": abstain_rate,
        "parse_failure_rate": parse_fail_rate,
        "per_class": per_class,
        "limitation": (
            "REFUTED claims indistinguishable from NOT ENOUGH INFO "
            "in BEIR qrels format."
        ),
    }


# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------

class GroundingCondition(str, Enum):
    normal            = "normal"
    no_retrieval      = "no_retrieval"
    swapped_context   = "swapped_context"
    random_in_domain  = "random_in_domain"
    shuffled_order    = "shuffled_order"
    corrupted_context = "corrupted_context"


# ---------------------------------------------------------------------------
# Corrupted-context passage manipulation (Faithfulness-QA style)
# ---------------------------------------------------------------------------

# Lexical inversions for directional/causal language.
# Each key maps to its semantic opposite; both directions are included.
_DIRECTIONAL_SWAPS: dict[str, str] = {
    "increases":  "decreases",  "decreases":  "increases",
    "increase":   "decrease",   "decrease":   "increase",
    "increased":  "decreased",  "decreased":  "increased",
    "improving":  "worsening",  "worsening":  "improving",
    "improves":   "worsens",    "worsens":    "improves",
    "improve":    "worsen",     "worsen":     "improve",
    "improved":   "worsened",   "worsened":   "improved",
    "promotes":   "inhibits",   "inhibits":   "promotes",
    "promote":    "inhibit",    "inhibit":    "promote",
    "promoted":   "inhibited",  "inhibited":  "promoted",
    "enhances":   "suppresses", "suppresses": "enhances",
    "enhance":    "suppress",   "suppress":   "enhance",
    "positive":   "negative",   "negative":   "positive",
    "beneficial": "harmful",    "harmful":    "beneficial",
    "higher":     "lower",      "lower":      "higher",
    "greater":    "lesser",     "lesser":     "greater",
    "more":       "less",       "less":       "more",
    "significant":"negligible", "negligible": "significant",
    "effective":  "ineffective","ineffective":"effective",
    "associated": "unassociated",
}


def corrupt_passage_text(text: str, rng: random.Random) -> str:
    """Apply controlled corruptions to a passage for the corrupted_context condition.

    Two transformations (following Faithfulness-QA, arXiv:2604.25313):
    1. Lexical inversion of directional/causal keywords.
    2. Numeric scaling: numbers × 10 (or × 0.1 randomly).
    """
    # 1. Lexical inversions (case-preserving)
    tokens = re.split(r'(\W+)', text)  # keep separators
    for i, tok in enumerate(tokens):
        lower = tok.lower()
        if lower in _DIRECTIONAL_SWAPS:
            replacement = _DIRECTIONAL_SWAPS[lower]
            if tok[0].isupper():
                replacement = replacement.capitalize()
            tokens[i] = replacement
    result = ''.join(tokens)

    # 2. Numeric scaling
    def scale_number(m: re.Match) -> str:
        val = float(m.group(0))
        factor = 10.0 if rng.random() > 0.5 else 0.1
        new_val = val * factor
        return str(int(new_val)) if new_val == int(new_val) else f"{new_val:.2g}"

    result = re.sub(r'\b\d+(?:\.\d+)?\b', scale_number, result)
    return result


# ---------------------------------------------------------------------------
# Passage selection per condition
# ---------------------------------------------------------------------------

def build_condition_passages(
    condition: GroundingCondition,
    query_results: list[RetrievalResult],
    all_results: dict[str, list[RetrievalResult]],
    corpus_df: Optional[pd.DataFrame],
    query_id: str,
    topk: int,
    rng: random.Random,
) -> list[RetrievalResult]:
    """Return the passage list to use for the given intervention condition.

    Args:
        condition:      Which intervention to apply.
        query_results:  Top-k results for the current query (normal pipeline output).
        all_results:    Dict of {query_id: [RetrievalResult]} for all queries in
                        this run — used by swapped_context to pick a different query.
        corpus_df:      Full corpus DataFrame for the domain (columns: doc_id, text,
                        title). None disables random_in_domain.
        query_id:       Current query's ID (excluded from swap candidates).
        topk:           Number of passages to return.
        rng:            Seeded random.Random instance for reproducibility.
    """
    if condition == GroundingCondition.normal:
        return query_results[:topk]

    if condition == GroundingCondition.no_retrieval:
        return []

    if condition == GroundingCondition.swapped_context:
        other_qids = [qid for qid in all_results if qid != query_id]
        if not other_qids:
            logger.warning("[grounding] swapped_context: only one query, falling back to normal")
            return query_results[:topk]
        swap_qid = rng.choice(other_qids)
        return all_results[swap_qid][:topk]

    if condition == GroundingCondition.random_in_domain:
        if corpus_df is None or len(corpus_df) == 0:
            logger.warning("[grounding] random_in_domain: corpus not available, falling back to swapped")
            return build_condition_passages(
                GroundingCondition.swapped_context, query_results, all_results,
                corpus_df, query_id, topk, rng,
            )
        sample = corpus_df.sample(
            n=min(topk, len(corpus_df)),
            random_state=rng.randint(0, 2 ** 31),
        )
        results: list[RetrievalResult] = []
        for _, row in sample.iterrows():
            raw_id = str(row.get("chunk_id", row.get("doc_id", "rand")))
            results.append(RetrievalResult(
                chunk_id=raw_id, doc_id=raw_id, domain="random",
                score=0.0, text=str(row.get("text", "")),
                title=str(row.get("title", "")),
            ))
        return results

    if condition == GroundingCondition.shuffled_order:
        shuffled = list(query_results[:topk])
        rng.shuffle(shuffled)
        return shuffled

    if condition == GroundingCondition.corrupted_context:
        corrupted: list[RetrievalResult] = []
        for p in query_results[:topk]:
            corrupted.append(RetrievalResult(
                chunk_id=p.chunk_id, doc_id=p.doc_id, domain=p.domain,
                score=p.score,
                text=corrupt_passage_text(p.text, rng),
                title=p.title,
            ))
        return corrupted

    raise ValueError(f"Unknown condition: {condition}")


# ---------------------------------------------------------------------------
# Response-type classification
# ---------------------------------------------------------------------------

class ResponseType(str):
    CONTENT_CHANGE    = "content_change"     # both answered, different content
    BOTH_ANSWER_SAME  = "both_answer_same"   # both answered, same content
    ABSTENTION_SHIFT  = "abstention_shift"   # one abstained, one answered
    BOTH_ABSTAIN      = "both_abstain"       # both abstained


def classify_response_type(
    normal_abstained: bool,
    condition_abstained: bool,
    similarity: float,
    threshold: float,
) -> str:
    """Classify the relationship between a normal-condition answer and a
    intervention-condition answer into one of four mutually exclusive types.

    See module docstring for interpretation of each type.
    """
    if normal_abstained and condition_abstained:
        return ResponseType.BOTH_ABSTAIN
    if not normal_abstained and not condition_abstained:
        if similarity < threshold:
            return ResponseType.CONTENT_CHANGE
        return ResponseType.BOTH_ANSWER_SAME
    return ResponseType.ABSTENTION_SHIFT


# ---------------------------------------------------------------------------
# Similarity metric (existing)
# ---------------------------------------------------------------------------

def compute_answer_similarity(a: str, b: str, embedder) -> float:
    """Cosine similarity between two answer strings via sentence-transformers."""
    if not a.strip() or not b.strip():
        return 0.0
    vecs = embedder.encode([a, b], normalize_embeddings=True)
    return float(np.dot(vecs[0], vecs[1]))


# ---------------------------------------------------------------------------
# Sentence faithfulness (new in v2)
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str, min_len: int = 25) -> list[str]:
    """Split text into sentences, filtering very short ones."""
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if len(s.strip()) >= min_len]


def compute_sentence_faithfulness(
    answer: str,
    passages: list[RetrievalResult],
    embedder,
    sim_threshold: float = 0.70,
) -> float:
    """Fraction of non-trivial answer sentences semantically entailed by at
    least one retrieved passage.

    Approximates RAGAS faithfulness using cosine similarity instead of NLI,
    avoiding a second model download while reusing the all-MiniLM-L6-v2
    embedder already loaded for change-rate measurement.

    Returns NaN when the answer or passages are empty (e.g., abstention).
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return float('nan')
    passage_texts = [p.text for p in passages if p.text.strip()]
    if not passage_texts:
        return 0.0

    all_texts = sentences + passage_texts
    embs = embedder.encode(all_texts, normalize_embeddings=True)
    ans_embs  = embs[:len(sentences)]
    pass_embs = embs[len(sentences):]

    # (n_sentences × n_passages) cosine similarity matrix
    sim_matrix = ans_embs @ pass_embs.T
    max_sims = sim_matrix.max(axis=1)          # best passage match per sentence
    return float((max_sims >= sim_threshold).mean())


# ---------------------------------------------------------------------------
# Citation support (v2 — replaces trivially-satisfied citation_precision)
# ---------------------------------------------------------------------------

def compute_citation_support(
    answer: str,
    passages: list[RetrievalResult],
    cited_ids: list[str],
    embedder,
    sim_threshold: float = 0.70,
) -> float:
    """Fraction of cited passages that semantically support the answer.

    For each cited passage, computes the maximum cosine similarity between
    any answer sentence embedding and any passage sentence embedding.
    A citation is counted as 'supported' when that max similarity ≥ threshold.

    This replaces the old citation_precision, which was trivially 1.0 because
    [P1]–[P5] position indices always resolved to passage IDs that existed in
    the passage list — an existence check, not an entailment check.

    Returns NaN for abstained answers (no cited_ids); returns 0.0 when
    cited passages exist but share no semantic content with the answer.
    """
    if not cited_ids:
        return float('nan')

    id_to_text = {p.doc_id: p.text for p in passages}
    cited_texts = [
        id_to_text[cid] for cid in cited_ids
        if cid in id_to_text and id_to_text[cid].strip()
    ]
    if not cited_texts:
        return 0.0

    ans_sentences = _split_sentences(answer)
    if not ans_sentences:
        return float('nan')

    supported = 0
    for cited_text in cited_texts:
        pass_sentences = _split_sentences(cited_text)
        if not pass_sentences:
            continue
        all_texts = ans_sentences + pass_sentences
        embs = embedder.encode(all_texts, normalize_embeddings=True)
        ans_embs  = embs[:len(ans_sentences)]
        pass_embs = embs[len(ans_sentences):]
        sim_matrix = ans_embs @ pass_embs.T
        if float(sim_matrix.max()) >= sim_threshold:
            supported += 1

    return supported / len(cited_texts)


# ---------------------------------------------------------------------------
# Per-query result row
# ---------------------------------------------------------------------------

@dataclass
class GroundingRow:
    query_id: str
    condition: str
    seed: int
    answer: str
    abstained: bool
    # --- similarity & change ---
    similarity_to_normal: float
    answer_changed: bool              # similarity < threshold (v1 metric, kept)
    response_type: str                # one of ResponseType.* (new in v2)
    # --- faithfulness ---
    sentence_faithfulness: float      # NaN when answer empty
    citation_support: float           # NaN when abstained; fraction of cited passages that entail answer
    latency_ms: float


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_grounding_eval(
    pipeline,
    generator: BaseGenerator,
    queries_df: pd.DataFrame,
    domain: str,
    cfg: dict,
    corpus_df: Optional[pd.DataFrame] = None,
    conditions: Optional[list[str]] = None,
    checkpoint_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run the full grounding intervention suite.

    Returns a DataFrame with one row per (query × condition × seed).

    Args:
        checkpoint_path: If given, saves an intermediate parquet after each
            query so that a crashed run can be resumed. On startup the file
            is read and already-complete queries are skipped.
    """
    grounding_cfg = cfg.get("grounding", {})
    topk      = cfg.get("generation", {}).get("topk_passages", 5)
    n_seeds   = int(grounding_cfg.get("n_seeds", 3))
    threshold = float(grounding_cfg.get("similarity_threshold", 0.90))
    base_seed = int(cfg.get("generation", {}).get("seed", 42))

    if conditions is None:
        conditions = grounding_cfg.get(
            "conditions",
            ["normal", "no_retrieval", "swapped_context",
             "random_in_domain", "shuffled_order"],
        )
    condition_enums = [GroundingCondition(c) for c in conditions]

    # Load corpus for random_in_domain
    if corpus_df is None and GroundingCondition.random_in_domain in condition_enums:
        corpus_path = Path(cfg["paths"]["data_root"]) / domain / "corpus.parquet"
        if corpus_path.exists():
            logger.info(f"[grounding:{domain}] Loading corpus from {corpus_path}")
            corpus_df = pd.read_parquet(corpus_path, columns=["doc_id", "text", "title"])
        else:
            logger.warning(f"[grounding:{domain}] corpus.parquet not found; random_in_domain disabled")

    logger.info("[grounding] Loading embedder (all-MiniLM-L6-v2) …")
    embedder = _get_embedder()

    # Step 1: pipeline inference for all queries
    logger.info(f"[grounding:{domain}] Pipeline inference for {len(queries_df)} queries …")
    all_results: dict[str, list[RetrievalResult]] = {}
    for _, qrow in queries_df.iterrows():
        qid = str(qrow["query_id"])
        try:
            result = pipeline.search(str(qrow["text"]))
            all_results[qid] = result["results"]
        except Exception as e:
            logger.warning(f"[grounding:{domain}] pipeline error for {qid}: {e}")
            all_results[qid] = []

    # ── Checkpoint resume ─────────────────────────────────────────────────────
    accumulated: list[dict] = []
    done_qids: set[str] = set()

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt_df = pd.read_parquet(checkpoint_path)
        accumulated = ckpt_df.to_dict(orient="records")
        expected = len(condition_enums) * n_seeds
        counts = ckpt_df.groupby("query_id").size()
        done_qids = set(counts[counts >= expected].index)
        logger.info(
            f"[grounding:{domain}] Checkpoint found — "
            f"{len(done_qids)} queries already complete, resuming …"
        )

    # Step 2: generation under each condition
    n_done = 0

    for _, qrow in queries_df.iterrows():
        qid        = str(qrow["query_id"])
        query_text = str(qrow["text"])
        query_results = all_results.get(qid, [])

        if qid in done_qids:
            n_done += 1
            continue

        for seed_offset in range(n_seeds):
            seed = base_seed + seed_offset
            rng  = random.Random(seed)

            answers:      dict[str, GeneratorResult]    = {}
            passage_lists: dict[str, list[RetrievalResult]] = {}

            for cond in condition_enums:
                passages = build_condition_passages(
                    cond, query_results, all_results, corpus_df,
                    qid, topk, rng,
                )
                passage_lists[cond.value] = passages
                try:
                    gen = generator.generate(query_text, passages, seed=seed)
                except Exception as e:
                    logger.warning(f"[grounding:{domain}] gen error ({cond.value}, {qid}): {e}")
                    gen = GeneratorResult(
                        answer="", cited_ids=[], model="error",
                        seed=seed, latency_ms=0.0, abstained=True,
                    )
                answers[cond.value] = gen

            # Normal-condition answer used as reference for similarity and response_type
            normal_gen     = answers.get("normal")
            normal_answer  = normal_gen.answer  if normal_gen else ""
            normal_abstain = normal_gen.abstained if normal_gen else True

            for cond in condition_enums:
                gen = answers[cond.value]
                passages = passage_lists[cond.value]

                sim = compute_answer_similarity(normal_answer, gen.answer, embedder)

                rtype = (
                    "reference"   # normal condition is its own reference
                    if cond == GroundingCondition.normal
                    else classify_response_type(
                        normal_abstain, gen.abstained, sim, threshold
                    )
                )

                # Sentence faithfulness (only meaningful when there are passages
                # and the model produced an actual answer)
                faith = (
                    compute_sentence_faithfulness(gen.answer, passages, embedder)
                    if not gen.abstained and passages
                    else float('nan')
                )

                # Citation support: only when model cited something AND had passages
                cit_support = (
                    compute_citation_support(gen.answer, passages, gen.cited_ids, embedder)
                    if not gen.abstained and passages and gen.cited_ids
                    else float('nan')
                )

                accumulated.append(dict(
                    query_id=qid,
                    condition=cond.value,
                    seed=seed,
                    answer=gen.answer,
                    abstained=gen.abstained,
                    similarity_to_normal=sim,
                    answer_changed=(sim < threshold),
                    response_type=rtype,
                    sentence_faithfulness=faith,
                    citation_support=cit_support,
                    latency_ms=gen.latency_ms,
                ))

        n_done += 1
        if n_done % 10 == 0:
            logger.info(f"[grounding:{domain}] {n_done}/{len(queries_df)} queries done")

        # ── Checkpoint save ───────────────────────────────────────────────────
        if checkpoint_path is not None and accumulated:
            pd.DataFrame(accumulated).to_parquet(checkpoint_path, index=False)

    return pd.DataFrame(accumulated)


def _get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Summary aggregation (v2: stratified + conditional grounding score)
# ---------------------------------------------------------------------------

def grounding_summary(df: pd.DataFrame, similarity_threshold: float = 0.90) -> dict:
    """Aggregate per-row results into per-condition summary + grounding scores.

    v1 metric: grounding_score = min(raw answer_change_rate) over content-swap conds.
    v2 metric: conditional_grounding_score = min(content_change / answered_pairs)
               over content-swap conds — excludes abstentions from denominator.

    Wallat et al. (arXiv:2412.18004): "correctness ≠ faithfulness" — raw change
    rate conflates genuine context use with model calibration (abstention).
    The conditional score isolates the grounding signal.
    """
    if df.empty:
        return {}

    summary: dict = {}
    for cond, group in df.groupby("condition"):
        n = len(group)
        rt = group["response_type"].value_counts()

        # Stratified response type counts
        cc  = int(rt.get(ResponseType.CONTENT_CHANGE,   0))
        bsa = int(rt.get(ResponseType.BOTH_ANSWER_SAME, 0))
        ast = int(rt.get(ResponseType.ABSTENTION_SHIFT, 0))
        ba  = int(rt.get(ResponseType.BOTH_ABSTAIN,     0))

        answered_pairs = cc + bsa
        cond_gs = cc / answered_pairs if answered_pairs > 0 else float("nan")
        cond_gs_ci = _binomial_ci(cc, answered_pairs) if answered_pairs > 0 else (float("nan"), float("nan"))

        # Sentence faithfulness and citation support (NaN for abstentions)
        faith_vals = group["sentence_faithfulness"].dropna()
        cit_vals   = group["citation_support"].dropna()

        abstain_k = int(group["abstained"].sum())
        abstain_ci = _binomial_ci(abstain_k, n)
        faith_ci = _mean_ci(faith_vals.values) if len(faith_vals) >= 2 else (float("nan"), float("nan"))
        cit_ci   = _mean_ci(cit_vals.values)   if len(cit_vals)   >= 2 else (float("nan"), float("nan"))

        summary[cond] = {
            # v1 metrics (kept for backward compatibility)
            "answer_change_rate":   float(group["answer_changed"].mean()),
            "mean_similarity":      float(group["similarity_to_normal"].mean()),
            "abstention_rate":      float(group["abstained"].mean()),
            "abstention_rate_ci95": abstain_ci,
            # v2: stratified response types
            "content_change_rate":   cc  / n,
            "both_answer_same_rate": bsa / n,
            "abstention_shift_rate": ast / n,
            "both_abstain_rate":     ba  / n,
            "n_answered_pairs":      answered_pairs,
            # v2: conditional grounding (answered pairs only)
            "conditional_grounding":       cond_gs,
            "conditional_grounding_ci95":  cond_gs_ci,
            "conditional_grounding_n":     answered_pairs,  # sample size for CGS
            # v2: sentence faithfulness
            "sentence_faithfulness":       float(faith_vals.mean()) if len(faith_vals) else float("nan"),
            "sentence_faithfulness_ci95":  faith_ci,
            "sentence_faithfulness_n":     int(len(faith_vals)),
            # v2: citation support
            "citation_support":            float(cit_vals.mean()) if len(cit_vals) else float("nan"),
            "citation_support_ci95":       cit_ci,
            "citation_support_n":          int(len(cit_vals)),
            "n_rows":                      n,
        }

    # --- Grounding scores ---
    content_swap = ["swapped_context", "random_in_domain"]

    # v1: raw min change rate
    raw_rates = [summary[c]["answer_change_rate"] for c in content_swap if c in summary]
    grounding_score_v1 = min(raw_rates) if raw_rates else float('nan')

    # v2: conditional min (answered pairs only) — the primary metric
    cond_rates = [
        summary[c]["conditional_grounding"]
        for c in content_swap
        if c in summary and not np.isnan(summary[c]["conditional_grounding"])
    ]
    grounding_score_v2 = min(cond_rates) if cond_rates else float('nan')

    return {
        "per_condition": summary,
        # v1 (kept for comparison)
        "grounding_score_v1": grounding_score_v1,
        "grounding_score_v1_note": (
            "Raw answer_change_rate min — includes abstention as 'changed'. "
            "Inflated when model abstains under wrong context."
        ),
        # v2 (primary)
        "grounding_score": grounding_score_v2,
        "grounding_score_note": (
            "Conditional: content_change / answered_pairs, min over content-swap "
            "conditions. Excludes abstention from denominator per "
            "Wallat et al. (arXiv:2412.18004). Direct analogue of "
            "Smin = min(S_key, S_ball) from dissertation."
        ),
    }


def format_summary_markdown(
    summary: dict,
    domain: str,
    n_queries: int,
    n_seeds: int,
) -> str:
    """Render the grounding summary as a markdown report."""
    gs_v1 = summary.get("grounding_score_v1", float('nan'))
    gs_v2 = summary.get("grounding_score", float('nan'))

    def fmt(v) -> str:
        return f"{v:.3f}" if not np.isnan(v) else "N/A"

    lines = [
        f"# Grounding Evaluation — {domain}",
        f"\n{n_queries} queries × {n_seeds} seeds\n",
        "## Response-type breakdown per condition",
        "",
        "| Condition | content_change | both_same | abstention_shift | both_abstain "
        "| cond_grounding | faithfulness | cit_support |",
        "|---|---|---|---|---|---|---|---|",
    ]

    per = summary.get("per_condition", {})
    ordered = [
        "normal", "no_retrieval", "swapped_context",
        "random_in_domain", "shuffled_order", "corrupted_context",
    ]
    small_n_warnings: list[str] = []

    for cond in ordered:
        if cond not in per:
            continue
        s = per[cond]

        cgs_val = s["conditional_grounding"]
        cgs_n   = s.get("conditional_grounding_n", 0)
        cgs_ci  = s.get("conditional_grounding_ci95", (float("nan"), float("nan")))

        if np.isnan(cgs_val):
            cgs_str = "N/A"
        else:
            warn = " ⚠️" if cgs_n < 30 else ""
            cgs_str = f"{cgs_val:.3f}{warn} (n={cgs_n})"
            if cgs_n < 30:
                small_n_warnings.append(
                    f"`{cond}` CGS based on only n={cgs_n} answered pairs "
                    f"(95% CI [{cgs_ci[0]:.3f}, {cgs_ci[1]:.3f}]) — interpret with caution."
                )

        faith_val = s["sentence_faithfulness"]
        faith_n   = s.get("sentence_faithfulness_n", 0)
        faith_ci  = s.get("sentence_faithfulness_ci95", (float("nan"), float("nan")))
        faith_str = (
            f"{faith_val:.3f} ±{(faith_ci[1]-faith_val):.3f} (n={faith_n})"
            if not np.isnan(faith_val)
            else "N/A"
        )

        cit_val  = s["citation_support"]
        cit_str  = fmt(cit_val)

        abstain_ci = s.get("abstention_rate_ci95", (float("nan"), float("nan")))
        abstain_str = (
            f"{s['abstention_rate']:.3f} [{abstain_ci[0]:.3f},{abstain_ci[1]:.3f}]"
        )

        lines.append(
            f"| {cond} "
            f"| {s['content_change_rate']:.3f} "
            f"| {s['both_answer_same_rate']:.3f} "
            f"| {s['abstention_shift_rate']:.3f} "
            f"| {s['both_abstain_rate']:.3f} "
            f"| {cgs_str} "
            f"| {faith_str} "
            f"| {cit_str} |"
        )

    if small_n_warnings:
        lines += ["", "### ⚠️ Small-sample warnings"]
        for w in small_n_warnings:
            lines.append(f"- {w}")

    lines += [
        "",
        "## Grounding scores",
        "",
        f"| Metric | Value | Note |",
        f"|---|---|---|",
        f"| Grounding score (v2, conditional) | **{fmt(gs_v2)}** | "
        f"content_change / answered pairs, min over swapped+random |",
        f"| Grounding score (v1, raw) | {fmt(gs_v1)} | "
        f"raw answer_change_rate min — includes abstention as 'changed' |",
        "",
        "## Interpretation",
        "",
        "- **content_change**: both conditions produced answers with different content",
        "  — genuine grounding signal (model uses retrieved context).",
        "- **both_answer_same**: both conditions produced the same answer",
        "  — parametric memory dominates; retrieval has no effect.",
        "- **abstention_shift**: one condition produced an answer, the other abstained",
        "  — model calibration; detects wrong context but doesn't generate differently.",
        "- **both_abstain**: both conditions abstained",
        "  — context-independent; model cannot answer regardless.",
        "",
        "> `shuffled_order` is a position-sensitivity control and is excluded from",
        "> the conditional grounding score.",
        "> `corrupted_context` (if run) tests whether the model follows false evidence.",
    ]
    return "\n".join(lines)
