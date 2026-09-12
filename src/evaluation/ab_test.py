"""
ab_test.py — Statistical A/B comparison for RAG grounding evaluation.

Compares two run_grounding_eval output DataFrames (control vs treatment)
across a predefined set of per-condition metrics using appropriate tests:

  Proportions  (abstain_rate, content_change_rate)
      → two-proportion z-test (chi-square equivalent for 2×2 tables)

  Continuous   (sentence_faithfulness, citation_support, similarity_to_normal)
      → paired Welch's t-test on matched (query_id, seed) pairs
        falls back to unpaired Welch's t-test when pairing fails

Bonferroni correction is applied across all n_tests simultaneously to
control family-wise error rate (FWER). Effect sizes use Cohen's h for
proportions and Cohen's d for continuous metrics.

Quick start
-----------
>>> import pandas as pd
>>> from src.evaluation.ab_test import compare_grounding, format_ab_report
>>> ctrl = pd.read_parquet("results/grounding_v3/finance_raw.parquet")
>>> trt  = pd.read_parquet("results/grounding_ab/finance_raw.parquet")
>>> result = compare_grounding(ctrl, trt, "strict", "partial", "finance")
>>> print(format_ab_report(result))
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_rel, ttest_ind, false_discovery_control


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class MetricResult:
    metric: str
    condition: str
    better_is: Literal["higher", "lower"]
    control_mean: float
    treatment_mean: float
    delta: float            # treatment − control  (positive = treatment is higher)
    relative_pct: float     # delta / |control| × 100
    p_value: float          # raw (uncorrected)
    p_value_corrected: float  # Bonferroni-corrected; filled after all metrics collected
    effect_size: float      # Cohen's h (proportions) or Cohen's d (continuous)
    n_control: int
    n_treatment: int
    test_type: str          # "z-test" | "paired t-test" | "t-test (unpaired)"

    @property
    def significant(self) -> bool:
        return self.p_value_corrected < 0.05

    @property
    def direction(self) -> Literal["improvement", "regression", "neutral"]:
        if not self.significant:
            return "neutral"
        return "improvement" if (
            (self.better_is == "higher" and self.delta > 0) or
            (self.better_is == "lower"  and self.delta < 0)
        ) else "regression"


@dataclass
class ABTestResult:
    control_name: str
    treatment_name: str
    domain: str
    n_queries: int
    n_seeds: int
    alpha: float
    n_tests: int
    correction: str = "bh"   # "bh" (Benjamini-Hochberg FDR) or "bonferroni"
    metrics: list[MetricResult] = field(default_factory=list)

    @property
    def wins(self)   -> int: return sum(1 for m in self.metrics if m.direction == "improvement")
    @property
    def losses(self) -> int: return sum(1 for m in self.metrics if m.direction == "regression")
    @property
    def ties(self)   -> int: return sum(1 for m in self.metrics if m.direction == "neutral")


# ── Statistical primitives ────────────────────────────────────────────────────

def _cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions (arcsine transformation)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        phi1 = 2 * np.arcsin(np.sqrt(np.clip(p1, 0, 1)))
        phi2 = 2 * np.arcsin(np.sqrt(np.clip(p2, 0, 1)))
    return float(phi2 - phi1)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled Cohen's d (treatment − control) effect size."""
    va = np.var(a, ddof=1) if len(a) > 1 else 0.0
    vb = np.var(b, ddof=1) if len(b) > 1 else 0.0
    pooled = np.sqrt((va + vb) / 2)
    return float((np.mean(b) - np.mean(a)) / pooled) if pooled > 1e-12 else 0.0


def _two_prop_z(n1: int, k1: int, n2: int, k2: int) -> tuple[float, float]:
    """Two-proportion z-test → (z_stat, two-sided p-value)."""
    p1, p2 = k1 / max(n1, 1), k2 / max(n2, 1)
    p_pool = (k1 + k2) / max(n1 + n2, 1)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / max(n1, 1) + 1 / max(n2, 1)))
    if se < 1e-12:
        return 0.0, 1.0
    z = (p2 - p1) / se
    return float(z), float(2 * (1 - norm.cdf(abs(z))))


# ── Metric builders ───────────────────────────────────────────────────────────

def _prop_metric(
    ctrl: pd.DataFrame,
    trt: pd.DataFrame,
    col: str,
    condition: str,
    better_is: Literal["higher", "lower"],
) -> MetricResult | None:
    """Compare a binary (bool/0-1) column as a proportion."""
    c = ctrl[col].dropna().astype(float)
    t = trt[col].dropna().astype(float)
    if c.empty or t.empty:
        return None
    n1, k1 = len(c), int(c.sum())
    n2, k2 = len(t), int(t.sum())
    p1, p2 = k1 / n1, k2 / n2
    _, p_raw = _two_prop_z(n1, k1, n2, k2)
    return MetricResult(
        metric=col, condition=condition, better_is=better_is,
        control_mean=p1, treatment_mean=p2,
        delta=p2 - p1,
        relative_pct=((p2 - p1) / max(abs(p1), 1e-9)) * 100,
        p_value=p_raw, p_value_corrected=p_raw,
        effect_size=_cohens_h(p1, p2),
        n_control=n1, n_treatment=n2, test_type="z-test",
    )


def _cont_metric(
    ctrl: pd.DataFrame,
    trt: pd.DataFrame,
    col: str,
    condition: str,
    better_is: Literal["higher", "lower"],
    merge_keys: tuple[str, ...] = ("query_id", "seed"),
) -> MetricResult | None:
    """Compare a continuous column — paired t-test when query_id+seed match."""
    keys = list(merge_keys)
    merged = pd.merge(
        ctrl[keys + [col]].dropna(subset=[col]),
        trt[keys + [col]].dropna(subset=[col]),
        on=keys, suffixes=("_c", "_t"),
    )
    if len(merged) >= 2:
        a, b = merged[f"{col}_c"].values, merged[f"{col}_t"].values
        _, p_raw = ttest_rel(a, b)
        test_type = "paired t-test"
    else:
        a = ctrl[col].dropna().values
        b = trt[col].dropna().values
        if len(a) < 2 or len(b) < 2:
            return None
        _, p_raw = ttest_ind(a, b, equal_var=False)
        test_type = "t-test (unpaired)"
    if np.isnan(p_raw):
        p_raw = 1.0
    return MetricResult(
        metric=col, condition=condition, better_is=better_is,
        control_mean=float(np.mean(a)), treatment_mean=float(np.mean(b)),
        delta=float(np.mean(b) - np.mean(a)),
        relative_pct=((np.mean(b) - np.mean(a)) / max(abs(np.mean(a)), 1e-9)) * 100,
        p_value=float(p_raw), p_value_corrected=float(p_raw),
        effect_size=_cohens_d(a, b),
        n_control=len(a), n_treatment=len(b), test_type=test_type,
    )


def _cgs_metric(
    ctrl: pd.DataFrame,
    trt: pd.DataFrame,
    condition: str,
) -> MetricResult | None:
    """Conditional grounding score = content_change / (content_change + both_answer_same)."""
    def _counts(df: pd.DataFrame) -> tuple[int, int]:
        cc  = (df["response_type"] == "content_change").sum()
        bsa = (df["response_type"] == "both_answer_same").sum()
        return int(cc), int(cc + bsa)

    cc1, d1 = _counts(ctrl)
    cc2, d2 = _counts(trt)
    if d1 == 0 and d2 == 0:
        return None
    p1 = cc1 / d1 if d1 > 0 else float("nan")
    p2 = cc2 / d2 if d2 > 0 else float("nan")
    if np.isnan(p1) or np.isnan(p2):
        return None
    _, p_raw = _two_prop_z(d1, cc1, d2, cc2)
    return MetricResult(
        metric="conditional_grounding_score", condition=condition,
        better_is="higher",
        control_mean=p1, treatment_mean=p2,
        delta=p2 - p1,
        relative_pct=((p2 - p1) / max(abs(p1), 1e-9)) * 100,
        p_value=p_raw, p_value_corrected=p_raw,
        effect_size=_cohens_h(p1, p2),
        n_control=d1, n_treatment=d2, test_type="z-test",
    )


# ── Main comparison entry point ───────────────────────────────────────────────

_CONDITIONS_CGS = [
    "swapped_context", "random_in_domain",
    "shuffled_order", "corrupted_context", "no_retrieval",
]
_CONDITIONS_CONT = [
    "swapped_context", "random_in_domain",
    "shuffled_order", "corrupted_context",
]


def compare_grounding(
    ctrl_df: pd.DataFrame,
    trt_df: pd.DataFrame,
    control_name: str,
    treatment_name: str,
    domain: str,
    alpha: float = 0.05,
) -> ABTestResult:
    """
    Compare two grounding eval DataFrames and return a statistical summary.

    Parameters
    ----------
    ctrl_df, trt_df  : raw parquet DataFrames from run_grounding_eval
    control_name     : short label for the control arm (e.g. "strict")
    treatment_name   : short label for the treatment arm (e.g. "partial")
    domain           : domain name for labelling
    alpha            : family-wise error rate (Bonferroni applied)

    Returns
    -------
    ABTestResult with Bonferroni-corrected per-metric results
    """
    raw: list[MetricResult] = []

    # ── normal condition ───────────────────────────────────────────────────────
    cn = ctrl_df[ctrl_df["condition"] == "normal"]
    tn = trt_df[trt_df["condition"] == "normal"]

    m = _prop_metric(cn, tn, "abstained", "normal", "lower")
    if m: raw.append(m)

    for col in ("sentence_faithfulness", "citation_support", "similarity_to_normal"):
        m = _cont_metric(cn, tn, col, "normal", "higher")
        if m: raw.append(m)

    # ── per-condition CGS ──────────────────────────────────────────────────────
    for cond in _CONDITIONS_CGS:
        cc = ctrl_df[ctrl_df["condition"] == cond]
        tc = trt_df[trt_df["condition"] == cond]
        if cc.empty and tc.empty:
            continue
        m = _cgs_metric(cc, tc, cond)
        if m: raw.append(m)

    # ── per-condition continuous ───────────────────────────────────────────────
    for cond in _CONDITIONS_CONT:
        cc = ctrl_df[ctrl_df["condition"] == cond]
        tc = trt_df[trt_df["condition"] == cond]
        for col in ("sentence_faithfulness", "citation_support"):
            m = _cont_metric(cc, tc, col, cond, "higher")
            if m: raw.append(m)

    # ── per-condition abstention (lower is better) ─────────────────────────────
    for cond in ("swapped_context", "random_in_domain", "shuffled_order", "corrupted_context"):
        cc = ctrl_df[ctrl_df["condition"] == cond]
        tc = trt_df[trt_df["condition"] == cond]
        m = _prop_metric(cc, tc, "abstained", cond, "lower")
        if m: raw.append(m)

    # ── Multiple-comparison correction ────────────────────────────────────────
    # Benjamini-Hochberg FDR (scipy ≥ 1.9): controls expected false-discovery
    # rate, more powerful than Bonferroni for correlated tests (same-domain
    # metrics share a common model and query sample).
    n_tests   = len(raw)
    p_raw     = [m.p_value for m in raw]
    p_adj     = list(false_discovery_control(p_raw, method="bh")) if n_tests > 0 else []
    for m, p_corr in zip(raw, p_adj):
        m.p_value_corrected = float(min(1.0, p_corr))

    n_queries = ctrl_df["query_id"].nunique()
    n_seeds   = ctrl_df["seed"].nunique()

    return ABTestResult(
        control_name=control_name,
        treatment_name=treatment_name,
        domain=domain,
        n_queries=n_queries,
        n_seeds=n_seeds,
        alpha=alpha,
        n_tests=n_tests,
        correction="bh",
        metrics=raw,
    )


# ── Report formatting ─────────────────────────────────────────────────────────

def _effect_label(h: float) -> str:
    h = abs(h)
    if h < 0.2:  return "negligible"
    if h < 0.5:  return "small"
    if h < 0.8:  return "medium"
    return "large"


def format_ab_report(result: ABTestResult) -> str:
    """Return a Markdown-formatted A/B test report."""
    correction_label = (
        "Benjamini-Hochberg FDR" if result.correction == "bh" else "Bonferroni FWER"
    )
    lines = [
        f"# A/B Test Report — {result.domain}",
        "",
        f"| | |",
        f"|---|---|",
        f"| Control | `{result.control_name}` |",
        f"| Treatment | `{result.treatment_name}` |",
        f"| Queries × seeds | {result.n_queries} × {result.n_seeds} = "
        f"{result.n_queries * result.n_seeds} observations per condition |",
        f"| Tests (n) | {result.n_tests} |",
        f"| Multiple-comparison correction | {correction_label} (α = {result.alpha}) |",
        f"| **Verdict** | **{result.wins} improvements · {result.losses} regressions · {result.ties} ties** |",
        "",
        "## Per-metric results",
        "",
        "| Condition | Metric | Control | Treatment | Δ | Δ% | Effect size | p (Bonferroni) | Verdict |",
        "|-----------|--------|---------|-----------|---|----|-------------|----------------|---------|",
    ]

    for m in result.metrics:
        sign = "+" if m.delta >= 0 else ""
        verdict_map = {
            "improvement": "✅ improvement",
            "regression":  "❌ regression",
            "neutral":     "— n.s.",
        }
        lines.append(
            f"| {m.condition} | {m.metric} "
            f"| {m.control_mean:.3f} | {m.treatment_mean:.3f} "
            f"| {sign}{m.delta:.3f} | {sign}{m.relative_pct:.1f}% "
            f"| {_effect_label(m.effect_size)} ({m.effect_size:+.2f}) "
            f"| {m.p_value_corrected:.4f} "
            f"| {verdict_map[m.direction]} |"
        )

    lines += ["", "## Significant findings"]

    improvements = [m for m in result.metrics if m.direction == "improvement"]
    regressions  = [m for m in result.metrics if m.direction == "regression"]

    if improvements:
        lines.append(f"\n### Improvements (α = {bonferroni_alpha:.4f})")
        for m in improvements:
            lines.append(
                f"- **{m.metric}** @ `{m.condition}`: "
                f"{m.control_mean:.3f} → {m.treatment_mean:.3f}  "
                f"(Δ = {m.delta:+.3f} / {m.relative_pct:+.1f}%, "
                f"{_effect_label(m.effect_size)} effect d={m.effect_size:+.2f}, "
                f"n={m.n_control}+{m.n_treatment}, "
                f"{m.test_type}, p_corr={m.p_value_corrected:.4f})"
            )

    if regressions:
        lines.append(f"\n### Regressions (α = {bonferroni_alpha:.4f})")
        for m in regressions:
            lines.append(
                f"- **{m.metric}** @ `{m.condition}`: "
                f"{m.control_mean:.3f} → {m.treatment_mean:.3f}  "
                f"(Δ = {m.delta:+.3f} / {m.relative_pct:+.1f}%, "
                f"{_effect_label(m.effect_size)} effect d={m.effect_size:+.2f}, "
                f"n={m.n_control}+{m.n_treatment}, "
                f"{m.test_type}, p_corr={m.p_value_corrected:.4f})"
            )

    if not improvements and not regressions:
        lines.append(
            "No statistically significant differences found "
            f"after Bonferroni correction (α = {bonferroni_alpha:.4f})."
        )

    return "\n".join(lines)
