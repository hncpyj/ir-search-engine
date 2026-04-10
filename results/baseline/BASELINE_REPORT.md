# Baseline Evaluation Report

**Date:** 2026-04-04  
**Config:** configs/default.yaml (CPU, dense-only, no BM25, no reranking)  
**Routing mode:** routed_with_general (confidence_threshold=0.45)  
**Classifier:** DistilBERT, 1 epoch, 14,723 training queries (val top-1 acc: 96.4%)  

---

## Infrastructure Status (All 6 Domains Rebuilt)

| Domain | Corpus | FAISS Index | Encoder | Vectors |
|--------|--------|-------------|---------|---------|
| general | Two-pass, 100% qrel | HNSW | msmarco-bert-base-dot-v5 | 50,242 |
| science | Full | Flat | scibert_scivocab (non-retrieval) | 12,047 |
| scidocs | Full | HNSW | scibert_scivocab (non-retrieval) | 46,629 |
| finance | Full | HNSW | BAAI/bge-base-en-v1.5 (retrieval) | 90,079 |
| medical | Two-pass, 100% qrel | HNSW | Bio_ClinicalBERT (non-retrieval) | 116,113 |
| biomedical | Full | Flat | BioBERT (non-retrieval) | 10,484 |

---

## Tier 1: Per-Domain Retrieval Metrics

### Full Pipeline (routed_with_general mode)

| Domain | nDCG@10 | MRR@10 | Recall@100 | Recall@1000 |
|--------|---------|--------|------------|-------------|
| science | 0.2514 | 0.2110 | 0.5589 | 0.5589 |
| finance | 0.2153 | 0.2513 | 0.6329 | 0.6329 |
| biomedical | 0.0759 | 0.1775 | 0.0864 | 0.0864 |
| medical | 0.0684 | 0.0832 | 0.0095 | 0.0095 |
| general | 0.0587 | 0.0561 | 0.0698 | 0.0698 |
| scidocs | 0.0074 | 0.0129 | 0.0325 | 0.0325 |
| **Macro avg** | **0.1129** | **0.1320** | **0.2317** | **0.2317** |

### Domain-Only Metrics (BEIR-Comparable)

| Domain | nDCG@10 | MRR@10 | Recall@100 |
|--------|---------|--------|------------|
| science | 0.2756 | 0.2418 | 0.5589 |
| finance | 0.2364 | 0.2554 | 0.6329 |
| general | 0.0613 | 0.0594 | 0.0698 |
| scidocs | 0.0023 | 0.0048 | 0.0083 |

---

## Tier 2: Resource Selection (Routing) Accuracy

**Methodology:** RouterRetriever oracle analysis (Lee et al., AAAI 2025)

| Metric | Value |
|--------|-------|
| **Top-1 accuracy** | **82.95%** |
| **Top-2 accuracy** | **90.10%** |
| **Macro F1** | **0.753** |

| Domain | Top-1 Acc | Confidence | n | Training Data |
|--------|----------|-----------|---|---------------|
| finance | 98.5% | 0.967 | 200 | 5,000 queries |
| science | 98.5% | 0.886 | 200 | 809 queries |
| biomedical | 96.5% | 0.920 | 200 | 2,914 queries |
| general | 95.0% | 0.946 | 200 | 5,000 queries |
| scidocs | 40.5% | 0.642 | 200 | 500 seed only |
| medical | 26.0% | 0.630 | 50 | 500 seed only |

---

## Tier 3: Cross-Domain Fusion Quality

**Judge:** cross-encoder/ms-marco-MiniLM-L-6-v2 | **26 queries**

| Metric | Value |
|--------|-------|
| **cross_encoder_nDCG@10** | **0.7858** |
| domain_coverage@10 | 0.5769 |
| mean_domains_in_top10 | 1.6923 |
| expected_domain_hit@10 | 0.1538 |

---

## Latency Profile (mean ms, CPU-only)

| Domain | Classify | Retrieve | Fuse | Total | P95 |
|--------|----------|----------|------|-------|-----|
| general | 146 | 297 | 0.6 | 444 | 1174 |
| science | 42 | 192 | 0.7 | 235 | 387 |
| scidocs | 39 | 175 | 0.7 | 215 | 320 |
| finance | 40 | 190 | 0.7 | 231 | 429 |
| biomedical | 34 | 155 | 0.7 | 190 | 272 |
| medical | 16 | 80 | 0.3 | 96 | 145 |
| **Average** | | | | **235** | **455** |

All domains well under 500ms mean latency gate.

---

## Key Findings

### Strengths
1. **Finance domain** benefits enormously from retrieval-trained encoder (bge-base-en-v1.5): nDCG@10=0.215, Recall@100=0.633
2. **Routing excellent for 4/6 domains**: finance, science, biomedical, general all >95% top-1 accuracy
3. **Cross-domain fusion quality** is strong (nDCG@10=0.786) even without BM25 or reranking
4. **Latency** well within acceptable bounds (mean 235ms, P95 455ms)

### Weaknesses
1. **SciDocs**: scibert encoder is not retrieval-trained -> nDCG@10=0.007 (effectively random)
2. **Medical routing**: 26% accuracy due to seed-query-only training data
3. **General domain recall**: Only 7% despite 100% qrel coverage (50k/8.8M corpus cap)
4. **No BM25 or reranking**: Hybrid fusion and ColBERT disabled

---

## Optimization Priorities

| # | Optimization | Expected Impact |
|---|-------------|-----------------|
| 1 | Replace scidocs encoder (scibert -> bge-base-en-v1.5) | nDCG: 0.007 -> ~0.15+ |
| 2 | Retrain classifier with more medical/scidocs data | Routing: 26%/40% -> ~80% |
| 3 | Enable BM25 hybrid retrieval | +0.02-0.05 nDCG per domain |
| 4 | Replace biomedical encoder (BioBERT -> bge-base-en-v1.5) | nDCG: 0.076 -> ~0.15+ |
| 5 | Tune RRF k parameter (k=30,60,100) | ~0.01 nDCG |

---

## Tracking Sheet

| Metric | Baseline | After Opt | Delta |
|--------|----------|-----------|-------|
| Macro nDCG@10 (6 domains) | 0.1129 | ___ | ___ |
| Best domain nDCG@10 | 0.2514 (science) | ___ | ___ |
| Routing Acc (top-1) | 82.95% | ___ | ___ |
| Cross-domain nDCG@10 | 0.7858 | ___ | ___ |
| Mean latency (ms) | 235 | ___ | ___ |

---

## Ablation Hierarchy

**Expected:** BM25 < Dense < Hybrid < Hybrid+Rerank

**Current state:** Only dense-only mode tested. BM25 and ColBERT reranking are disabled. The ablation hierarchy cannot be verified without enabling these components.

---

## Summary

The baseline evaluation across all 6 domains reveals a functional multi-domain IR system with strong routing for well-represented domains (finance 98.5%, science 98.5%) but significant encoder and data quality gaps. The top-performing domains use either retrieval-trained encoders (finance with bge-base-en-v1.5: nDCG=0.215) or well-matched domain encoders (science with scibert on SciFact: nDCG=0.251). The weakest domain (scidocs: nDCG=0.007) uses a non-retrieval encoder, confirming that encoder selection is the primary determinant of retrieval quality. Cross-domain fusion quality is reasonable (nDCG=0.786) suggesting the RRF mechanism works well. The most impactful next step is replacing non-retrieval encoders in scidocs and biomedical domains.
