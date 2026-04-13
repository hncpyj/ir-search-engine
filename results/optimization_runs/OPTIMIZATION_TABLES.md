# Optimization Results — Complete Tables

Generated: 2026-04-11 15:48

## Table 1: Baseline Per-Domain Retrieval Metrics

| Domain | nDCG@10 | MRR@10 | R@100 | R@1000 |
|--------|---------|--------|-------|--------|
| general | 0.0065 | 0.0063 | 0.0070 | 0.0070 |
| scidocs | 0.0081 | 0.0139 | 0.0207 | 0.0207 |
| science | 0.1240 | 0.1194 | 0.1567 | 0.1567 |
| finance | 0.0213 | 0.0271 | 0.0401 | 0.0401 |
| medical | 0.7313 | 0.8167 | 0.0715 | 0.0715 |
| biomedical | 0.0312 | 0.0625 | 0.0380 | 0.0380 |
| **Macro** | **0.1537** | — | — | — |

## Table 2: Routing Accuracy

| Domain | Top-1 Accuracy | N Queries | Mean Confidence |
|--------|---------------|-----------|----------------|
| general | 0.9500 | 200 | 0.9876 |
| scidocs | 0.9650 | 200 | 0.9706 |
| science | 0.9881 | 84 | 0.9819 |
| finance | 0.9950 | 199 | 0.9893 |
| medical | 0.7692 | 13 | 0.9467 |
| biomedical | 0.9780 | 91 | 0.9747 |
| **Overall** | **0.9695** | **787** | — |

Macro F1: **0.9469**

## Table 3: Retrieval Depth Tuning

| topk | Macro nDCG@10 | Macro R@100 | Macro R@1000 |
|------|---------------|-------------|-------------|
| topk_100 | 0.0861 | 0.0557 | 0.0557 |
| topk_1000 | 0.0829 | 0.0569 | 0.1219 |
| topk_200 | 0.0847 | 0.0563 | 0.0694 |
| topk_500 | 0.0830 | 0.0562 | 0.0950 |

Best: **topk_100**

## Table 4: RRF Parameter Tuning

| RRF k | Macro nDCG@10 |
|-------|---------------|
| k_100 | 0.0840 |
| k_30 | 0.0880 |
| k_60 | 0.0861 |

Best: **k_30**

## Table 5: Random vs Curated Benchmark (Test 2)

| Domain | Curated nDCG@10 | Random nDCG@10 | Delta |
|--------|----------------|----------------|-------|
| general | 0.0052 | 0.0137 | +0.0085 |
| scidocs | 0.0051 | 0.0122 | +0.0070 |
| science | 0.0939 | 0.2069 | +0.1130 |
| finance | 0.0103 | 0.0296 | +0.0193 |
| medical | 0.3755 | 0.3755 | +0.0000 |
| biomedical | 0.0262 | 0.0688 | +0.0426 |
| **Macro** | **0.0861** | **0.1178** | **-0.0317** |

Total random queries: 715

## Table 6: Expanded Subjective Evaluation (Test 3, 55 queries)

| Metric | All | Cross-Domain | Single-Domain |
|--------|-----|-------------|---------------|
| Routing Accuracy | 0.6545 | 0.6400 | 0.6667 |
| CE nDCG@10 | 0.8008 | 0.7886 | 0.8110 |
| Domain Hit Rate | 0.4182 | — | — |

## Table 7: Cross-Domain Fusion Metrics

| Metric | Value |
|--------|-------|
| domain_coverage@10 | 0.8846 |
| mean_domains_in_top10 | 1.9231 |
| expected_domain_hit@10 | 0.1923 |
| n_queries | 26 |
| cross_encoder_nDCG@10 | 0.7821 |

## Table 8a: Latency Breakdown (Without Reranking)

| Stage | Mean (ms) | P95 (ms) | % of Total |
|-------|-----------|----------|------------|
| retrieve | 387.1 | 667.9 | 95.9% |
| classify | 16.1 | 19.0 | 4.0% |
| fuse | 0.3 | 0.4 | 0.1% |
| normalize | 0.0 | 0.0 | 0.0% |
| rerank | 0.0 | 0.0 | 0.0% |
| **total** | **403.5** | **685.9** | **100.0%** |

### Per-Domain (Without Reranking)

| Domain | Classify | Retrieve | Fuse | Rerank | Total | P95 |
|--------|----------|----------|------|--------|-------|-----|
| general | 16.0 | 290.7 | 0.3 | 0.0 | 307.0 | 1024.8 |
| scidocs | 16.4 | 410.2 | 0.3 | 0.0 | 427.0 | 630.3 |
| science | 18.0 | 345.1 | 0.3 | 0.0 | 363.5 | 492.9 |
| finance | 15.5 | 499.4 | 0.3 | 0.0 | 515.3 | 824.9 |
| medical | 15.2 | 565.5 | 0.3 | 0.0 | 581.0 | 846.0 |
| biomedical | 15.3 | 211.7 | 0.3 | 0.0 | 227.3 | 296.4 |

## Table 8b: Latency Breakdown (With Reranking)

| Stage | Mean (ms) | P95 (ms) | % of Total |
|-------|-----------|----------|------------|
| rerank | 1230.0 | 1380.4 | 74.9% |
| retrieve | 395.0 | 661.7 | 24.1% |
| classify | 16.1 | 18.8 | 1.0% |
| fuse | 0.3 | 0.4 | 0.0% |
| normalize | 0.0 | 0.0 | 0.0% |
| **total** | **1641.4** | **2025.1** | **100.0%** |

### Per-Domain (With Reranking)

| Domain | Classify | Retrieve | Fuse | Rerank | Total | P95 |
|--------|----------|----------|------|--------|-------|-----|
| general | 15.5 | 277.6 | 0.3 | 912.2 | 1205.6 | 2176.2 |
| scidocs | 16.0 | 415.9 | 0.3 | 1331.5 | 1763.8 | 2041.3 |
| science | 18.1 | 352.8 | 0.3 | 1597.1 | 1968.3 | 2194.0 |
| finance | 15.4 | 521.3 | 0.3 | 1092.1 | 1629.1 | 2011.8 |
| medical | 16.6 | 588.2 | 0.3 | 1231.7 | 1836.8 | 2160.7 |
| biomedical | 15.1 | 214.0 | 0.3 | 1215.5 | 1444.9 | 1566.8 |

## Table 9: Classifier Retraining Impact

| Domain | Before | After | Delta |
|--------|--------|-------|-------|
| general | 0.9500 | 0.9500 | +0.0000 |
| scidocs | 0.9650 | 0.9550 | -0.0100 |
| science | 0.9881 | 0.9524 | -0.0357 |
| finance | 0.9950 | 0.9849 | -0.0101 |
| medical | 0.7692 | 0.9231 | +0.1539 |
| biomedical | 0.9780 | 0.9890 | +0.0110 |
| **Overall** | **0.9695** | **0.9644** | **-0.0051** |

