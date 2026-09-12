# Multi-Domain University Retrieval Engine

End-to-end retrieval system for 6 academic/professional domains.
Built for QMUL MSc Information Retrieval (ECS736P/U).

## Architecture

```
OFFLINE:  Raw docs → Parse/Clean/Chunk → Domain BERT encoder → FAISS index per domain
ONLINE:   Query → BERT Classifier → Route → Dense+BM25 Retrieval → RRF Fusion → ColBERT Rerank → Results
```

### Domains

| Domain | Dataset | Encoder |
|--------|---------|---------|
| general | microsoft/ms_marco | msmarco-bert-base-dot-v5 |
| science | BeIR/scifact | allenai/scibert_scivocab_uncased |
| finance | BeIR/fiqa | ProsusAI/finbert |
| medical | BeIR/trec-covid | emilyalsentzer/Bio_ClinicalBERT |
| legal | isaacus/echr-retrieval | nlpaueb/legal-bert-base-uncased |
| biomedical | BeIR/nfcorpus | dmis-lab/biobert-base-cased-v1.1 |

## Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt

# GPU: faiss-gpu requires CUDA 11+
# CPU-only fallback: replace faiss-gpu with faiss-cpu in requirements.txt
```

## Reproduce: Step-by-step

### Step 1 — Download & prepare corpora

```bash
# Smoke test (fast, ~50k MS MARCO passages, full BEIR datasets)
python scripts/build_corpus.py --config configs/default.yaml

# Production (full 8.8M MS MARCO + all BEIR)
python scripts/build_corpus.py --config configs/default.yaml --override configs/full.yaml

# Single domain only
python scripts/build_corpus.py --config configs/default.yaml --domain science
```

Output: `data/processed/{domain}/corpus.parquet`, `queries.parquet`, `qrels.parquet`

### Step 2 — Build FAISS indexes

```bash
# All domains (smoke)
python scripts/build_faiss.py --config configs/default.yaml

# Single domain
python scripts/build_faiss.py --config configs/default.yaml --domain science

# Full / production
python scripts/build_faiss.py --config configs/default.yaml --override configs/full.yaml
```

Output: `indexes/faiss/{domain}/faiss.index`, `docstore.parquet`, `id_mapping.json`

### Step 3 — (Optional) Build BM25 indexes

Only runs for domains with `bm25_enabled: true` (default: false in smoke config).

```bash
python scripts/build_bm25.py --config configs/full.yaml   # all bm25-enabled domains
python scripts/build_bm25.py --config configs/full.yaml --domain science
```

### Step 4 — Train domain classifier

```bash
python scripts/train_domain_classifier.py --config configs/default.yaml
# Target: >85% top-1 accuracy, >97% top-2 accuracy
```

Output: `models/domain_classifier/`

### Step 5 — Search demo

```bash
# Single query
python scripts/search.py --query "what causes alzheimer's disease"

# Medical domain query
python scripts/search.py --query "COVID-19 clinical trial outcomes ICU patients"

# Legal domain query
python scripts/search.py --query "ECHR article 6 fair trial right to counsel"

# Batch
python scripts/search.py --batch my_queries.txt --topk 10 --output results.jsonl

# Full config (GPU + ColBERT)
python scripts/search.py --query "..." --override configs/full.yaml
```

### Step 6 — Evaluate

```bash
# Per-domain evaluation (smoke)
python scripts/evaluate.py --config configs/default.yaml --domain science

# All domains
python scripts/evaluate.py --config configs/default.yaml

# With latency measurement
python scripts/evaluate.py --config configs/default.yaml --latency

# Full ablation table
python scripts/evaluate.py --config configs/full.yaml --ablation

# Save results to custom directory
python scripts/evaluate.py --config configs/default.yaml --output results/run1/
```

## Evaluation Metrics

| Domain | Metrics |
|--------|---------|
| general (MS MARCO) | MRR@10, Recall@100, Recall@1000 |
| science/finance/medical/bio (BEIR) | nDCG@10, Recall@100 |
| legal (ECHR) | nDCG@10, Recall@100 |

## Ablation Conditions

Triggered with `--ablation` flag on `evaluate.py`:

| Condition | Description |
|-----------|-------------|
| `dense_only` | Dense FAISS only, no BM25, no reranking |
| `bm25_only` | BM25 only, no dense |
| `hybrid_no_rerank` | Dense + BM25 (RRF), no ColBERT |
| `hybrid_with_rerank` | Dense + BM25 (RRF) + ColBERT |
| `routing_broadcast` | All 6 domain indexes queried |
| `routing_routed_general` | Top-1 domain + general fallback |
| `routing_routed_only` | Top-1 domain only |

## RAG Grounding Evaluation

Does the system actually *use* the passages it retrieves, or does it answer from parametric memory? To measure this, we apply causal context interventions across three domains (100 queries × 3 seeds each) and compare generated answers against the `normal` baseline using sentence-level cosine similarity (`all-MiniLM-L6-v2`).

### Intervention Conditions

| Condition | What it tests |
|-----------|---------------|
| `normal` | Baseline — top-5 retrieved passages |
| `no_retrieval` | Parametric memory — model answers with no context |
| `swapped_context` | Context sensitivity — passages from a different query (same domain) |
| `random_in_domain` | Weak swap — random passages from the same domain corpus |
| `shuffled_order` | Position bias — same passages, permuted order |
| `corrupted_context` | Noise robustness — minor token-level corruption applied |

**Grounding score** = min(conditional grounding) over content-swap conditions (`swapped_context`, `random_in_domain`), where conditional grounding = fraction of *answered* pairs where the answer changed. Abstentions are excluded from the denominator, following Wallat et al. (arXiv:2412.18004).

### Results (100 queries × 3 seeds)

| Domain | Grounding Score | Swapped Δ | Random Δ | Sentence Faithfulness |
|--------|:--------------:|:---------:|:--------:|:---------------------:|
| general | **0.875** | 98.3% | 96.0% | 0.75 |
| science | **0.905** | 84.7% | 85.0% | 0.26 |
| finance | **1.000** | 97.3% | 95.0% | 0.41 |

**Swapped / Random Δ**: fraction of query pairs where the answer changed when retrieval context was replaced with an irrelevant substitute.  
**Sentence Faithfulness**: cosine similarity between answer sentences and retrieved passages (normal condition); lower = answers diverge from source text.  
Science uses `claim_verify` prompt mode (SUPPORTED / REFUTED / NOT ENOUGH INFO), which suppresses short-form answers and reduces faithfulness scores by design.

```bash
# Run grounding evaluation (single domain)
python scripts/eval_grounding.py --config configs/default.yaml \
  --domain science --n-queries 100 --n-seeds 3 \
  --output results/grounding_v3/

# Smoke test (5 queries, 1 seed)
python scripts/eval_grounding.py --config configs/default.yaml \
  --domain general --n-queries 5 --n-seeds 1
```

Output: `{domain}_raw.parquet`, `{domain}_summary.json`, `{domain}_summary.md`

## Statistical A/B Testing

`scripts/eval_ab.py` compares two grounding evaluation runs (e.g., different prompt modes, temperatures, or retrieval strategies) across **21 metrics** spanning all intervention conditions. To control the false discovery rate across correlated tests, we apply **Benjamini-Hochberg FDR correction** (α = 0.05) rather than Bonferroni, which would be overly conservative for metrics that share the same underlying query pairs.

Effect sizes are reported as Cohen's *h* (proportions) or Cohen's *d* (continuous). Primary test: paired Welch's *t*-test; independent fallback for unmatched samples. Significant findings include direction (improvement / regression) and Wilson 95% CIs.

```bash
python scripts/eval_ab.py \
  --control   results/grounding_v3/science_raw.parquet \
  --treatment results/grounding_ab/science_raw.parquet \
  --control-name strict --treatment-name partial \
  --domain science --output results/grounding_ab/
```

Output: Markdown report + machine-readable JSON with per-metric p-values, corrected p-values, and effect sizes.

## Configuration

- `configs/default.yaml` — smoke test (50k MS MARCO, CPU, no reranking, HNSW)
- `configs/full.yaml` — production overrides (full corpus, GPU+fp16, IVF_PQ, ColBERT)

Key config keys:

```yaml
routing.mode: routed_with_general   # broadcast | routed_with_general | routed_only
routing.confidence_threshold: 0.45
reranking.enabled: false            # set true for ColBERT reranking
retrieval.topk_dense: 100           # 1000 in full config
```

## Project Structure

```
ir-search-engine/
├── configs/            YAML configs (default + full)
├── src/
│   ├── adapters/       Per-dataset data loaders (BaseAdapter + 6 implementations)
│   ├── preprocessing/  SlidingWindowChunker, QueryNormalizer
│   ├── indexing/       FAISSIndexBuilder, BM25IndexBuilder
│   ├── retrieval/      DenseRetriever, BM25Retriever, RRFFusion
│   ├── reranking/      ColBERTReranker (RAGatouille rerank-only mode)
│   ├── classification/ DomainClassifier inference + training
│   ├── pipeline/       SearchPipeline (online orchestrator)
│   ├── evaluation/     metrics.py, ablation.py
│   └── config.py       Config loader with deep-merge
├── scripts/            CLI entrypoints
├── data/processed/     Chunked parquet files per domain
├── indexes/faiss/      FAISS indexes per domain
├── indexes/bm25/       Lucene BM25 indexes per domain
├── models/             domain_classifier/
└── results/            Evaluation outputs, ablation tables
```

## ColBERT Integration

ColBERT v2 is used in **rerank-only mode (Mode B)**:
- No ColBERT index is built.
- `RAGatouille.rerank(query, documents, k)` scores fused top-K candidates.
- GPU memory: ~2GB for 200 passages × 180 tokens on RTX 3060 Ti 12GB.
- Set `reranking.topk_rerank: 50` for smoke, `200` for full.

```python
# Under the hood (colbert_reranker.py):
from ragatouille import RAGPretrainedModel
model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
results = model.rerank(query=query, documents=[...], k=50)
```

## Hardware Notes

Default (smoke) config is CPU-safe and runs on any machine.
Full config targets RTX 3060 Ti 12GB:
- Encoding: fp16, batch_size=128, ~2k passages/sec
- MS MARCO full encode: ~1.5 hours
- BEIR datasets: minutes each
- ColBERT reranking: ~200ms per query for topk=200
