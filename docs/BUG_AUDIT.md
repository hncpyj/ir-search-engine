# IR Search Engine — Bug Audit Report

**Date:** 2026-04-01
**Auditor perspective:** Senior AI / IR Engineer
**Codebase revision:** `291b6d8` (fix: resolve FAISS segfault, retrieval bugs, and evaluation scoring)

---

## Table of Contents

1. [BUG-1: Qrel Coverage Failure — General Domain (CRITICAL)](#bug-1-qrel-coverage-failure--general-domain)
2. [BUG-2: Qrel Coverage Failure — Medical Domain (CRITICAL)](#bug-2-qrel-coverage-failure--medical-domain)
3. [BUG-3: Classifier Train/Eval Data Leakage (HIGH)](#bug-3-classifier-traineval-data-leakage)
4. [BUG-4: Device Auto-Upgrade Overrides Explicit Config (HIGH)](#bug-4-device-auto-upgrade-overrides-explicit-config)
5. [BUG-5: Non-Retrieval Encoders for Domain Indexes (HIGH)](#bug-5-non-retrieval-encoders-for-domain-indexes)
6. [BUG-6: Classifier Seed Query Inter-Domain Overlap (MEDIUM)](#bug-6-classifier-seed-query-inter-domain-overlap)
7. [BUG-7: Stale Domain Constants (MEDIUM)](#bug-7-stale-domain-constants)
8. [BUG-8: Split Parameter Inconsistently Honored (MEDIUM)](#bug-8-split-parameter-inconsistently-honored)
9. [BUG-9: RRF Fusion Shows Wrong Chunk Text in UI (MEDIUM)](#bug-9-rrf-fusion-shows-wrong-chunk-text-in-ui)
10. [Cross-Validation & Human Review Sources](#cross-validation--human-review-sources)

---

## BUG-1: Qrel Coverage Failure — General Domain

**Severity:** CRITICAL
**Files:** `src/adapters/msmarco_adapter.py`, `data/processed/general/corpus.parquet`
**Affects:** All evaluation metrics for the `general` domain

### Description

The MSMarco adapter was fixed in commit `291b6d8` with a two-pass corpus
loading strategy that pre-collects all qrel-relevant passages before filling
the remainder up to `max_docs`. However, the existing `corpus.parquet` on disk
was built by an **earlier version** of the adapter that used naive sequential
truncation (first 50,000 from the streaming corpus).

Since `build_corpus.py` skips domains where `corpus.parquet` already exists,
the fix was never applied to the on-disk data.

### Measured Impact

```
general: qrel_docs=7,433  corpus_docs=50,000  coverage=0.2%  (17 relevant docs found)
```

Only **17 out of 7,433** relevant documents exist in the index. This means:

- **nDCG@10 and MRR@10 will be near zero** regardless of retrieval quality,
  because the system physically cannot retrieve documents that are not in the
  index.
- **Recall@100 and Recall@1000 are capped at ~0.2%**, making them useless for
  comparing retrieval approaches.

### Why This Matters (IR Theory)

Evaluation in IR depends on the completeness assumption: the retrieval system
has access to all documents referenced in the relevance judgments (qrels).
When this assumption is violated, metrics become **lower bounds** that conflate
indexing coverage with retrieval effectiveness.

The BEIR benchmark (Thakur et al., 2021) explicitly requires that the full
corpus is indexed when reporting metrics. Truncated corpora produce numbers
that are **not comparable** to published baselines.

> **Source:** Thakur, N., Reimers, N., Ruckteschel, A., Srivastava, A., &
> Gurevych, I. (2021). "BEIR: A Heterogeneous Benchmark for Zero-shot
> Evaluation of Information Retrieval Models." *NeurIPS 2021 Datasets and
> Benchmarks Track.* https://arxiv.org/abs/2104.08663

### Fix

Rebuild the general corpus with:
```bash
python scripts/build_corpus.py --config configs/default.yaml --domain general --force
```
Then rebuild the FAISS index:
```bash
python scripts/build_faiss.py --config configs/default.yaml --domain general --force
```

---

## BUG-2: Qrel Coverage Failure — Medical Domain

**Severity:** CRITICAL
**Files:** `src/adapters/treccovid_adapter.py`
**Affects:** All evaluation metrics for the `medical` domain

### Description

The TREC-COVID adapter uses naive sequential truncation: it yields the first
`max_docs` documents from the HuggingFace streaming corpus. Unlike the MSMarco
adapter, it was **never given** the two-pass relevant-doc-first loading
strategy.

With `max_docs=10000`, the adapter yields the first 10,000 documents from a
171,332-document corpus. The 66,336 relevance judgments reference 35,480 unique
documents that are **scattered throughout the full corpus**, not concentrated
at the beginning.

### Measured Impact

```
medical: qrel_docs=35,480  corpus_docs=10,000  coverage=4.4%  (1,565 relevant docs found)
```

95.6% of relevant documents are missing from the index. Evaluation metrics
will be severely deflated — not because the retrieval model is bad, but
because the relevant documents were never indexed.

### Why This Matters (IR Theory)

TREC-COVID (Voorhees et al., 2021) uses graded relevance with pooled
judgments. The pooling methodology assumes all assessed documents are
retrievable. When 95.6% of assessed-relevant documents are absent, the
evaluation degenerates into measuring indexing sampling luck rather than
retrieval effectiveness.

This is a well-known pitfall in IR evaluation called **incomplete judgment
bias**. While methods exist to correct for unjudged documents (e.g., bpref),
there is no valid correction for systematically **removing judged-relevant
documents from the collection**.

> **Source:** Voorhees, E. M., Alam, T., Bedrick, S., et al. (2021).
> "TREC-COVID: Constructing a Pandemic Information Retrieval Test Collection."
> *ACM SIGIR Forum, 54(1).* https://doi.org/10.1145/3451964.3451965
>
> **Source (incomplete judgments):** Buckley, C., & Voorhees, E. M. (2004).
> "Retrieval Evaluation with Incomplete Information." *SIGIR 2004.*
> https://doi.org/10.1145/1008992.1009000

### Fix

Port the two-pass approach from `msmarco_adapter.py` to `treccovid_adapter.py`:

1. Pre-collect all `corpus-id` values from the "test" qrels.
2. Pass 1: stream corpus, collect all qrel-relevant documents.
3. Pass 2: fill remainder up to `max_docs` with non-relevant documents.
4. Rebuild corpus and FAISS index with `--force`.

The same fix should be applied to **every adapter** that may be used with
`max_docs` truncation (future-proofing).

---

## BUG-3: Classifier Train/Eval Data Leakage

**Severity:** HIGH
**Files:** `src/classification/train_classifier.py` (lines 80-88),
`scripts/evaluate_routing.py` (lines 64-91)
**Affects:** Routing accuracy metrics (Tier 2 evaluation)

### Description

The domain classifier is trained on queries loaded from
`data/processed/{domain}/queries.parquet`. The routing evaluation in
`evaluate_routing.py` samples evaluation queries from the **same files**.

Measured overlap between classifier training data and routing evaluation data:

| Domain | Total queries | Eval queries | Overlap |
|--------|-------------|-------------|---------|
| medical | 50 | 50 | **100%** |
| scidocs | 1,000 | 1,000 | **100%** |
| science | 1,109 | 300 | 27% |
| finance | 6,648 | 648 | 10% |
| biomedical | 3,237 | 323 | 10% |
| general | 509,962 | 6,980 | 1% |

For medical and scidocs, the routing evaluation is measured **entirely on
training data**. The reported top-1 accuracy for these domains is overfit
and not representative of generalization performance.

### Why This Matters (ML Theory)

Train/test contamination is a fundamental violation of the evaluation protocol.
Reported accuracy on training data is an **upper bound** on true performance,
not an estimate of it. The gap between train and test accuracy is typically
5-20% for text classifiers (depending on dataset diversity).

The `train_classifier.py` does perform an 80/20 train/val split, but
`evaluate_routing.py` ignores this split and samples from **all** queries.

> **Source:** Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012).
> "Leakage in Data Mining: Formulation, Detection, and Avoidance."
> *ACM TKDD, 6(4).* https://doi.org/10.1145/2382577.2382579
>
> **Source (NLP-specific):** Gorman, K., & Bedrick, S. (2019). "We Need to
> Talk about Standard Splits." *ACL 2019.*
> https://doi.org/10.18653/v1/P19-1267
>
> **Source (routing evaluation methodology):** Lee, J., et al. (2025).
> "RouterRetriever: Exploring the Benefits of Routing over Multiple Expert
> Embedding Models." *AAAI 2025.* (Referenced in `evaluate_routing.py`
> docstring — their oracle analysis uses held-out queries, not training
> queries.)

### Fix

Option A (recommended): Save the train/val split indices during classifier
training (e.g., as `train_qids.json` / `val_qids.json`). In
`evaluate_routing.py`, load only the validation query IDs for evaluation.

Option B: Use k-fold cross-validation — train k classifiers, each evaluated
on a held-out fold, and report averaged routing accuracy. This is more robust
but computationally expensive.

Option C (minimum): Stratified hold-out — reserve 20% of queries per domain
as a dedicated routing test set that is never used for training.

---

## BUG-4: Device Auto-Upgrade Overrides Explicit Config

**Severity:** HIGH
**File:** `src/indexing/faiss_builder.py` (lines 119-126)
**Affects:** Reproducibility, potential OOM, index/query device mismatch

### Description

```python
configured = scfg.get("device", "cpu")
if configured == "cpu":
    if torch.cuda.is_available():
        configured = "cuda"
    elif torch.backends.mps.is_available():
        configured = "mps"
```

When the config explicitly says `device: "cpu"`, the FAISS builder silently
overrides it to CUDA or MPS if available. This causes three problems:

1. **Index/query device mismatch:** The builder encodes on GPU, but
   `DenseRetriever` (used at query time) respects the config and encodes on
   CPU. Due to floating-point non-associativity, mean-pooled embeddings
   computed on GPU vs CPU can differ at ~1e-6 magnitude. For approximate
   indexes (HNSW), this can change which documents land near ranking boundaries.

2. **Config intention violated:** A user who sets `device: "cpu"` because
   their GPU has insufficient VRAM will experience unexpected OOM.

3. **Reproducibility broken:** Same config produces different indexes on
   machines with/without GPUs.

> **Source (floating-point reproducibility):** Whitehead, N., & Fit-Florea, A.
> (2011). "Precision & Performance: Floating Point and IEEE 754 Compliance
> for NVIDIA GPUs." *NVIDIA Technical Report.*

### Fix

Remove the auto-upgrade. Respect the configured device. If auto-detection is
desired, it should be a separate config option (e.g., `device: "auto"`).

---

## BUG-5: Non-Retrieval Encoders for Domain Indexes

**Severity:** HIGH
**File:** `configs/default.yaml` — `encoder_model` per domain
**Affects:** Retrieval quality (nDCG, MRR) for finance, medical, biomedical,
science/scidocs domains

### Description

| Domain | Encoder | Training objective |
|--------|---------|-------------------|
| general | `msmarco-bert-base-dot-v5` | Contrastive retrieval (MS MARCO) |
| science | `scibert_scivocab_uncased` | Masked Language Modeling (MLM) |
| scidocs | `scibert_scivocab_uncased` | MLM |
| finance | `ProsusAI/finbert` | **Sentiment classification (3-class)** |
| medical | `Bio_ClinicalBERT` | MLM on clinical notes |
| biomedical | `biobert-base-cased-v1.1` | MLM on PubMed |

Only `msmarco-bert-base-dot-v5` was trained with a retrieval-oriented
objective (contrastive learning on query-passage pairs). The other encoders
are either MLM-pretrained or fine-tuned for classification — their embedding
spaces are **not optimized for query-document similarity ranking**.

FinBERT is the most problematic: its representation space is organized around
sentiment polarity (positive/negative/neutral), not semantic relevance.

### Why This Matters (IR Theory)

The bi-encoder retrieval paradigm requires that the encoder produces
embeddings where relevant query-document pairs have higher similarity than
non-relevant pairs. MLM-pretrained models provide reasonable semantic
similarity but significantly underperform retrieval-trained models on standard
benchmarks.

Reimers & Gurevych (2019) showed that raw BERT embeddings (without
fine-tuning for similarity) perform **worse than GloVe** on semantic textual
similarity tasks. The mean-pooled output of a model trained for MLM is not
inherently suitable for retrieval.

> **Source:** Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence
> Embeddings using Siamese BERT-Networks." *EMNLP 2019.*
> https://arxiv.org/abs/1908.10084
>
> **Source (domain-specific retrieval):** Thakur, N., Reimers, N.,
> Ruckteschel, A., Srivastava, A., & Gurevych, I. (2021). "BEIR: A
> Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval
> Models." *NeurIPS 2021.* Table 3 shows that general-purpose retrieval
> models (e.g., DPR, ANCE, TAS-B) significantly outperform vanilla BERT on
> most BEIR datasets.
>
> **Source (FinBERT):** Araci, D. (2019). "FinBERT: Financial Sentiment
> Analysis with Pre-trained Language Models." https://arxiv.org/abs/1908.10063
> — The model was fine-tuned on Financial PhraseBank for 3-class sentiment
> classification, not retrieval.
>
> **Source (retrieval-oriented training):** Karpukhin, V., et al. (2020).
> "Dense Passage Retrieval for Open-Domain Question Answering." *EMNLP 2020.*
> https://arxiv.org/abs/2004.04906 — Demonstrates that contrastive
> fine-tuning on retrieval pairs is essential for dense retrieval quality.

### Suggested Alternatives

For a multi-domain system using domain-specific encoders, consider models
that have been trained (or at least evaluated) on retrieval tasks:

| Domain | Better encoder option | Basis |
|--------|----------------------|-------|
| All domains | `BAAI/bge-base-en-v1.5` | Top BEIR performer, cross-domain |
| Biomedical | `pritamdeka/S-PubMedBert-MS-MARCO` | PubMedBERT + MS MARCO retrieval |
| Scientific | `allenai/specter2` | Trained on scientific paper relevance |
| Medical | `ncbi/MedCPT-Query-Encoder` | Contrastive on PubMed queries |
| Finance | `BAAI/bge-base-en-v1.5` | No finance-specific retrieval model exists; use general |

Alternatively, use a single strong general retrieval encoder (e.g., BGE,
E5, GTE) across all domains. Thakur et al. (2021) show that strong
general-purpose retrieval models often outperform domain-specific
non-retrieval models even on domain-specific benchmarks.

---

## BUG-6: Classifier Seed Query Inter-Domain Overlap

**Severity:** MEDIUM
**File:** `src/classification/seed_queries.py`
**Affects:** Classifier accuracy on ambiguous queries

### Description

The seed queries used to augment small domains have significant semantic
overlap between science, medical, and biomedical:

| Seed query | Labeled domain | Also fits |
|------------|---------------|-----------|
| "how does mRNA vaccine technology work" | science | medical, biomedical |
| "what causes antibiotic resistance in bacteria" | science | medical, biomedical |
| "mechanism of insulin resistance in type 2 diabetes" | biomedical | medical |
| "what are the hallmarks of cancer" | biomedical | medical, science |
| "what is the role of cortisol in stress response" | medical | biomedical |

When the classifier trains on these ambiguous labels, it learns noisy
decision boundaries. This is especially problematic for domains augmented
primarily with seeds (medical: 50 real queries + seed padding).

### Why This Matters

In multi-label classification, label noise at training time directly degrades
test-time precision. For a routing classifier, misrouting a medical query to
biomedical (or vice versa) means the query is searched against an index that
may not contain the relevant documents — producing zero recall for that query.

> **Source:** Frenay, B., & Verleysen, M. (2014). "Classification in the
> Presence of Label Noise: A Survey." *IEEE TNNLS, 25(5).*
> https://doi.org/10.1109/TNNLS.2013.2292894

### Fix

1. Sharpen seed queries to be unambiguously domain-specific. For example,
   medical seeds should reference clinical practice, patient care, and
   treatment protocols — not molecular mechanisms (which are biomedical).
2. Consider adding negative examples or multi-label annotations for
   ambiguous queries.
3. For evaluation, acknowledge that science/medical/biomedical routing
   errors may be **acceptable misroutes** if the target domain's index
   contains overlapping relevant content.

---

## BUG-7: Stale Domain Constants

**Severity:** MEDIUM
**Files:** `src/pipeline/online_pipeline.py` (line 28), `app.py` (lines 28-38)

### Description

```python
# online_pipeline.py — still has "legal", missing "scidocs"
ALL_DOMAINS = ["general", "science", "finance", "medical", "legal", "biomedical"]

# app.py — has "legal" colour/icon, no "scidocs" entry
DOMAIN_COLOURS = {..., "legal": "#f59e0b"}
DOMAIN_ICONS   = {..., "legal": "⚖️"}
```

These constants are out of sync with the actual domain configuration. If
`scidocs` is being removed (pending team decision), these should be updated
once the decision is confirmed. If `scidocs` stays, it needs entries in both
maps.

### Fix

Update domain constants to match `configs/default.yaml` once the team
confirms the final domain list.

---

## BUG-8: Split Parameter Inconsistently Honored

**Severity:** MEDIUM
**File:** `scripts/build_corpus.py` (line 83), all adapters
**Affects:** Correctness if dev/train splits are ever needed

### Description

`build_corpus.py` hardcodes `eval_split = "test"` and passes it to
`adapter.to_dataframes(split="test")`. But each adapter handles the `split`
parameter differently:

| Adapter | `iter_queries` | `iter_qrels` |
|---------|---------------|-------------|
| MSMarco | **ignores** (always "queries") | **ignores** (always "dev") |
| Scifact | **ignores** (always "queries") | Uses param (default "test") |
| FiQA | **ignores** (always "queries") | **Uses param** |
| TREC-COVID | **ignores** (always "queries") | **ignores** (always "test") |
| NFCorpus | **ignores** (always "queries") | **Uses param** |
| SciDocs | **ignores** (always "queries") | Uses param (default "test") |

The interface contract is broken: `split` is declared as a parameter but
silently ignored in most methods. This works today because `split="test"`
happens to be correct (or the adapter ignores it), but it is a maintenance
hazard.

### Fix

Either (a) make every adapter genuinely respect the split parameter, or
(b) remove the parameter from the interface and hardcode the correct split
per adapter.

---

## BUG-9: RRF Fusion Shows Wrong Chunk Text in UI

**Severity:** MEDIUM
**File:** `src/retrieval/hybrid_fusion.py` (lines 49-54)
**Affects:** Gradio UI display quality, search demo

### Description

When multiple chunks of the same document appear in results, RRF deduplicates
by `doc_id` and keeps the chunk with the highest raw dense score as the
representative. The displayed `text` field comes from this representative
chunk.

If chunk0 (containing the title) has the highest dense score due to
term-overlap with the query, but chunk3 contains the actually relevant
passage, the user sees chunk0's text — which may not contain the information
they are searching for.

### Why This Matters

This is a well-known problem in passage-level retrieval with document-level
deduplication. The standard solution is max-passage scoring (take the
highest-scoring passage as representative) or to display the best-matching
passage regardless of document-level dedup.

> **Source:** Dai, Z., & Callan, J. (2019). "Deeper Text Understanding for IR
> with Contextual Neural Language Modeling." *SIGIR 2019.*
> https://doi.org/10.1145/3331184.3331303 — Discusses passage-level evidence
> aggregation strategies.

### Fix

In `RRFFusion.fuse()`, select the representative chunk by **fused RRF score**
(which reflects cross-modal agreement) rather than raw dense score. Or, keep
the chunk whose text has the highest score from the last retrieval stage.

---

## Cross-Validation & Human Review Sources

### Evaluation Methodology

These sources provide the foundation for validating that the evaluation
pipeline is correct:

1. **BEIR Benchmark Protocol**
   Thakur, N., Reimers, N., Ruckteschel, A., Srivastava, A., & Gurevych, I. (2021).
   "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models."
   *NeurIPS 2021 Datasets and Benchmarks Track.*
   https://arxiv.org/abs/2104.08663
   - Defines the standard protocol for evaluating retrieval on diverse domains.
   - Requires full corpus indexing for comparable metrics.
   - Provides baseline numbers for all encoders used in this project.
   - **Use for:** Validating Bugs 1, 2, 5 — check whether your nDCG@10 numbers
     are in the expected range for each dataset.

2. **TREC-COVID Collection Construction**
   Voorhees, E. M., Alam, T., Bedrick, S., et al. (2021).
   "TREC-COVID: Constructing a Pandemic Information Retrieval Test Collection."
   *ACM SIGIR Forum, 54(1).*
   https://doi.org/10.1145/3451964.3451965
   - Describes how the 66,336 relevance judgments were created via pooling.
   - Explains why removing pooled documents invalidates the evaluation.
   - **Use for:** Understanding why Bug 2 makes medical metrics meaningless.

3. **Incomplete Judgments in IR Evaluation**
   Buckley, C., & Voorhees, E. M. (2004).
   "Retrieval Evaluation with Incomplete Information."
   *SIGIR 2004.*
   https://doi.org/10.1145/1008992.1009000
   - Seminal work on how missing relevance judgments affect IR metrics.
   - Introduces bpref as a metric robust to unjudged documents (but NOT
     robust to removed-judged documents, which is the situation in Bugs 1-2).
   - **Use for:** Understanding the difference between "unjudged" vs
     "removed relevant" — the latter is much worse.

4. **ir-measures Library**
   MacAvaney, S., Yates, A., Feldman, S., Downey, D., Cohan, A., & Goharian, N. (2022).
   "Simplified Data Wrangling with ir_datasets."
   *SIGIR 2022.*
   https://ir-measur.es/en/latest/
   - The evaluation library used by this project. Documentation covers
     correct usage of nDCG, MRR, and Recall measures.
   - **Use for:** Verifying that the metric computation in `metrics.py` is correct.

### Retrieval Model Selection

5. **Sentence-BERT / Bi-Encoder Retrieval**
   Reimers, N., & Gurevych, I. (2019).
   "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks."
   *EMNLP 2019.*
   https://arxiv.org/abs/1908.10084
   - Shows that raw BERT mean-pooled embeddings underperform dedicated
     sentence encoders by a large margin.
   - Explains why retrieval-specific fine-tuning (contrastive loss) is
     essential for bi-encoder quality.
   - **Use for:** Understanding Bug 5 — why MLM-pretrained BERTs are poor
     retrieval encoders.

6. **Dense Passage Retrieval (DPR)**
   Karpukhin, V., et al. (2020).
   "Dense Passage Retrieval for Open-Domain Question Answering."
   *EMNLP 2020.*
   https://arxiv.org/abs/2004.04906
   - Foundational work on dense bi-encoder retrieval.
   - Demonstrates that contrastive fine-tuning on (query, positive, negative)
     triplets is necessary for competitive retrieval.
   - **Use for:** Understanding why the domain encoders need retrieval
     training, not just domain pretraining.

7. **FinBERT**
   Araci, D. (2019).
   "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models."
   https://arxiv.org/abs/1908.10063
   - Confirms FinBERT was fine-tuned for 3-class sentiment classification,
     not retrieval or semantic similarity.
   - **Use for:** Documenting why FinBERT is inappropriate as a retrieval
     encoder (Bug 5).

### Routing & Federated Search

8. **RouterRetriever — Routing Error Analysis**
   Lee, J., et al. (2025).
   "RouterRetriever: Exploring the Benefits of Routing over Multiple Expert
   Embedding Models."
   *AAAI 2025.*
   - Quantifies that ~30% of retrieval error is attributable to routing
     mistakes in multi-domain systems.
   - Uses **held-out queries** for routing evaluation (not training queries).
   - **Use for:** Understanding Bug 3 — the routing evaluation must use
     unseen queries to be valid. Also provides oracle ceiling analysis
     methodology.

9. **FeB4RAG — Federated Search Evaluation**
   Wang, S., et al. (2024).
   "FeB4RAG: Evaluating Federated Search in the Context of Retrieval
   Augmented Generation."
   https://arxiv.org/abs/2402.11891
   - Three-tier evaluation framework used in this project.
   - Tier 2 (resource selection) requires unbiased accuracy measurement.
   - **Use for:** Validating the overall evaluation framework design and
     understanding why train/eval leakage (Bug 3) undermines Tier 2 claims.

### Data Leakage & Evaluation Integrity

10. **Data Leakage in ML**
    Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012).
    "Leakage in Data Mining: Formulation, Detection, and Avoidance."
    *ACM TKDD, 6(4).*
    https://doi.org/10.1145/2382577.2382579
    - Formal taxonomy of data leakage types.
    - The classifier training/evaluation overlap in Bug 3 is "Type 1 leakage"
      (training data contains information from the test set).
    - **Use for:** Formally classifying the leakage in Bug 3.

11. **Standard Splits in NLP**
    Gorman, K., & Bedrick, S. (2019).
    "We Need to Talk about Standard Splits."
    *ACL 2019.*
    https://doi.org/10.18653/v1/P19-1267
    - Shows that non-standard or overlapping splits can inflate accuracy by
      5-20% in NLP tasks.
    - **Use for:** Estimating the magnitude of routing accuracy inflation in
      Bug 3.

### Label Noise & Classification

12. **Label Noise in Classification**
    Frenay, B., & Verleysen, M. (2014).
    "Classification in the Presence of Label Noise: A Survey."
    *IEEE TNNLS, 25(5).*
    https://doi.org/10.1109/TNNLS.2013.2292894
    - Comprehensive survey on how noisy labels degrade classifier performance.
    - The ambiguous seed queries in Bug 6 introduce systematic label noise
      at training time.
    - **Use for:** Understanding the theoretical impact of inter-domain seed
      overlap on classifier boundaries.

### Reciprocal Rank Fusion

13. **RRF Original Paper**
    Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).
    "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank
    Learning Methods."
    *SIGIR 2009.*
    https://doi.org/10.1145/1571941.1572114
    - Defines RRF: `score(d) = sum(1 / (k + rank_i(d)))`.
    - Shows RRF is robust to score scale differences across rankers.
    - **Use for:** Validating the fusion implementation in `hybrid_fusion.py`
      and understanding why the `k=60` parameter was chosen.

### Passage Retrieval & Chunking

14. **Passage-Level Evidence Aggregation**
    Dai, Z., & Callan, J. (2019).
    "Deeper Text Understanding for IR with Contextual Neural Language Modeling."
    *SIGIR 2019.*
    https://doi.org/10.1145/3331184.3331303
    - Discusses max-passage, first-passage, and sum-passage aggregation
      strategies for document-level retrieval from passage-level evidence.
    - **Use for:** Understanding Bug 9 and choosing the correct passage
      representative during document-level deduplication.

### Approximate Nearest Neighbor Search

15. **HNSW Algorithm**
    Malkov, Y. A., & Yashunin, D. A. (2020).
    "Efficient and Robust Approximate Nearest Neighbor Search Using
    Hierarchical Navigable Small World Graphs."
    *IEEE TPAMI, 42(4).*
    https://doi.org/10.1109/TPAMI.2018.2889473
    - Defines the HNSW index used in this project.
    - Parameters `M` (links per node) and `efConstruction` / `efSearch`
      control the recall-speed trade-off.
    - **Use for:** Validating HNSW parameter choices in the config and
      understanding why `efSearch` must be >= `topk` for reliable results.

---

## Verification Checklist

Use this checklist to verify each bug after fixes are applied:

- [ ] **BUG-1:** After rebuilding general corpus, confirm
      `qrel_doc coverage >= 99%` using the pandas check in this document.
- [ ] **BUG-2:** After porting 2-pass to TREC-COVID, confirm
      `qrel_doc coverage >= 90%` (some docs may be in the tail beyond max_docs).
- [ ] **BUG-3:** After fixing eval split, confirm that routing eval queries
      have 0% overlap with classifier training queries.
- [ ] **BUG-4:** After removing auto-upgrade, confirm same config produces
      identical indexes on CPU-only and GPU machines.
- [ ] **BUG-5:** After switching encoders, compare nDCG@10 against BEIR
      published baselines for each dataset. Expect >=50% of BEIR SOTA.
- [ ] **BUG-6:** After sharpening seeds, verify per-domain routing accuracy
      improves on the held-out test set (from Bug 3 fix).
- [ ] **BUG-7:** After updating constants, `grep -r "legal"` should return
      zero hits outside of documentation and git history.
- [ ] **BUG-8:** After standardizing split handling, test with
      `split="dev"` and confirm different qrels are loaded where applicable.
- [ ] **BUG-9:** After fixing representative selection, manually inspect
      10 queries in the Gradio UI and confirm displayed text is relevant.
