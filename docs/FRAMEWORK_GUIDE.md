# Multi-Domain IR Search Engine — Framework Guide

**Date:** 2026-04-01
**Codebase revision:** `291b6d8`

---

## Table of Contents

1. [What Is This Framework?](#1-what-is-this-framework)
2. [Core IR Concepts](#2-core-ir-concepts)
3. [Architecture Overview](#3-architecture-overview)
4. [Two-Phase Design](#4-two-phase-design)
5. [Layer 1: Configuration](#5-layer-1-configuration)
6. [Layer 2: Data Adapters](#6-layer-2-data-adapters)
7. [Layer 3: Preprocessing](#7-layer-3-preprocessing)
8. [Layer 4: Indexing](#8-layer-4-indexing)
9. [Layer 5: Retrieval](#9-layer-5-retrieval)
10. [Layer 6: Classification & Routing](#10-layer-6-classification--routing)
11. [Layer 7: Reranking](#11-layer-7-reranking)
12. [Layer 8: The Online Pipeline](#12-layer-8-the-online-pipeline)
13. [Layer 9: Evaluation](#13-layer-9-evaluation)
14. [Layer 10: Scripts (CLI Entry Points)](#14-layer-10-scripts)
15. [Layer 11: Web UI](#15-layer-11-web-ui)
16. [Data Flow Summary](#16-data-flow-summary)
17. [Per-Domain Encoder Models](#17-per-domain-encoder-models)
18. [Directory Structure](#18-directory-structure)
19. [Key Design Decisions & Trade-offs](#19-key-design-decisions--trade-offs)
20. [Glossary](#20-glossary)

---

## 1. What Is This Framework?

This is a **multi-domain Information Retrieval (IR) search engine** that
searches across **6 specialized domains** simultaneously. Instead of one
monolithic index covering all topics, it maintains **separate indexes per
domain**, each powered by a domain-specific language model. A **learned
router** (domain classifier) decides which domains to search for each query.

Think of it like having 6 specialist librarians (one for science, one for
finance, one for medicine, etc.) and a receptionist who listens to your
question and directs you to the right librarian(s).

### The 6 Domains

| Domain | Dataset | Description | Corpus Size |
|--------|---------|-------------|-------------|
| `general` | MS MARCO | Web search passages | ~8.8M (capped to 50k in smoke test) |
| `scidocs` | SciDocs | Scientific paper citation matching | ~25k docs |
| `science` | SciFact | Scientific claim verification | ~5k docs |
| `finance` | FiQA | Financial question answering | ~57k docs |
| `medical` | TREC-COVID | COVID-19 biomedical literature | ~171k (capped to 10k) |
| `biomedical` | NFCorpus | Biomedical nutrition/health | ~3.6k docs |

> **Note:** The team is considering reducing to 5 domains by removing
> `scidocs`. Check `configs/default.yaml` for the current domain list.

---

## 2. Core IR Concepts

Before diving into the code, here are the fundamental IR concepts this
framework implements:

### 2.1 Bi-Encoder Dense Retrieval

Traditional search (BM25) matches keywords. Dense retrieval uses neural
networks to encode both queries and documents into fixed-size vectors
(embeddings) in a shared space. Documents whose embeddings are close to
the query embedding are considered relevant.

```
Query: "how does mRNA work?"
    → Encoder → [0.12, -0.45, 0.78, ...]  (768-dimensional vector)

Document: "mRNA vaccines deliver genetic instructions..."
    → Encoder → [0.15, -0.42, 0.81, ...]  (768-dimensional vector)

Similarity = cosine(query_vec, doc_vec) = 0.97  → HIGH relevance
```

The encoder is a BERT-based model (Transformer). Both query and document pass
through the **same model** (or the same architecture), which is why it's
called a **bi-encoder** — two encoding passes, one model.

**Key property:** All embeddings are L2-normalized (unit vectors), so inner
product equals cosine similarity. This is why the code uses `IndexFlatIP`
(Inner Product) in FAISS.

### 2.2 FAISS (Facebook AI Similarity Search)

FAISS is a library for efficient similarity search over large collections of
vectors. Instead of comparing a query vector against every document vector
(O(n) brute force), FAISS uses specialized data structures:

- **Flat** — Exact brute-force search. Best for tiny corpora (<10k).
- **HNSW** (Hierarchical Navigable Small World) — Graph-based approximate
  search. No training needed, good up to ~1M vectors. Like a "skip list"
  in vector space.
- **IVF_PQ** (Inverted File + Product Quantization) — Partitions vectors
  into clusters, compresses with PQ. Needs training. Good for >500k vectors.

### 2.3 BM25 (Best Matching 25)

A classical term-frequency-based ranking function. Unlike dense retrieval,
BM25 works on exact keyword matching:

```
score(q, d) = Σ IDF(qi) · (tf(qi, d) · (k1 + 1)) / (tf(qi, d) + k1 · (1 - b + b · |d|/avgdl))
```

Where:
- `tf(qi, d)` = how many times query term qi appears in document d
- `IDF(qi)` = inverse document frequency (rare terms score higher)
- `k1`, `b` = tuning parameters
- `|d|/avgdl` = document length normalization

BM25 excels at exact term matching but misses synonyms and paraphrases.
Dense retrieval captures semantic meaning but can miss specific keywords.
**Hybrid retrieval** (dense + BM25) combines both strengths.

### 2.4 Reciprocal Rank Fusion (RRF)

When you have multiple ranked lists (e.g., one from dense retrieval, one
from BM25, or from different domain indexes), you need to merge them.
RRF is a simple, robust method:

```
score_RRF(doc) = Σ  1 / (k + rank_in_list_i)
                i
```

Where `k=60` is a smoothing constant. RRF is **rank-based** (not score-based),
so it doesn't care whether one ranker produces scores in [0,1] and another in
[0,100] — only the relative ordering matters.

### 2.5 Relevance Judgments (QRels)

To evaluate a retrieval system, you need ground truth: for each test query,
which documents are relevant? These are called **qrels** (query relevance
judgments). They come from benchmark datasets where human annotators (or
pooling strategies) labeled documents as relevant/non-relevant/partially-relevant.

A **qrel** entry: `(query_id, doc_id, relevance_score)`.

### 2.6 Evaluation Metrics

- **nDCG@10** (Normalized Discounted Cumulative Gain at rank 10): Measures
  ranking quality considering graded relevance. A perfect ranking of the top
  10 results scores 1.0. Penalizes relevant documents that appear lower in
  the ranking.
- **MRR@10** (Mean Reciprocal Rank at rank 10): For each query, the score is
  1/rank of the first relevant result. Averaged over all queries.
- **Recall@100 / @1000**: Fraction of all relevant documents that appear in
  the top 100 or 1000 results. Measures completeness.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          OFFLINE PHASE (build once)                     │
│                                                                         │
│  HuggingFace        Adapters        Chunker        FAISS Builder       │
│  Datasets ──────► base_adapter ──► SlidingWindow ──► FAISSIndexBuilder │
│                   msmarco_adapter   Chunker          (encode + index)   │
│                   scifact_adapter                                       │
│                   fiqa_adapter      ┌─────────────► BM25IndexBuilder   │
│                   treccovid_adapter │  (optional)                       │
│                   nfcorpus_adapter  │                                   │
│                   scidocs_adapter   │  queries.parquet                  │
│                                     │       │                           │
│                                     │       ▼                           │
│                                     │  train_classifier                 │
│                                     │       │                           │
│                                     │       ▼                           │
│                                     │  DistilBERT 6-class model        │
│                                     │  (models/domain_classifier/)      │
│                                                                         │
│  Artifacts:                                                             │
│    data/processed/{domain}/corpus.parquet, queries.parquet, qrels.parquet│
│    indexes/faiss/{domain}/faiss.index, docstore.parquet, id_mapping.json│
│    models/domain_classifier/pytorch_model.bin, label_map.json           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          ONLINE PHASE (per query)                       │
│                                                                         │
│  User Query                                                             │
│      │                                                                  │
│      ▼                                                                  │
│  QueryNormalizer ──► DomainClassifier ──► Routing Decision              │
│                           │                    │                        │
│                           │         ┌──────────┼──────────┐             │
│                           │         ▼          ▼          ▼             │
│                           │    DenseRetriever DenseRetriever ...        │
│                           │    (domain A)     (domain B)                │
│                           │         │          │                        │
│                           │    [+ BM25Retriever if enabled]             │
│                           │         │          │                        │
│                           │         ▼          ▼                        │
│                           │    Within-domain RRF fusion (per domain)    │
│                           │              │                              │
│                           │              ▼                              │
│                           │    Cross-domain RRF fusion                  │
│                           │              │                              │
│                           │              ▼                              │
│                           │    [ColBERTReranker if enabled]             │
│                           │              │                              │
│                           │              ▼                              │
│                           │    Final ranked results                     │
│                           │    + classification info                    │
│                           │    + per-stage latency                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Two-Phase Design

### Offline Phase (build once, run by `run_all.py`)

| Step | Script | What it does | Output |
|------|--------|-------------|--------|
| 1/5 | `scripts/build_corpus.py` | Load datasets via adapters, chunk documents | `data/processed/{domain}/*.parquet` |
| 2/5 | `scripts/build_faiss.py` | Encode chunks, build FAISS indexes | `indexes/faiss/{domain}/*` |
| 3/5 | `scripts/build_bm25.py` | Build BM25 lexical indexes (optional) | `indexes/bm25/{domain}/bm25.pkl` |
| 4/5 | `scripts/train_domain_classifier.py` | Train DistilBERT query router | `models/domain_classifier/*` |
| 5/5 | `scripts/evaluate.py` | Run evaluation metrics | `results/*.json` |

### Online Phase (per query, run by `SearchPipeline` or `app.py`)

```
raw_query → normalize → classify → route → retrieve → fuse → [rerank] → results
```

Each stage is timed independently (latency tracked in milliseconds).

---

## 5. Layer 1: Configuration

### `src/config.py`

| Function | Purpose |
|----------|---------|
| `load_config(path, override_path)` | Loads a base YAML config, optionally deep-merges an override on top |
| `_deep_merge(base, override)` | Recursive dict merge — override values replace base values; nested dicts are merged recursively |

### Config Files

| File | Purpose |
|------|---------|
| `configs/default.yaml` | **Smoke test** — small subsets, CPU-safe, fast iteration |
| `configs/full.yaml` | **Production** — full datasets, GPU, BM25 + ColBERT reranking enabled |
| `configs/classifier_balanced.yaml` | Alternative classifier training config |

### Key Config Sections

```yaml
system:          # device (cpu/cuda), fp16, random seed
domains:         # per-domain: dataset, max_docs, encoder_model, FAISS params
chunking:        # max_tokens, stride for sliding window
retrieval:       # topk_dense, topk_bm25, k_fusion, rrf_k
routing:         # mode (broadcast/routed_with_general/routed_only), threshold
reranking:       # enabled, ColBERT model, topk_rerank
classification:  # model_path, base_model, labels, max_length
paths:           # data_root, index_root, model_root, results_root
evaluation:      # metrics list, latency trial count
```

### How Overrides Work

`full.yaml` only specifies values that differ from `default.yaml`. Deep merge
means you can override a single nested field without repeating the entire
section:

```yaml
# default.yaml                    # full.yaml (override)
domains:                           domains:
  general:                           general:
    max_docs: 50000                    max_docs: null      # ← overrides to "all"
    batch_size: 32                     batch_size: 128     # ← overrides
    faiss_type: "HNSW"                 faiss_type: "IVF_PQ" # ← overrides
    encoder_model: "msmarco-..."       # ← kept from default (not in override)
```

---

## 6. Layer 2: Data Adapters (`src/adapters/`)

The adapters are the **data loading layer** — one adapter per dataset. They
all follow the same abstract interface, converting each dataset's unique
schema into a unified data model.

### 6.1 Data Model (dataclasses in `base_adapter.py`)

#### `Document`
```python
@dataclass
class Document:
    doc_id: str      # "{domain}:{original_id}" — e.g. "science:4983"
    title: str       # document title (may be empty)
    text: str        # document body text
    domain: str      # "general", "science", "finance", etc.
    orig_id: str     # original ID from the dataset (without domain prefix)
```

All IDs are **namespaced** with the domain prefix to avoid collisions when
merging results across domains. `"science:4983"` and `"general:4983"` are
different documents.

#### `Query`
```python
@dataclass
class Query:
    query_id: str    # "{domain}:{original_qid}"
    text: str        # the query string
    domain: str      # which domain this query belongs to
```

#### `QRel` (Query Relevance Judgment)
```python
@dataclass
class QRel:
    query_id: str    # "{domain}:{original_qid}"
    doc_id: str      # "{domain}:{original_doc_id}"
    relevance: int   # 0 = not relevant, 1 = relevant, 2 = highly relevant
```

QRels are the ground truth labels used for evaluation. They come from
benchmark datasets where human annotators labeled documents.

### 6.2 `BaseAdapter` (Abstract Base Class)

```python
class BaseAdapter(ABC):
    def __init__(self, domain: str, config: dict):
        self.domain = domain
        self.max_docs = config.get("max_docs", None)

    @abstractmethod
    def iter_documents(self) -> Iterator[Document]: ...

    @abstractmethod
    def iter_queries(self, split: str = "test") -> Iterator[Query]: ...

    @abstractmethod
    def iter_qrels(self, split: str = "test") -> Iterator[QRel]: ...

    def to_dataframes(self, split) -> tuple[DataFrame, DataFrame, DataFrame]:
        """Materializes all three iterators into Pandas DataFrames."""
```

**Why iterators?** Datasets like MS MARCO have 8.8M passages. Loading
everything into memory at once would OOM. Iterators + HuggingFace streaming
(`streaming=True`) process data one row at a time.

**Why `to_dataframes()`?** After iterating, the build scripts need DataFrames
for Parquet serialization and downstream processing. This method converts
the lazy iterators into materialized tables.

### 6.3 Concrete Adapters

Each adapter maps its dataset's specific schema (column names, config names,
splits) into the unified `Document`/`Query`/`QRel` format.

| Adapter Class | File | Domain | Dataset | Notable Details |
|---------------|------|--------|---------|----------------|
| `MSMarcoAdapter` | `msmarco_adapter.py` | `general` | `mteb/msmarco` | **Two-pass loading:** pre-collects qrel-relevant passages, then fills remainder. This ensures evaluation metrics are valid even with `max_docs` truncation. Qrels in `split="dev"` only. |
| `ScidocsAdapter` | `scidocs_adapter.py` | `scidocs` | `mteb/scidocs` | Citation matching task. 25k docs, 1k queries. |
| `ScifactAdapter` | `scifact_adapter.py` | `science` | `mteb/scifact` | Claim verification. Tiny corpus (~5k docs). Qrels in `split="test"`. |
| `FiqaAdapter` | `fiqa_adapter.py` | `finance` | `mteb/fiqa` | Financial QA. 57k docs. Qrels support multiple splits (`train`/`dev`/`test`). |
| `TrecCovidAdapter` | `treccovid_adapter.py` | `medical` | `mteb/trec-covid` | 171k docs, 50 queries, 66k graded qrels. Naive truncation (see BUG_AUDIT.md). |
| `NfcorpusAdapter` | `nfcorpus_adapter.py` | `biomedical` | `BeIR/nfcorpus` | Nutrition/health. Tiny corpus (~3.6k docs). Uses `BeIR/nfcorpus-qrels` for qrels. |

### 6.4 How Adapters Are Used

```python
# In build_corpus.py:
ADAPTER_MAP = {
    "general": MSMarcoAdapter,
    "scidocs": ScidocsAdapter,
    "science": ScifactAdapter,
    # ...
}

AdapterClass = ADAPTER_MAP[domain]
adapter = AdapterClass(domain_cfg)
corpus_df, queries_df, qrels_df = adapter.to_dataframes(split="test")
```

---

## 7. Layer 3: Preprocessing (`src/preprocessing/`)

### 7.1 `SlidingWindowChunker` (`chunker.py`)

Long documents must be split into smaller pieces for embedding. Transformer
models have a maximum input length (typically 512 tokens), and shorter chunks
produce more focused embeddings.

#### How It Works

1. **Prepend title** to body text: `"Title: xyz. Body text here..."`
2. **Tokenize** the full text using the domain's tokenizer
3. **Slide a window** of `max_tokens=180` tokens with `stride=40` overlap
4. **Decode** each token window back to text

```
Original: [tok0, tok1, tok2, ..., tok499]  (500 tokens)

Chunk 0: [tok0   ... tok179]   (tokens 0-179)
Chunk 1: [tok140 ... tok319]   (tokens 140-319, 40-token overlap with chunk 0)
Chunk 2: [tok280 ... tok459]   (tokens 280-459, 40-token overlap with chunk 1)
Chunk 3: [tok420 ... tok499]   (tokens 420-499, final partial chunk)
```

#### Why Overlap?

Without overlap, a relevant passage that spans the boundary between two
chunks would be split in half, and neither chunk would capture the full
meaning. The 40-token overlap ensures continuity.

#### `Chunk` Dataclass

```python
@dataclass
class Chunk:
    chunk_id: str        # "{doc_id}__chunk{i}" — e.g. "science:4983__chunk0"
    doc_id: str          # parent document ID
    text: str            # decoded text for this window
    start_token: int     # position in original token sequence
    end_token: int       # position in original token sequence
```

#### Key Methods

| Method | Input | Output | Notes |
|--------|-------|--------|-------|
| `chunk_text(doc_id, text)` | doc_id + raw text | Iterator of `Chunk` | Core logic. Short docs (<=180 tokens) yield one chunk. |
| `chunk_document(doc)` | dict with doc_id, title, text, etc. | Iterator of dicts | Prepends title, calls `chunk_text`, adds metadata fields. |
| `chunk_documents(documents)` | list of doc dicts | Iterator of dicts | Batch convenience — iterates `chunk_document` over list. |

### 7.2 `QueryNormalizer` (`normalizer.py`)

Lightweight, deterministic preprocessing for query strings. Applied to every
query before classification and retrieval.

| Step | What it does | Example |
|------|-------------|---------|
| Unicode NFKC normalization | Normalizes accents, full-width chars | `"ﬁnance"` → `"finance"` |
| Lowercasing | Case-insensitive matching | `"CRISPR"` → `"crispr"` |
| Control char removal | Strips invisible characters | `"hello\x00world"` → `"hello world"` |
| Whitespace collapsing | Multiple spaces → single | `"hello   world"` → `"hello world"` |

**What it does NOT do:** Stemming, stopword removal, lemmatization. These
are deliberate omissions — the BERT-based encoders handle semantic
understanding internally. Aggressive text normalization would destroy
information the models need.

---

## 8. Layer 4: Indexing (`src/indexing/`)

### 8.1 `FAISSIndexBuilder` (`faiss_builder.py`)

The central class of the offline phase. For each domain, it:

1. **Loads** the chunked `corpus.parquet`
2. **Encodes** all chunk texts into dense vectors
3. **Builds** a FAISS index
4. **Saves** artifacts to disk

#### Encoding: `encode_texts()` (module-level function)

```python
def encode_texts(texts, model_name, device, batch_size, fp16, max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    # ... batch loop ...
    emb = _mean_pool(out.last_hidden_state, enc["attention_mask"])
    emb = F.normalize(emb, p=2, dim=-1)  # L2 normalize
    return np.vstack(all_embs).astype(np.float32)
```

**Mean pooling:** Each token in the input produces a hidden state vector.
Mean pooling averages all token vectors (weighted by the attention mask, which
excludes padding tokens) into a single fixed-size vector representing the
entire text.

```
Input tokens: ["how", "does", "mRNA", "work", "?", "[PAD]", "[PAD]"]
Hidden states: [v1,    v2,    v3,     v4,    v5,   v_pad,   v_pad]
Attention mask: [1,     1,     1,      1,     1,    0,       0    ]

Mean pool = (v1 + v2 + v3 + v4 + v5) / 5   ← PAD tokens excluded
```

**L2 normalization:** After mean pooling, each vector is normalized to unit
length (||v|| = 1). This converts inner product search into cosine similarity:

```
cosine(a, b) = dot(a, b) / (||a|| · ||b||)
```

When both a and b are unit vectors, `||a|| = ||b|| = 1`, so `cosine = dot`.

#### Index Construction: `build_index()`

```python
def build_index(self, embeddings):
    if ft == "Flat":
        index = faiss.IndexFlatIP(D)           # Exact search
    elif ft == "HNSW":
        index = faiss.IndexHNSWFlat(D, M, IP)  # Approximate graph search
        index.hnsw.efConstruction = 200         # Build-time quality
        index.hnsw.efSearch = 128               # Query-time quality
    elif ft == "IVF_PQ":
        index = faiss.IndexIVFPQ(...)           # Trained approximate search
        index.train(embeddings)                 # Requires training step
    index.add(embeddings)
```

**HNSW parameters explained:**
- `M=32`: Each node connects to 32 neighbors in the graph. Higher M = better
  recall but more memory and slower build.
- `efConstruction=200`: During index building, consider 200 candidates for
  each insertion. Higher = better quality graph.
- `efSearch=128`: During search, explore 128 candidates. Must be >= topk
  for reliable results.

#### Subprocess Architecture (macOS Fix)

On macOS Apple Silicon, PyTorch and FAISS both use Apple's Accelerate BLAS
framework. Loading both in the same process causes a segfault at
`index.add()`. The fix: encode with PyTorch in the main process, save
embeddings to a `.npy` temp file, then spawn a **clean subprocess**
(`_faiss_index_worker.py`) that imports only numpy + FAISS to build the index.

```
Main process (has torch):     Subprocess (no torch):
  encode_texts() →              _faiss_index_worker.py
  save .npy file →                load .npy
                                  build FAISS index
                                  save artifacts
```

#### Saved Artifacts (per domain)

| File | Contents |
|------|----------|
| `faiss.index` | The FAISS index binary (vector data + graph structure) |
| `docstore.parquet` | Chunk metadata: chunk_id, doc_id, text, title, domain |
| `id_mapping.json` | Maps FAISS internal integer IDs → chunk_id strings |
| `index_metadata.json` | Encoder model name, dimension, vector count, index type |

### 8.2 `_faiss_index_worker.py` (`src/indexing/`)

Standalone subprocess script. Receives configuration as a JSON CLI argument.
Imports only `numpy`, `pandas`, and `faiss` — never `torch`. Builds the
FAISS index and saves all artifacts.

### 8.3 `BM25IndexBuilder` (`bm25_builder.py`)

Optional lexical index using `rank-bm25` (pure Python BM25Okapi).

```python
class BM25IndexBuilder:
    def build(self, corpus_parquet, output_dir):
        # 1. Load parquet
        # 2. Tokenize: "hello world" → ["hello", "world"]
        # 3. Fit BM25Okapi on tokenized texts
        # 4. Pickle the BM25 model + chunk_ids list
```

No Java required (unlike Pyserini/Anserini). Simple whitespace tokenization
+ lowercasing. Saves as `bm25.pkl`.

---

## 9. Layer 5: Retrieval (`src/retrieval/`)

### 9.1 `RetrievalResult` (dataclass in `dense_retriever.py`)

Every retrieved passage is wrapped in this dataclass, carrying all scores
through the pipeline:

```python
@dataclass
class RetrievalResult:
    chunk_id: str          # "science:4983__chunk0"
    doc_id: str            # "science:4983" (parent document)
    domain: str            # "science"
    score: float           # dense inner-product score
    text: str              # passage text
    title: str             # document title
    bm25_score: float      # BM25 score (0 if BM25 not used)
    fused_score: float     # RRF fused score (set after fusion)
    colbert_score: float   # ColBERT reranker score (set after reranking)
```

### 9.2 `DenseRetriever` (`dense_retriever.py`)

Wraps a per-domain FAISS index + encoder for online search.

#### Lifecycle

```python
retriever = DenseRetriever(domain, index_dir, encoder_model, device)
retriever.load()           # Load FAISS index + docstore + encoder model
results = retriever.search("my query", topk=100)  # Encode + search
```

#### `load()` — What Gets Loaded

1. `faiss.index` → FAISS index object (read-only after load)
2. `docstore.parquet` → DataFrame of chunk metadata
3. `id_mapping.json` → dict mapping FAISS int IDs to chunk_id strings
4. `index_metadata.json` → validates encoder model matches
5. Encoder model (AutoTokenizer + AutoModel from HuggingFace)

Also builds a reverse lookup: `chunk_to_row` maps chunk_id → DataFrame row
index for O(1) metadata lookups.

#### `search(query, topk)` — Search Flow

```
1. _encode([query])
   → tokenize → BERT forward pass → mean pool → L2 normalize
   → float32 array (1, 768)

2. self._index.search(vec, topk)
   → FAISS returns: scores[0] (float[]), indices[0] (int[])
   → indices are internal FAISS IDs (0, 1, 2, ...)

3. For each (score, idx):
   → _lookup(idx, score)
   → id_mapping[idx] → chunk_id
   → chunk_to_row[chunk_id] → row in docstore
   → Construct RetrievalResult with text, title, doc_id, score
```

#### `search_batch()` — Batch Search

Same as `search()` but encodes multiple queries at once for evaluation
efficiency. FAISS `search()` supports batch queries natively.

### 9.3 `BM25Retriever` (`bm25_retriever.py`)

Wraps a pickled `rank-bm25` BM25Okapi index for one domain.

#### `search(query, topk)`

```python
def search(self, query, topk=100):
    tokens = query.lower().split()           # Simple whitespace tokenization
    scores = self._bm25.get_scores(tokens)   # Score ALL passages
    top_idxs = np.argpartition(scores, -topk)[-topk:]  # Top-k by partition
    top_idxs = top_idxs[np.argsort(scores[top_idxs])[::-1]]  # Sort top-k
    # Filter scores <= 0, look up metadata, return RetrievalResults
```

**Note:** BM25 scores ALL passages (O(n)), unlike FAISS which uses
approximate search. For large corpora, BM25 is slower than FAISS at query
time.

### 9.4 `RRFFusion` (`hybrid_fusion.py`)

Reciprocal Rank Fusion — merges multiple ranked lists into one.

```python
class RRFFusion:
    def __init__(self, rrf_k=60):
        self.rrf_k = rrf_k

    def fuse(self, ranked_lists, topk=1000):
        for ranked_list in ranked_lists:
            for rank, result in enumerate(ranked_list):
                rrf_score = 1.0 / (self.rrf_k + rank + 1)
                rrf_scores[result.doc_id] += rrf_score
        # Sort by cumulative RRF score, return top-k
```

**Used twice in the pipeline:**

1. **Within-domain fusion:** Merges dense + BM25 results for the same domain.
   Only happens when BM25 is enabled for that domain.
2. **Cross-domain fusion:** Merges results from different domain indexes.
   Always happens when multiple domains are queried.

**Deduplication:** Results from multiple chunks of the same document
(same `doc_id`) are merged. The representative chunk (for display) is the one
with the highest raw dense score. RRF score accumulates across all chunks.

---

## 10. Layer 6: Classification & Routing (`src/classification/`)

### 10.1 `DomainClassifier` (`domain_classifier.py`)

A fine-tuned DistilBERT model that classifies queries into one of 6 domains.
This is the **router** — it decides which domain indexes to search.

#### `ClassificationResult` (dataclass)

```python
@dataclass
class ClassificationResult:
    top1_domain: str           # e.g. "medical"
    top2_domain: str           # e.g. "biomedical"
    top1_prob: float           # e.g. 0.72
    top2_prob: float           # e.g. 0.15
    all_probs: dict[str, float]  # {"medical": 0.72, "biomedical": 0.15, ...}
```

#### `classify(query)` — How It Works

```
1. Tokenize query (max_length=64)
2. Forward pass through DistilBERT → logits (6-dimensional vector)
3. Softmax → probabilities
4. Sort by probability → top-1 and top-2 domains
```

#### Saved Artifacts (`models/domain_classifier/`)

| File | Purpose |
|------|---------|
| `pytorch_model.bin` / `model.safetensors` | Model weights |
| `config.json` | Model architecture config |
| `tokenizer.json` + `vocab.txt` | Tokenizer files |
| `label_map.json` | Maps integer labels to domain names: `{0: "general", 1: "science", ...}` |

### 10.2 Training (`train_classifier.py`)

#### Data Pipeline

```python
def load_domain_queries(data_root, min_per_domain=200, max_per_domain=5000):
    for domain in DOMAIN_LABELS:
        texts = load from queries.parquet
        if len(texts) < min_per_domain:
            augment with seed queries (repeat until enough)
        if len(texts) > max_per_domain:
            randomly sample down to cap
    return texts, integer_labels
```

**Class balancing:** Domains with few real queries (medical: 50, science:
300) are augmented with handwritten seed queries. Domains with many queries
(general: 509k) are capped at 5,000. This prevents the classifier from
being dominated by the largest domain.

#### Training Setup

- **Base model:** `distilbert-base-uncased` (66M parameters, 6 layers)
- **Task:** 6-class sequence classification (softmax over domains)
- **Split:** 80% train, 20% validation (stratified by domain)
- **Optimizer:** AdamW, lr=2e-5, warmup=10%
- **Epochs:** 3 (with early stopping, patience=2)
- **Metric:** top-1 accuracy + top-2 accuracy

#### `QueryDataset` (PyTorch Dataset)

```python
class QueryDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings  # pre-tokenized input_ids + attention_mask
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }
```

**Note:** All queries are tokenized at once and held in memory. This works
for the current scale (~10k queries total) but would OOM if scaled to
millions.

### 10.3 Seed Queries (`seed_queries.py`)

Handcrafted representative queries for domains with too few real queries.
20 queries per domain for: science, medical, scidocs, biomedical.

```python
SEEDS = {
    "science": [
        "what is the mechanism of crispr gene editing",
        "how do black holes form and evaporate",
        ...
    ],
    "medical": [
        "what are the symptoms of type 2 diabetes",
        "clinical trial outcomes for lung cancer treatment",
        ...
    ],
    # ...
}
```

---

## 11. Layer 7: Reranking (`src/reranking/`)

### `ColBERTReranker` (`colbert_reranker.py`)

Optional second-stage reranker using ColBERT v2 (late-interaction model).
**Currently disabled** in default config (`reranking.enabled: false`).

#### What is ColBERT?

Unlike bi-encoders (which produce a single vector per text), ColBERT produces
**one vector per token**. Relevance is computed via MaxSim:

```
score(q, d) = Σ   max   cos(q_i, d_j)
             q_i  d_j

For each query token, find the most similar document token, sum those scores.
```

This captures fine-grained token-level interactions, producing better
rankings than bi-encoders — but at much higher computational cost.

#### How It's Used Here

ColBERT is used in **rerank-only mode** (no ColBERT index is built). It
re-scores the top-k candidates from the fusion stage:

```python
def rerank(self, query, candidates, topk=None):
    cap = topk or self.topk  # default 200
    to_rerank = candidates[:cap]
    tail = candidates[cap:]  # beyond cap, kept as-is

    texts = [f"{r.title} {r.text}" for r in to_rerank]
    raw = self._model.rerank(query=query, documents=texts, k=len(texts))
    # Map back to RetrievalResult, sorted by ColBERT score
    reranked.extend(tail)  # append unscored tail
    return reranked
```

Uses RAGatouille's `RAGPretrainedModel.rerank()` under the hood.

---

## 12. Layer 8: The Online Pipeline (`src/pipeline/`)

### `SearchPipeline` (`online_pipeline.py`)

The **orchestrator** — ties all components together for end-to-end search.

#### Initialization

```python
class SearchPipeline:
    def __init__(self, cfg):
        self.normalizer = QueryNormalizer()
        self.classifier = None          # loaded in load()
        self.dense_retrievers = {}      # domain → DenseRetriever
        self.bm25_retrievers = {}       # domain → BM25Retriever
        self.fuser = RRFFusion(rrf_k=60)
        self.reranker = None            # ColBERTReranker if enabled
```

#### `load()` — Startup (One-Time)

1. Load the domain classifier from `models/domain_classifier/`
2. For each enabled domain with a FAISS index:
   - Create and load a `DenseRetriever`
   - If BM25 is enabled, create and load a `BM25Retriever`
3. If reranking is enabled, load `ColBERTReranker`

#### `search(raw_query)` — Per-Query Flow

```python
def search(self, raw_query):
    # 1. Normalize
    norm_q = self.normalizer.normalize(raw_query)

    # 2. Classify → which domain(s)?
    clf = self.classifier.classify(norm_q)

    # 3. Route → select active domain indexes
    active_domains = self._get_active_domains(clf)

    # 4. Retrieve from each active domain
    for domain in active_domains:
        dense_results = self.dense_retrievers[domain].search(norm_q, topk=100)
        bm25_results = self.bm25_retrievers[domain].search(norm_q, topk=100)  # if enabled
        within_fused = self.fuser.fuse([dense_results, bm25_results])

    # 5. Cross-domain fusion
    fused = self.fuser.fuse(all_domain_lists, topk=100)

    # 6. Rerank (if enabled)
    final = self.reranker.rerank(norm_q, fused)  # or just fused

    return {results, classification, active_domains, latency}
```

#### Routing Modes (`_get_active_domains()`)

| Mode | Behavior | When to use |
|------|----------|-------------|
| `broadcast` | Query **all** enabled domain indexes | Maximum recall, highest latency |
| `routed_with_general` | Query top-1 predicted domain + `general` always; add top-2 if confidence < 0.45 | **Default.** Good balance of recall and speed. |
| `routed_only` | Query top-1 predicted domain only; add top-2 if low confidence | Fastest, but misrouting = zero recall |

**Confidence threshold (0.45):** When the classifier's top-1 probability is
below 0.45, the query is ambiguous. Adding the top-2 domain hedges against
routing errors.

---

## 13. Layer 9: Evaluation (`src/evaluation/`)

### 13.1 `metrics.py` — Standard IR Metrics

#### `evaluate_domain(qrels_df, results_by_qid, domain)`

Computes retrieval metrics for one domain using the `ir-measures` library.

```python
measures = [nDCG@10, MRR@10, Recall@100, Recall@1000]
agg = ir_measures.calc_aggregate(measures, qrels, run_scored)
```

#### `_strip_chunk_suffix(doc_id)`

Converts chunk IDs back to document IDs for evaluation:
`"science:4983__chunk0"` → `"science:4983"`.

QRels reference original document IDs, but retrieval returns chunk IDs.
This function bridges the gap.

#### `_build_run(results_by_qid)`

Converts `RetrievalResult` lists into the format `ir-measures` expects.
**Critical scoring logic:**

```python
# Use ColBERT score if available, else fused (RRF) score.
# fused_score is cross-domain comparable; raw dense score is NOT.
if r.colbert_score and r.colbert_score > 0:
    score = r.colbert_score
else:
    score = r.fused_score if r.fused_score > 0 else r.score
```

Raw dense scores are **not comparable across domains** — different encoders
produce different score scales. RRF fused scores are rank-based, making them
cross-domain comparable.

#### `measure_latency(search_fn, queries, n_trials, warmup)`

Runs `n_trials` queries through the pipeline and collects per-stage latency
statistics (mean and p95 for each stage: normalize, classify, retrieve,
fuse, rerank, total).

### 13.2 `ablation.py` — Ablation Study Framework

Defines experimental conditions as config overrides and runs the full
evaluation under each condition.

#### `AblationCondition` (dataclass)

```python
@dataclass
class AblationCondition:
    name: str           # "dense_only", "hybrid_with_rerank", etc.
    description: str    # Human-readable description
    overrides: dict     # Config overrides to apply
```

#### Default Conditions

| Condition | What it tests |
|-----------|--------------|
| `dense_only` | Dense retrieval only (no BM25, no reranking) |
| `bm25_only` | BM25 only (no dense) |
| `hybrid_no_rerank` | Dense + BM25 with RRF (no ColBERT) |
| `hybrid_with_rerank` | Dense + BM25 + RRF + ColBERT reranking |
| `routing_broadcast` | Query all domains |
| `routing_routed_general` | Routed to top-1 + general |
| `routing_routed_only` | Routed to top-1 only |

#### `run_ablation()` — How It Works

For each condition:
1. Deep-merge condition overrides onto base config
2. Create a fresh `SearchPipeline` with the modified config
3. Run all queries through the pipeline
4. Compute metrics
5. Collect into a comparison DataFrame

### 13.3 `cross_domain_eval.py` — FeB4RAG Tier 3

Evaluates cross-domain fusion quality using the FeB4RAG framework
(Wang et al., 2024).

#### Two Judge Backends

| Judge | How it scores | Pros | Cons |
|-------|--------------|------|------|
| `OllamaJudge` | Local LLM (Llama 3.1 8B) rates relevance 0-2 via REST API | No API cost, ~80% human agreement | Requires Ollama running locally |
| `CrossEncoderJudge` | `cross-encoder/ms-marco-MiniLM-L-6-v2` neural scorer | No external dependencies | Weaker on out-of-domain content |

#### Metrics Computed

- `domain_coverage@N` — fraction of queries with >=2 domains in top-N
- `mean_domains_in_topN` — average distinct domains in top-N
- `expected_domain_hit@N` — do all expected domains appear?
- `{judge}_nDCG@N` — mean nDCG@N using weak relevance labels

---

## 14. Layer 10: Scripts

### CLI Entry Points (`scripts/`)

| Script | Pipeline Step | What it does |
|--------|-------------|-------------|
| `build_corpus.py` | Step 1/5 | Loads datasets via adapters, chunks documents, saves as parquet. Skips domains where output exists (use `--force` to rebuild). |
| `build_faiss.py` | Step 2/5 | Encodes chunks with domain encoder, builds FAISS index. Supports `--domain` for single-domain builds. |
| `build_bm25.py` | Step 3/5 | Builds BM25 index from corpus parquet (optional, requires `bm25_enabled: true`). |
| `train_domain_classifier.py` | Step 4/5 | Loads queries from all domains, trains DistilBERT classifier, saves model. |
| `evaluate.py` | Step 5/5 | Runs full pipeline per query, computes nDCG/MRR/Recall, optionally runs ablation. |
| `evaluate_routing.py` | Tier 2 | Tests classifier accuracy: does it predict the correct domain for each query? |
| `evaluate_cross_domain.py` | Tier 3 | FeB4RAG cross-domain fusion quality with LLM/cross-encoder judge. |
| `evaluate_subjective.py` | — | Subjective quality evaluation. |
| `search.py` | — | Single-query CLI demo. |

### `run_all.py` — Pipeline Orchestrator

Runs steps 1-5 sequentially with CLI flags for skipping steps:

```bash
python run_all.py                        # smoke test (default)
python run_all.py --full                 # full data + GPU
python run_all.py --domain science       # single domain
python run_all.py --skip-corpus          # skip step 1
python run_all.py --eval-only            # skip steps 1-4
python run_all.py --search "my query"    # search demo instead of eval
python run_all.py --ablation             # run ablation conditions
```

---

## 15. Layer 11: Web UI

### `app.py` — Gradio Interface

A web-based search interface built with Gradio.

#### Components

- **Query input** with example queries (one per domain)
- **Top-K slider** (1-20 results)
- **Routing mode selector** (radio buttons)
- **Classification panel** — shows predicted domain + confidence
- **Latency bar** — per-stage timing breakdown
- **Results panel** — color-coded cards with domain badges, scores, and text snippets

#### Domain Color Coding

| Domain | Color | Icon |
|--------|-------|------|
| general | Indigo (#6366f1) | 🌐 |
| science | Sky blue (#0ea5e9) | 🔬 |
| finance | Emerald (#10b981) | 📈 |
| medical | Red (#ef4444) | 🏥 |
| legal | Amber (#f59e0b) | ⚖️ |
| biomedical | Violet (#8b5cf6) | 🧬 |

#### How It Works

1. On startup, `load_pipeline()` creates a `SearchPipeline` and calls `.load()`
2. On each query, `search()` calls `_pipeline.search(query)` and formats
   the results as HTML cards
3. The routing mode can be changed live via the radio buttons (overrides
   `_pipeline.cfg["routing"]["mode"]`)

```bash
python app.py                  # Launch on port 7860
python app.py --port 7861      # Custom port
python app.py --share          # Create public Gradio link
```

---

## 16. Data Flow Summary

```
HuggingFace Datasets
    │
    ▼ (Adapters: msmarco, scifact, fiqa, treccovid, nfcorpus, scidocs)
    │
corpus.parquet          queries.parquet          qrels.parquet
(doc_id, title,         (query_id, text,         (query_id, doc_id,
 text, domain)           domain)                  relevance)
    │                        │                        │
    ▼ (SlidingWindowChunker) │                        │
    │                        │                        │
chunked corpus.parquet       │                        │
(chunk_id, doc_id,           │                        │
 text, title, domain,        │                        │
 start_token, end_token)     │                        │
    │                        │                        │
    ├─────────────────┐      │                        │
    │                 │      │                        │
    ▼                 ▼      ▼                        │
FAISSIndexBuilder   BM25IndexBuilder                  │
    │                 │      │                        │
    ▼                 ▼      ▼                        │
faiss.index       bm25.pkl  train_classifier          │
docstore.parquet            │                         │
id_mapping.json             ▼                         │
                    domain_classifier/                │
                    (pytorch_model.bin,               │
                     label_map.json)                  │
                            │                         │
    ┌───────────────────────┘                         │
    │                                                 │
    ▼                                                 ▼
SearchPipeline.search(query)                   evaluate_domain()
    │                                           (compares retrieval
    ▼                                            results against qrels)
Ranked results with:
  - domain classification
  - per-stage latency
  - dense/bm25/fused/colbert scores
```

---

## 17. Per-Domain Encoder Models

Each domain uses a specialized BERT variant as its bi-encoder. The rationale
is that domain-specific pretraining captures terminology and semantic
relationships that a general BERT would miss.

| Domain | Encoder | Hidden Size | Pretraining Data | Training Objective |
|--------|---------|-------------|-----------------|-------------------|
| general | `sentence-transformers/msmarco-bert-base-dot-v5` | 768 | MS MARCO query-passage pairs | **Contrastive retrieval** (dot product) |
| scidocs | `allenai/scibert_scivocab_uncased` | 768 | 1.14M scientific papers (Semantic Scholar) | Masked Language Modeling (MLM) |
| science | `allenai/scibert_scivocab_uncased` | 768 | Same as scidocs | MLM |
| finance | `ProsusAI/finbert` | 768 | Financial news + articles | Fine-tuned for **sentiment classification** |
| medical | `emilyalsentzer/Bio_ClinicalBERT` | 768 | MIMIC-III clinical notes | MLM on clinical text |
| biomedical | `dmis-lab/biobert-base-cased-v1.1` | 768 | PubMed abstracts + PMC full texts | MLM on biomedical text |

**Important note:** Only `msmarco-bert-base-dot-v5` was trained for
retrieval. The others are pretrained language models whose embeddings were
not optimized for query-document similarity. See `BUG_AUDIT.md` (Bug 5) for
details and recommended alternatives.

---

## 18. Directory Structure

```
ir-search-engine/
├── configs/
│   ├── default.yaml               # Smoke test config (CPU, small subsets)
│   ├── full.yaml                  # Production overrides (GPU, full data)
│   └── classifier_balanced.yaml   # Alternative classifier config
│
├── src/
│   ├── config.py                  # YAML loading + deep merge
│   │
│   ├── adapters/                  # Dataset loading layer
│   │   ├── base_adapter.py        # ABC + Document/Query/QRel dataclasses
│   │   ├── msmarco_adapter.py     # general domain (2-pass loading)
│   │   ├── scidocs_adapter.py     # scidocs domain
│   │   ├── scifact_adapter.py     # science domain
│   │   ├── fiqa_adapter.py        # finance domain
│   │   ├── treccovid_adapter.py   # medical domain
│   │   └── nfcorpus_adapter.py    # biomedical domain
│   │
│   ├── preprocessing/             # Text preprocessing
│   │   ├── chunker.py             # SlidingWindowChunker (token-aware)
│   │   └── normalizer.py          # QueryNormalizer (unicode, lowercase, etc.)
│   │
│   ├── indexing/                   # Offline index building
│   │   ├── faiss_builder.py       # FAISSIndexBuilder (encode + index)
│   │   ├── _faiss_index_worker.py # Subprocess worker (macOS BLAS fix)
│   │   └── bm25_builder.py        # BM25IndexBuilder (rank-bm25)
│   │
│   ├── retrieval/                 # Online retrieval
│   │   ├── dense_retriever.py     # DenseRetriever + RetrievalResult
│   │   ├── bm25_retriever.py      # BM25Retriever
│   │   └── hybrid_fusion.py       # RRFFusion (Reciprocal Rank Fusion)
│   │
│   ├── classification/            # Domain routing
│   │   ├── domain_classifier.py   # DomainClassifier (inference)
│   │   ├── train_classifier.py    # Training logic + QueryDataset
│   │   └── seed_queries.py        # Handcrafted augmentation queries
│   │
│   ├── reranking/                 # Second-stage reranking
│   │   └── colbert_reranker.py    # ColBERTReranker (MaxSim scoring)
│   │
│   ├── pipeline/                  # End-to-end orchestration
│   │   └── online_pipeline.py     # SearchPipeline (the main entry point)
│   │
│   └── evaluation/                # Metrics and analysis
│       ├── metrics.py             # nDCG, MRR, Recall via ir-measures
│       ├── ablation.py            # Ablation study framework
│       └── cross_domain_eval.py   # FeB4RAG Tier 3 (LLM/CE judge)
│
├── scripts/                       # CLI entry points
│   ├── build_corpus.py            # Step 1: download + chunk
│   ├── build_faiss.py             # Step 2: encode + index
│   ├── build_bm25.py              # Step 3: BM25 index (optional)
│   ├── train_domain_classifier.py # Step 4: train router
│   ├── evaluate.py                # Step 5: compute metrics
│   ├── evaluate_routing.py        # Tier 2: routing accuracy
│   ├── evaluate_cross_domain.py   # Tier 3: cross-domain fusion
│   ├── evaluate_subjective.py     # Subjective evaluation
│   └── search.py                  # Single-query demo
│
├── run_all.py                     # Orchestrates steps 1-5
├── app.py                         # Gradio web UI
├── build_indexes.py               # Legacy index builder
├── requirements.txt               # Python dependencies
│
├── data/
│   └── processed/{domain}/        # Built by step 1
│       ├── corpus.parquet         # Chunked documents
│       ├── queries.parquet        # Evaluation queries
│       └── qrels.parquet          # Relevance judgments
│
├── indexes/
│   ├── faiss/{domain}/            # Built by step 2
│   │   ├── faiss.index
│   │   ├── docstore.parquet
│   │   ├── id_mapping.json
│   │   └── index_metadata.json
│   └── bm25/{domain}/             # Built by step 3
│       └── bm25.pkl
│
├── models/
│   └── domain_classifier/         # Built by step 4
│       ├── pytorch_model.bin (or model.safetensors)
│       ├── config.json
│       ├── tokenizer.json
│       ├── vocab.txt
│       └── label_map.json
│
├── results/                       # Built by step 5
│   ├── {domain}_metrics.json
│   ├── {domain}_ablation.csv
│   ├── {domain}_latency.json
│   └── macro_average.json
│
└── docs/
    ├── BUG_AUDIT.md               # Known bugs and fixes
    └── FRAMEWORK_GUIDE.md         # This document
```

---

## 19. Key Design Decisions & Trade-offs

### 19.1 Domain-Specific Encoders vs. Single Universal Encoder

**Choice made:** One encoder per domain.
**Trade-off:** Better domain coverage in theory, but requires loading 6 models
into memory at query time. Also, most domain encoders aren't retrieval-trained
(see Bug 5). A single strong retrieval encoder (e.g., BGE, E5) would use less
memory and likely produce better results.

### 19.2 Chunking at 180 Tokens with 40-Token Overlap

**Choice made:** Small chunks with moderate overlap.
**Trade-off:** Small chunks produce more focused embeddings but lose
document-level context. 180 tokens ≈ 2-3 sentences. Larger chunks (256-512)
would capture more context but dilute the embedding signal for specific
passages. The 40-token overlap prevents boundary artifacts.

### 19.3 Routed Search vs. Broadcast

**Choice made:** Default is `routed_with_general` (query top-1 domain +
general, add top-2 if unsure).
**Trade-off:** Routing reduces latency (search 2 indexes instead of 6) but
introduces routing error — if the classifier is wrong, relevant documents
in the correct domain are never searched. The `broadcast` mode eliminates
routing error but is 3-6x slower.

### 19.4 RRF for Fusion (Not Score Interpolation)

**Choice made:** RRF with k=60.
**Trade-off:** RRF is robust to score scale differences between rankers
(dense scores ≠ BM25 scores ≠ ColBERT scores). Score interpolation
(`α·dense + (1-α)·bm25`) requires careful tuning of α and score
normalization. RRF "just works" with no tuning.

### 19.5 Subprocess for FAISS Index Building

**Choice made:** Spawn a clean Python process for FAISS operations.
**Trade-off:** Adds complexity (temp files, JSON CLI args, subprocess
management) but completely eliminates the macOS BLAS segfault. The
alternative (in-process FAISS) works on Linux/Windows but crashes on Apple
Silicon.

### 19.6 DistilBERT for Routing (Not a Larger Model)

**Choice made:** DistilBERT-base (66M params, 6 layers).
**Trade-off:** Fast inference (~5ms per query on CPU) but limited capacity
for nuanced domain distinctions. A larger model (BERT-base, 110M) would be
more accurate but 2x slower. For a routing classifier where speed matters
and the task is relatively simple (6-class text classification), DistilBERT
is a reasonable choice.

---

## 20. Glossary

| Term | Definition |
|------|-----------|
| **Bi-encoder** | Architecture where query and document are encoded independently by the same model, then compared via similarity (dot product / cosine). |
| **BM25** | Best Matching 25 — classical probabilistic ranking function based on term frequency and inverse document frequency. |
| **Chunk** | A segment of a document, created by the sliding window chunker. Each document may produce 1+ chunks. |
| **ColBERT** | Contextualized Late Interaction over BERT — a model that produces per-token embeddings and computes relevance via MaxSim. More accurate but slower than bi-encoders. |
| **Dense retrieval** | Using neural network embeddings + vector similarity search (as opposed to keyword matching). |
| **Domain** | A topical category (general, science, finance, medical, biomedical, scidocs) with its own index and encoder. |
| **Docstore** | A Parquet file containing chunk metadata (text, title, IDs). Used to look up passage content after FAISS returns integer IDs. |
| **efSearch** | HNSW parameter controlling query-time search quality. Higher = more accurate but slower. Must be >= topk. |
| **Embedding** | A fixed-size dense vector (768 dimensions) representing a piece of text in a continuous space. |
| **FAISS** | Facebook AI Similarity Search — library for efficient nearest neighbor search over dense vectors. |
| **HNSW** | Hierarchical Navigable Small World — graph-based approximate nearest neighbor algorithm used in FAISS. |
| **Inner Product (IP)** | Dot product between two vectors. When vectors are L2-normalized, IP equals cosine similarity. |
| **IVF_PQ** | Inverted File with Product Quantization — FAISS index that partitions vectors into clusters and compresses them for memory efficiency. |
| **L2 normalization** | Scaling a vector to unit length (||v|| = 1). Enables inner product ≡ cosine similarity. |
| **Mean pooling** | Averaging all token embeddings (weighted by attention mask) to produce a single vector for the entire input. |
| **MRR@k** | Mean Reciprocal Rank at k — average of 1/rank of the first relevant result, across all queries. |
| **nDCG@k** | Normalized Discounted Cumulative Gain at k — measures ranking quality with graded relevance, penalizing relevant docs at lower ranks. |
| **QRel** | Query Relevance judgment — a triple (query_id, doc_id, relevance_score) from benchmark ground truth. |
| **Recall@k** | Fraction of all relevant documents that appear in the top-k results. |
| **Routing** | Deciding which domain index(es) to search for a given query, based on the domain classifier's prediction. |
| **RRF** | Reciprocal Rank Fusion — method for merging multiple ranked lists by summing reciprocal ranks. |
| **Streaming** | HuggingFace datasets mode that processes data row-by-row without loading the entire dataset into memory. |
