# MAILMIND RAG PIPELINE — INTERNAL TECHNICAL STUDY GUIDE

> **CONFIDENTIAL & INTERNAL USE ONLY**  
> *This document is an authoritative internal technical study guide and architecture reference for the MailMind system. It reflects the production implementation, real-time ingestion mechanisms, hybrid retrieval pipeline, BiLSTM classification, and empirical evaluation framework.*

---

## SECTION 1 — SYSTEM OVERVIEW AND ONE-PAGE BIG PICTURE

MailMind is an intelligent email information retrieval, prioritization, and grounded summarization assistant designed for complex, heterogeneous personal email collections (specifically tested on university campus and recruitment communications).

### What Happens When a User Enters a Question?

When a user submits a natural language question (via CLI `/ask "When is my Microsoft assessment?"`, REST API `POST /ask`, or the Streamlit interface), the system processes the request through a 16-step pipeline:

```
[1. User Query] ──► [2. RAGPipeline.ask()] ──► [3. reload_if_changed()]
                                                        │
[6. Dense Candidates] ◄── [5. FAISS Search] ◄── [4. embed_query()]
         │                                              │
         ▼                                              ▼
[7. Query Analysis] ──► [8. BM25 Search] ──► [9. Lexical Candidates]
                                                        │
[10. Reciprocal Rank Fusion (RRF)] ◄────────────────────┘
         │
         ▼
[11. True Dot-Product Reconstruction]
         │
         ▼
[12. Multi-Signal Metadata Reranking & Message-ID Deduplication]
         │
         ▼
[13. Similarity Threshold Filter (0.25)] ──► (If Empty: Safe Abstention)
         │
         ▼
[14. Top-5 Evidence-Focused Context Assembly]
         │
         ▼
[15. Grounded LLM Generation (Gemini 3.5 Flash-Lite / 2.0 Flash)]
         │
         ▼
[16. Grounded Answer + Validated Source Citations -> User UI/CLI]
```

### Complete 16-Step Execution Flow

1. **Query Entry**: The user submits a query string to the CLI (`rag/run.py`), FastAPI backend (`api/main.py`), or Streamlit UI (`streamlit_app/app.py`).
2. **Pipeline Invocation**: The query is routed to `pipeline.ask(query)` in `rag/pipeline.py`.
3. **Retrieval Freshness Invalidation**: `pipeline.ask()` calls `pipeline.reload_if_changed()`, which performs a sub-millisecond file modification check (`st_mtime_ns`, `st_size`) on `emails.faiss` and `metadata.json`. If real-time ingestion occurred, the in-memory index and metadata reload under a reentrant lock (`threading.RLock()`), and the BM25 index is automatically invalidated.
4. **Query Embedding**: `retriever.py` calls `embed_query(query)` in `rag/embedder.py`. `sentence-transformers/all-MiniLM-L6-v2` converts the text into a 384-dimensional dense vector and applies L2 normalization.
5. **Dense FAISS Vector Search**: `store.search(query_vector, top_k=20)` executes an exact inner product search (`faiss.IndexFlatIP`) across 13,297 chunk vectors.
6. **Dense Candidate Generation**: FAISS yields the Top-20 dense candidates ranked by cosine similarity.
7. **Query Analysis**: `query_analyzer.analyze_query(query)` tokenizes terms, identifies proper nouns/entities, detects temporal keywords (`UPCOMING`, `DATE_SEARCH`), extracts email senders, and forms bigrams.
8. **Lexical BM25 Search**: `hybrid_retriever.get_bm25_index(store)` searches the in-memory inverted index using the analyzed query terms to retrieve the Top-20 lexical candidates.
9. **Lexical Candidate Generation**: BM25 identifies keyword-exact documents (capturing specialized acronyms, course codes like `CSE3002`, or short identifiers that dense embeddings dilute).
10. **Reciprocal Rank Fusion (RRF)**: `search_hybrid()` merges dense and lexical candidates using RRF with smoothing constant $k_{rrf} = 60$. Dense and lexical candidate lists are fused:
    $$\text{RRF Score}(d) = \frac{w_{\text{dense}}}{60 + r_{\text{dense}}(d)} + \frac{w_{\text{lex}}}{60 + r_{\text{lex}}(d)}$$
11. **True Dot-Product Vector Reconstruction**: For candidates recovered solely via BM25, their dense vectors are reconstructed from FAISS via `store.index.reconstruct(vid)` to compute their true semantic inner product with the query.
12. **Multi-Signal Metadata Reranking**: `rerank_candidates()` in `rag/reranker.py` scores candidates using base weights that sum to 1.00:
    - Semantic similarity cosine score: 0.45 (`RERANK_SEMANTIC_WEIGHT`)
    - Subject token overlap: 0.20 (`RERANK_SUBJECT_WEIGHT`)
    - Sender matching: 0.15 (`RERANK_SENDER_WEIGHT`)
    - Entity exact alignment: 0.10 (`RERANK_ENTITY_WEIGHT`)
    - Body text token match: 0.10 (`RERANK_BODY_WEIGHT`)
    Followed by additive modifiers: temporal intent bonus (0.30 when temporal intent is detected) or recency freshness prior (0.10 when dated), subject bigram bonus (0.08), and venue/date bonuses. Concurrently, candidate chunks sharing the same `message_id` are deduplicated, preserving only the highest-scoring chunk per email.
13. **Similarity Threshold Filtering**: Candidates failing the `MIN_SIMILARITY_THRESHOLD = 0.25` are purged. If no candidates survive (e.g. for completely out-of-domain questions), the pipeline returns an uncertainty abstention response without invoking the LLM API.
14. **Context Construction**: The top 5 surviving candidates are converted into clean evidence blocks containing metadata headers and body snippets (`build_context_text()`).
15. **Grounded Generation**: `build_rag_prompt()` injects the evidence context and question into a strict system prompt instructing Google Gemini 3.5 Flash-Lite / 2.0 Flash to answer exclusively from the retrieved text, cite sources using `[Source i]`, and abstain if evidence is insufficient.
16. **Citation Formatting & Delivery**: The generated answer is paired with verified `SearchResult` source citations and returned as an `AskResponse`.

---

## SECTION 2 — REPOSITORY ARCHITECTURE AND KEY MODULES

### Directory Structure Map

```
d:/Github/MailMind/
├── api/                             # FastAPI Backend & Gmail Webhooks
│   ├── gmail/                       # OAuth2 with PKCE, Watcher, Pub/Sub processing
│   ├── routes/                      # API endpoints (search, ask, sync, webhook)
│   └── main.py                      # Application lifespan and routing
├── assets/                          # Diagrams, architecture flowcharts, study guide
├── data/rag/index/                  # Persisted FAISS index & metadata
│   ├── emails.faiss                 # Flat inner product binary index
│   └── metadata.json                # Vector ID to DocumentChunk metadata mapping
├── datasets/                        # Preprocessed email datasets (Parquet / JSON)
├── evaluation/                      # Rigorous evaluation framework
│   ├── benchmark_v1.json            # Frozen benchmark (50 queries, 8 categories)
│   ├── benchmark.py                 # Benchmark loader & coverage verifier
│   ├── retrieval_eval.py            # Retrieval evaluator (Recall, Success, MRR, NDCG)
│   ├── rag_eval.py                  # RAG answer evaluator (LLM judge & deterministic)
│   ├── classifier_eval.py           # BiLSTM test evaluation on held-out split
│   ├── reliability_eval.py          # 7 deterministic reliability checks & latency
│   ├── reporter.py                  # HTML/Markdown evaluation report generator
│   ├── run.py                       # Unified evaluation runner CLI
│   └── results/                     # Generated evaluation reports and JSON outputs
├── ml/categorization/               # BiLSTM Email Classifier
│   ├── train_bilstm.py              # Model training script
│   ├── predict.py                   # Classifier inference interface
│   └── model/                       # Serialized Keras 3 model & tokenizer
├── rag/                             # Production RAG Pipeline Core
│   ├── config.py                    # Centralized settings & hyper-parameters
│   ├── schemas.py                   # Typed Pydantic models
│   ├── loader.py                    # Parquet / JSON email dataset loader
│   ├── normalizer.py                # HTML cleaning and header normalization
│   ├── chunker.py                   # 400-word chunker with metadata header
│   ├── embedder.py                  # all-MiniLM-L6-v2 SentenceTransformer embedder
│   ├── vector_store.py              # FAISS store with reload_if_changed & thread lock
│   ├── hybrid_retriever.py          # BM25 inverted index & Reciprocal Rank Fusion
│   ├── query_analyzer.py            # Intent, entity, and temporal query extractor
│   ├── reranker.py                  # 6-signal composite reranker & dedup
│   ├── prompt.py                    # Evidence-focused prompt builder
│   ├── generator.py                 # Gemini LLM grounded generator via google-genai
│   ├── conversation.py              # Multi-turn conversational memory & rephrasing
│   ├── summarizer.py                # Single & multi-email summarization
│   └── pipeline.py                  # RAGPipeline unified facade
├── streamlit_app/                   # Interactive User Interface
│   └── app.py                       # Multi-tab Streamlit dashboard with live reload
└── tests/                           # Comprehensive Pytest Suite (161 tests)
```

---

## SECTION 3 — COMPONENT-BY-COMPONENT SPECIFICATIONS

### 1. `rag/config.py` — Centralized Configuration
- **Purpose**: Defines configuration variables and environment parameters using Pydantic's `BaseSettings`.
- **Key Parameters**:
  - `EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"`
  - `EMBEDDING_DIMENSION = 384`
  - `INITIAL_RETRIEVAL_K = 20` (dense and lexical retrieval depth)
  - `FINAL_RETRIEVAL_K = 5` (final reranked context depth)
  - `CHUNK_SIZE = 400` words, `CHUNK_OVERLAP = 50` words
  - `MIN_SIMILARITY_THRESHOLD = 0.25`
  - `GEMINI_MODEL = "gemini-3.5-flash-lite"` (or `gemini-2.0-flash`)
- **Significance**: Provides a single source of truth for paths, thresholds, and weights across CLI, API, Streamlit, and evaluation runners.

### 2. `rag/vector_store.py` — FAISS Vector Store & Dynamic Invalidation
- **Index Type**: `faiss.IndexFlatIP(384)` (Flat Inner Product, exact cosine similarity over L2-normalized vectors).
- **Concurrency & Freshness**:
  - Protected by a reentrant lock: `self._lock = threading.RLock()`.
  - Maintains `self._disk_signature = (faiss_mtime_ns, faiss_size, meta_mtime_ns, meta_size)` and `self._version`.
  - Implements `reload_if_changed(index_dir)`: Checks `os.stat` in ~2 microseconds. If disk files changed (due to background sync or Pub/Sub push), reloads the index and metadata into memory and increments `_version`.
  - Employs atomic write replacement with retry backoff to avoid Windows file-locking collisions during concurrent read/write operations.

### 3. `rag/hybrid_retriever.py` — BM25 & Reciprocal Rank Fusion
- **BM25 Inverted Index**:
  - In-memory BM25 index built directly from tokenized email chunk texts and subjects.
  - Automatically invalidates and rebuilds when `store.version` increments (`_store_version != store.version`).
- **Reciprocal Rank Fusion**:
  - Fuses dense and lexical ranks using $k_{rrf} = 60$:
    $$\text{RRF Score} = \frac{1.0}{60 + r_{\text{dense}}} + \frac{1.0}{60 + r_{\text{lex}}}$$
  - For candidates found only by BM25, reconstructs the dense vector via `store.index.reconstruct(vid)` to calculate exact cosine similarity.

### 4. `rag/query_analyzer.py` — Intent & Entity Extraction
- **Capabilities**:
  - Normalizes text and extracts terms and bigrams.
  - Detects temporal keywords (`today`, `tomorrow`, `schedule`, `deadline`, `upcoming`, `dates`) and semester markers.
  - Extracts proper noun entities using capitalization and regex patterns.
  - Identifies target email senders and domain authorities.

### 5. `rag/reranker.py` — Multi-Signal Scoring & Deduplication
- **Scoring Model**:
  $$\text{Base Score} = 0.45 \cdot S_{\text{semantic}} + 0.20 \cdot S_{\text{subject}} + 0.15 \cdot S_{\text{sender}} + 0.10 \cdot S_{\text{entity}} + 0.10 \cdot S_{\text{body}} + \Delta_{\text{bonuses}}$$
  $$\text{Final Score} = \text{Base Score} + \begin{cases} 0.30 \cdot S_{\text{temporal}}, & \text{if temporal intent detected} \\ 0.10 \cdot S_{\text{freshness}}, & \text{if dated without temporal intent} \end{cases}$$
- **Deduplication**: When multiple 400-word chunks originate from the same `message_id`, preserves only the highest-scoring chunk to maximize diversity in the LLM's prompt context.

### 6. `rag/generator.py` & `rag/prompt.py` — Grounded Generation
- **SDK**: Modern `google-genai` SDK (`from google import genai`).
- **Grounding Constraints**: System prompt explicitly instructs the LLM:
  - Base the response ONLY on the provided retrieved email excerpts.
  - Cite sources using `[Source i]` markers.
  - If the excerpts do not contain the answer, state clearly that the mailbox does not contain sufficient information.

### 7. `ml/categorization/` — BiLSTM Email Priority Classifier
- **Architecture**: Bidirectional LSTM on Keras 3 / TensorFlow with an embedding layer, spatial dropout, bidirectional LSTM (64 units), dense layer (64 units, ReLU), and softmax output (3 classes: `spam`, `general`, `important`).
- **Dataset**: `email_classification_dataset.parquet` (5,420 total samples; 10% stratified held-out test split of 542 samples).

### 8. `api/routes/gmail.py` & `api/gmail/` — Ingestion & Pub/Sub Security
- **OAuth2 Flow**: Implements Google OAuth2 with PKCE (`code_challenge` / `code_verifier`) persisted in session state.
- **Webhook Security**: `POST /gmail/webhook` authenticates Google-signed OIDC JWT Bearer tokens, verifying signature, audience (`PUBSUB_VERIFICATION_AUDIENCE`), and service account email (`PUBSUB_SERVICE_ACCOUNT_EMAIL`).

---

## SECTION 4 — RIGOROUS EVALUATION BENCHMARK & CURRENT EVIDENCE

### Benchmark Composition (`evaluation/benchmark_v1.json`)

The frozen benchmark comprises **50 curated queries** across **8 domain categories**:
- `placement`: 21 queries
- `academic`: 9 queries
- `schedules_venues`: 5 queries
- `security`: 4 queries
- `sender_based`: 4 queries
- `irrelevant` (negative out-of-domain queries): 3 queries
- `temporal`: 2 queries
- `summarization`: 2 queries

**Splits**:
- **Frozen Test Set (`test`)**: 36 queries (34 domain-answerable + 2 negative)
- **Development Set (`dev`)**: 14 queries (13 domain-answerable + 1 negative)
- **Ground Truth**: 105 graded relevance judgments (grades: 2 = directly relevant, 1 = partially relevant, 0 = irrelevant). 100% of ground-truth message IDs verified present in the 10,782 indexed corpus emails.

### 1. Retrieval Performance Comparison

| Benchmark Split | Metric | Legacy Retriever | Improved Retriever (Hybrid + Reranking) | Engineering Target | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Full Benchmark**<br>*(50 queries: 47 domain, 3 neg)* | **Recall@5**<br>**Recall@10**<br>**Recall@20**<br>**Success@10**<br>**Negative FP Rate** | 50.00%<br>57.45%<br>68.26%<br>72.34%<br>3/3 (100.0%) | **72.52%**<br>**81.74%**<br>**85.28%**<br>**89.36%**<br>**0/3 (0.00%)** | $\ge 80.00\%$<br>$\ge 80.00\%$<br>$\ge 80.00\%$<br>$\ge 85.00\%$<br>0/3 (0.00%) | Below Target (+22.52%)<br>**PASS** (+24.29%)<br>**PASS** (+17.02%)<br>**PASS** (+17.02%)<br>**PASS** (-100.0%) |
| **Frozen Test Split**<br>*(36 queries: 34 domain, 2 neg)* | **Recall@5**<br>**Recall@10**<br>**Recall@20**<br>**Success@10**<br>**Negative FP Rate** | 49.51%<br>53.92%<br>64.95%<br>70.59%<br>2/2 (100.0%) | **70.34%**<br>**79.17%**<br>**82.60%**<br>**85.29%**<br>**0/2 (0.00%)** | $\ge 80.00\%$<br>$\ge 80.00\%$<br>$\ge 80.00\%$<br>$\ge 85.00\%$<br>0/2 (0.00%) | Below Target (+20.83%)<br>Slightly Below Target (+25.25%)<br>**PASS** (+17.65%)<br>**PASS** (+14.70%)<br>**PASS** (-100.0%) |
| **Development Split**<br>*(14 queries: 13 domain, 1 neg)* | **Recall@5**<br>**Recall@10**<br>**Recall@20**<br>**Success@10**<br>**Negative FP Rate** | 51.28%<br>66.67%<br>76.92%<br>76.92%<br>1/1 (100.0%) | **78.21%**<br>**88.46%**<br>**92.31%**<br>**100.00%**<br>**0/1 (0.00%)** | $\ge 80.00\%$<br>$\ge 80.00\%$<br>$\ge 80.00\%$<br>$\ge 85.00\%$<br>0/1 (0.00%) | Below Target (+26.93%)<br>**PASS** (+21.79%)<br>**PASS** (+15.39%)<br>**PASS** (+23.08%)<br>**PASS** (-100.0%) |

### 2. Category-wise Retrieval Breakdown (Improved Retriever, Full Benchmark)

| Category | Query Count | Recall@5 | Recall@10 | Recall@20 | Success@10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `academic` | 9 | 83.33% | 83.33% | 88.89% | 100.00% |
| `placement` | 21 | 76.59% | 84.13% | 87.30% | 90.48% |
| `schedules_venues` | 5 | 100.00% | 100.00% | 100.00% | 100.00% |
| `security` | 4 | 75.00% | 100.00% | 100.00% | 100.00% |
| `sender_based` | 4 | 12.50% | 37.50% | 37.50% | 50.00% |
| `temporal` | 2 | 25.00% | 37.50% | 50.00% | 50.00% |
| `summarization` | 2 | 75.00% | 100.00% | 100.00% | 100.00% |
| `irrelevant` (negative) | 3 | N/A | N/A | N/A | 0% FP (100% Rejected) |

### 3. RAG Answer Generation Quality (18 Curated Queries, LLM-as-Judge)

| Metric | Score | Target | Description & Operational Definition |
| :--- | :---: | :---: | :--- |
| **Answer Correctness** | **95.80%** | $\ge 85.00\%$ | Factual agreement with verified reference answer (1–5 Likert normalized) |
| **Groundedness / Faithfulness** | **83.30%** | $\ge 90.00\%$ | Absence of unsupported claims or hallucinations against retrieved context |
| **Answer Relevance** | **100.00%** | $\ge 90.00\%$ | Direct responsiveness to user inquiry |
| **Citation Correctness** | **87.50%** | $\ge 85.00\%$ | Accurate attribution to retrieved email sources |
| **Abstention Accuracy** | **100.00%** | 100.00% | Correct refusal on unanswerable / out-of-domain queries |
| **Unsupported Claims Detected** | **13** | 0 | Total hallucinated factual statements identified across 18 evaluations |

### 4. BiLSTM Email Classifier Performance (542 Held-out Test Samples)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `general` | 86.67% | 88.14% | 87.39% | 236 |
| `important` | 80.15% | 76.76% | 78.42% | 142 |
| `spam` | 98.80% | 100.00% | 99.39% | 164 |
| **Overall Accuracy** | **88.75%** | — | — | 542 |
| **Macro Average** | **88.54%** | **88.30%** | **88.40%** | 542 |
| **Weighted Average** | **88.63%** | **88.75%** | **88.67%** | 542 |

**Confusion Matrix**:
- Actual `general`: 208 predicted general, 27 predicted important, 1 predicted spam
- Actual `important`: 32 predicted general, 109 predicted important, 1 predicted spam
- Actual `spam`: 0 predicted general, 0 predicted important, 164 predicted spam

### 5. Latency Profile & Arithmetic Consistency Analysis

| Pipeline Stage | Latency | Profiling Method | Scope & Details |
| :--- | :---: | :--- | :--- |
| **Query Embedding** | `12.48 ms` | Isolated micro-benchmark | `all-MiniLM-L6-v2` dense vectorization (384-d) |
| **FAISS Dense Search** | `1.47 ms` | Isolated micro-benchmark | Inner product search over 13,297 vectors |
| **BM25 Lexical Search** | `1.87 ms` | Isolated micro-benchmark | In-memory token inverted index search |
| **Metadata Reranking & Dedup** | `17.49 ms` | Isolated micro-benchmark | 6-weight scoring & message-ID deduplication |
| **Context Assembly** | `0.10 ms` | Isolated micro-benchmark | Header formatting and snippet assembly |
| **Isolated Subtotal** | **`33.41 ms`** | Arithmetic sum | Sum of isolated sub-functions |
| **Total End-to-End Retrieval** | **`51.10 ms`** | End-to-end `retrieve_with_context` | Full pipeline including query analysis, RRF score fusion, vector reconstruction for BM25 hits, candidate thresholding, and object conversion |
| **LLM Generation** | `3480.33 ms` | Network API call | Gemini 3.5 Flash-Lite grounded answer synthesis |

> **Arithmetic Discrepancy Clarification**: The isolated subtotal of 33.41 ms does not match the end-to-end retrieval latency of 51.10 ms because isolated calls measure sub-components in isolation without query analysis regex execution, Reciprocal Rank Fusion calculation, FAISS vector reconstruction (`store.index.reconstruct`) for BM25 candidates, candidate filtering, and data schema marshalling. Total retrieval on CPU remains well under 55 ms.

---

## SECTION 5 — HOW TO REPRODUCE AND RUN EVALUATIONS

All evaluation workflows are unified in `evaluation/run.py`:

```bash
# 1. Run complete evaluation suite across all 50 queries
python -m evaluation.run --all

# 2. Run retrieval benchmark only on frozen test split
python -m evaluation.run --split test --retrieval

# 3. Run RAG evaluation on curated representative queries
python -m evaluation.run --rag

# 4. Run RAG evaluation across all queries in offline mode (deterministic rule judge)
python -m evaluation.run --rag-all --offline

# 5. Run BiLSTM classifier evaluation on held-out test data
python -m evaluation.run --classifier

# 6. Run reliability and latency verification suite
python -m evaluation.run --reliability
```

To run all automated Pytest unit and regression tests:
```bash
pytest tests/ -v
# Verified: 161 passed in ~30s
```

---

## SECTION 6 — KNOWN LIMITATIONS AND WHAT NOT TO CLAIM

1. **Do NOT claim Recall@5 meets the 80% target.** Full benchmark Recall@5 is 72.52% (frozen test: 70.34%). This is because complex recruitment events produce more than 5 distinct official circulars (shortlists, venue updates, reminders).
2. **Do NOT claim Groundedness is 100% or exceeds 90%.** Measured groundedness is 83.30% with 13 unsupported statements detected. The LLM occasionally embellishes introductory remarks or contextual dates.
3. **Do NOT claim sender-based or temporal retrieval is solved.** Performance on `sender_based` (37.50% Recall@10) and `temporal` (37.50% Recall@10) lags significantly behind academic (83.33%) and placement (84.13%) retrieval.
4. **Do NOT claim negative query evaluation is statistically definitive.** Negative query sample size is small ($N=3$ in full, $N=2$ in test). While the system achieves 0% false positives (100% abstention), this small sample introduces higher statistical uncertainty.
5. **Do NOT claim we fine-tuned Gemini.** Gemini is accessed off-the-shelf via API with prompt engineering and strict context bounding.
6. **Do NOT claim an enterprise cloud vector database was used.** The system runs locally with FAISS `IndexFlatIP` and an in-memory BM25 index.
