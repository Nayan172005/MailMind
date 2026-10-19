# MailMind End-to-End Evaluation Report

**Date:** 2026-10-18 17:15:24  
**Benchmark Version:** 1.0.0  
**Embedding Model:** sentence-transformers/all-MiniLM-L6-v2 (384-dim)  
**Vector Store:** FAISS IndexFlatIP (Cosine) | **Indexed Corpus:** 10,782 emails  
**RAG Generator:** Google Gemini (gemini-3.5-flash-lite)  
**Classifier:** BiLSTM (Spam / General / Important) on Keras 3  

---

## 1. Executive Summary

| Evaluation Pillar | Primary Metric | Value | Baseline / Target | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Retrieval Recall@10** | Recall@10 | **81.74%** | Target: >= 80.00% (Legacy: 57.45%) | **PASS** |
| **Retrieval Recall@20** | Recall@20 | **85.28%** | Target: >= 80.00% (Legacy: 68.26%) | **PASS** |
| **Retrieval Success@10** | Success@10 | **89.36%** | Target: >= 85.00% (Legacy: 72.34%) | **PASS** |
| **Negative Query Rejection** | False-Positive Rate | **0/3 (0.00%)** | Target: 0/3 (0.00%) | **PASS** |
| **RAG Groundedness** | Faithfulness | **83.30%** | Target: >= 90.00% | **BELOW TARGET** |
| **RAG Abstention** | Out-of-Domain Refusal | **100.00%** | Target: 100.00% | **PASS** |
| **Email Classifier** | Macro-F1 (Held-out Test) | **88.40%** | Target: >= 90.00% (Acc: 88.75%) | **BELOW TARGET** |
| **Reliability Suite** | System Health Checks | **7/7 Passed** | 100% Deterministic | **PASS** |

---

## 2. Retrieval Evaluation: Legacy vs Improved

Evaluation performed across **50 benchmark queries** (47 eligible domain queries + 3 negative out-of-domain queries).
Convention: Relevance grades 1 (partially relevant) and 2 (directly relevant) are counted as relevant for binary recall.

| Metric | Legacy Retriever | Improved Retriever | Target | Absolute Improvement | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Recall@5** | 50.00% | **72.52%** | >= 80.00% | +22.52% | **BELOW TARGET** |
| **Recall@10** | 57.45% | **81.74%** | >= 80.00% | +24.29% | **PASS** |
| **Recall@20** | 68.26% | **85.28%** | >= 80.00% | +17.02% | **PASS** |
| **Success@10** | 72.34% | **89.36%** | >= 85.00% | +17.02% | **PASS** |
| **Negative Query FP Rate** | 3/3 (100.00%) | **0/3 (0.00%)** | 0/3 (0.00%) | -100.00% | **PASS** |

### Category-wise Breakdown (Improved Retriever)

| Category | Queries | Recall@5 | Recall@10 | Recall@20 | Success@10 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `academic` | 9 | 83.33% | 83.33% | N/A | 100.00% |
| `placement` | 21 | 76.59% | 84.13% | N/A | 90.48% |
| `schedules_venues` | 5 | 100.00% | 100.00% | N/A | 100.00% |
| `security` | 4 | 75.00% | 100.00% | N/A | 100.00% |
| `sender_based` | 4 | 12.50% | 37.50% | N/A | 50.00% |
| `temporal` | 2 | 25.00% | 37.50% | N/A | 50.00% |
| `summarization` | 2 | 75.00% | 100.00% | N/A | 100.00% |

---

## 3. RAG Generation Evaluation

Evaluates the Gemini generation pipeline against ground truth reference answers and retrieved context.

| Metric | Score | Target | Description |
| :--- | :--- | :--- | :--- |
| **Answer Correctness** | **95.80%** | >= 85% | Factual agreement with verified reference answer |
| **Groundedness / Faithfulness** | **83.30%** | >= 90% | Zero unsupported claims or hallucinations |
| **Answer Relevance** | **100.00%** | >= 90% | Direct responsiveness to user inquiry |
| **Citation / Source Correctness** | **87.50%** | >= 85% | Accurate attribution to retrieved email sources |
| **Abstention Accuracy** | **100.00%** | 100% | Correct refusal on unanswerable / out-of-domain queries |
| **Unsupported Claims Detected** | **13** | 0 | Total hallucinated factual statements |

---

## 4. BiLSTM Classifier Evaluation

Evaluated strictly on the **held-out test split (542 samples)** from `ml/categorization/datasets/email_classification_dataset.parquet` (test_size=0.10, seed=42, stratified).

- **Accuracy:** 88.75%
- **Macro-F1:** 88.40%
- **Weighted-F1:** 88.67%

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Test Support |
| :--- | :--- | :--- | :--- | :--- |
| `general` | 86.67% | 88.14% | 87.39% | 236 |
| `important` | 80.15% | 76.76% | 78.42% | 142 |
| `spam` | 98.80% | 100.00% | 99.39% | 164 |

### Confusion Matrix

| Actual \ Predicted | `general` | `important` | `spam` |
| :--- | :---: | :---: | :---: |
| `general` | 208 | 27 | 1 |
| `important` | 32 | 109 | 1 |
| `spam` | 0 | 0 | 164 |

---

## 5. End-to-End Reliability and Latency

- **System Checks Passed:** 7 / 7 (100% deterministic)

### Granular Latency Profile

| Component | Latency | Description |
| :--- | :--- | :--- |
| **SentenceTransformer Embedding** | `12.48 ms` | `all-MiniLM-L6-v2` query vectorization |
| **FAISS Dense Vector Search** | `1.47 ms` | Inner-product search over indexed vectors |
| **BM25 Lexical Inverted Search** | `1.87 ms` | In-memory token inverted index search |
| **Metadata Reranker & Dedup** | `17.49 ms` | Intent, temporal, sender & bigram reranking |
| **Context Assembly** | `0.10 ms` | Evidence extraction and citation formatting |
| **Total Retrieval Latency** | `51.10 ms` | Measured end-to-end `retrieve_with_context` |
| **Gemini Generation Latency** | `3480.33 ms` | LLM grounded generation (network API call) |

---

## 6. Error Analysis and Known Limitations

1. **Multiple Similar Circulars:** Corporate placement drives often send multiple circulars (e.g., initial notification, reminder, shortlist, and venue confirmation). While Improved retrieval retrieves the correct set in top 10, Recall@5 can be constrained when more than 5 distinct emails exist for a single event.
2. **Negative Query Thresholding:** The similarity threshold (0.25) combined with out-of-domain candidate validation effectively blocks distinct out-of-domain questions with zero false positives.
3. **Idempotency Guarantee:** Ingestion pipeline verifies message IDs before insertion, ensuring index consistency during repetitive sync operations.