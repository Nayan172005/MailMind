# MailMind Evaluation Framework

A rigorous, one-command end-to-end evaluation suite for **MailMind** (Intelligent Email Summarization and Information Retrieval System).

## Overview

The evaluation suite tests all key system capabilities against authentic corpus data without altering any production databases or modifying user mailbox data:

1. **Frozen Relevance-Judged Retrieval Benchmark (`benchmark_v1.json`)**:
   - 50 representative queries spanning 9 core categories:
     * Placement Notifications & Selections
     * Company-Specific Assessments
     * Schedules & Venues
     * Academic Notices & Course Registrations
     * Security Alerts
     * Sender-Based Queries
     * Temporal Queries
     * Summarization-Oriented Questions
     * Out-of-Domain & Irrelevant Queries (Negative Controls)
   - Graded Relevance Judgments:
     * **Grade 2**: Directly relevant (primary target document answering query)
     * **Grade 1**: Partially relevant (relevant context, follow-up, general policy)
     * **Grade 0**: Irrelevant (unrelated or false positive match)
   - Fixed split: **38 Test queries** and **12 Development queries**.
   - Corpus coverage verification: 100% of benchmark message IDs verified against the 10,782 indexed emails.

2. **Retrieval Metrics Engine (`retrieval_eval.py`)**:
   - Macro-averaged metrics across eligible queries:
     * `Recall@5`, `Recall@10`, `Recall@20` (binary convention: grades 1 & 2 counted as relevant)
     * `Precision@5`, `Precision@10`
     * `MRR` (Mean Reciprocal Rank)
     * `nDCG@10` (using graded relevance: 2, 1, 0 with standard logarithmic discounting)
     * `Success@5` and `Success@10` (Hit@K)
   - Separate negative query evaluation (False Positive Rate and max similarity score on irrelevant queries).
   - Side-by-side comparison of **Legacy Retriever** (FAISS top-k + threshold + message_id dedup) versus **Improved Retriever** (query analyzer + deeper initial candidate pool + metadata reranking).

3. **RAG Answer Generation Evaluator (`rag_eval.py`)**:
   - Evaluates production `retrieve_with_context` + Gemini answer generation against ground truth reference answers:
     * **Answer Correctness**: Agreement with verified facts
     * **Faithfulness / Groundedness**: Strict reliance on retrieved context with zero hallucination
     * **Answer Relevance**: Responsiveness to user intent
     * **Citation Correctness**: Accurate email source references
     * **Abstention Accuracy**: Correct refusal on out-of-domain queries
   - Works both with an LLM-as-Judge (Gemini with explicit 1-5 rubrics) and fully offline via deterministic fact/evidence matching.

4. **BiLSTM Classifier Evaluation (`classifier_eval.py`)**:
   - Evaluates strictly on the held-out test split (542 emails) from `ml/categorization/datasets/email_classification_dataset.parquet` (test_size=0.10, stratified, seed=42).
   - Never evaluates on or mixes training data.
   - Reports 3x3 confusion matrix, per-class Precision/Recall/F1 for `Spam`, `General`, and `Important`, macro-F1, weighted-F1, and accuracy.

5. **End-to-End Reliability Suite (`reliability_eval.py`)**:
   - Email parsing and metadata preservation
   - Classification & spam-handling paths
   - 384-d embedding dimensions and vector-to-message mapping consistency
   - Retrieval and source attribution schemas
   - Ingestion idempotency and duplicate deduplication
   - Gmail / Pub/Sub webhook authentication health
   - Granular latency profiling (retrieval latency strictly isolated from LLM generation latency).

---

## Quickstart: Running Evaluations

Run everything with a single command:

```powershell
python -m evaluation.run --all
```

### Component Subcommands

Run only retrieval evaluation:
```powershell
python -m evaluation.run --retrieval
```

Run only RAG generation evaluation:
```powershell
python -m evaluation.run --rag
```

Run only BiLSTM classifier evaluation:
```powershell
python -m evaluation.run --classifier
```

Run only reliability checks and latency profiling:
```powershell
python -m evaluation.run --reliability
```

Evaluate only the held-out test split:
```powershell
python -m evaluation.run --split test --all
```

Run purely offline without external network or LLM API calls:
```powershell
python -m evaluation.run --offline --all
```

---

## Output Artifacts

All evaluation runs write structured, persistent artifacts to `evaluation/results/`:

| Artifact | Format | Description |
| :--- | :--- | :--- |
| `eval_report_latest.json` | JSON | Complete machine-readable results including all metrics, per-query breakdowns, and latencies |
| `retrieval_per_query.csv` | CSV | Per-query retrieved IDs, scores, relevance grades, and failure diagnoses |
| `EVALUATION_REPORT.md` | Markdown | Comprehensive executive summary with tables, category breakdowns, and error analysis |
| `dashboard.html` | HTML | Interactive, standalone dashboard with metric cards, comparison tables, and tabs (no server needed) |

---

## Benchmark Structure & Schema

The benchmark is stored in `evaluation/benchmark_v1.json`:

```json
{
  "id": "q10",
  "query": "When is my Microsoft assessment?",
  "category": "placement",
  "split": "dev",
  "expected_message_ids": ["19bb5a81a1c389ec", "19bb5a49f6c00e3a", "19b973b78c7602fb"],
  "relevance_judgments": {
    "19bb5a81a1c389ec": {
      "grade": 2,
      "reason": "Direct notification of Microsoft assessment date and test link.",
      "source": "verified"
    },
    "19bb5a49f6c00e3a": {
      "grade": 2,
      "reason": "Reminder / updated instructions for Microsoft online assessment.",
      "source": "verified"
    },
    "19b973b78c7602fb": {
      "grade": 1,
      "reason": "Initial registration and test window announcement for Microsoft.",
      "source": "verified"
    }
  },
  "reference_answer": "Your Microsoft online assessment is scheduled according to the CDC test notification email...",
  "key_evidence": ["Microsoft", "assessment", "test window"],
  "expect_abstention": false
}
```

### Proposing Labels with LLM Judge

To propose candidate relevance judgments for new queries:
```python
from evaluation.benchmark import propose_llm_labels
proposals = propose_llm_labels("query text", candidate_emails)
```
Proposals must be reviewed before freezing into official benchmarks.
