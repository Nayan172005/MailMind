# MailMind

> Intelligent Email Summarization and Information Retrieval System

MailMind is an intelligent email management system that uses Retrieval-Augmented Generation (RAG) to help users search, understand, and interact with their email data through natural language queries.

## Features

- **Semantic Email Search** — Find emails using natural language queries instead of exact keyword matching
- **Question Answering** — Ask questions about your emails and get grounded, cited answers
- **Email Summarization** — Summarize individual emails or groups of related emails
- **Conversational QA** — Ask follow-up questions with context-aware retrieval
- **Email Classification** — BiLSTM-based spam/general/important categorization
- **Grounded Generation** — Answers are strictly based on retrieved email content (no hallucination)

## Architecture

```
Gmail Parquet/JSON/CSV
       │
       ▼
  Email Loader ──────────────────► Load & parse email dataset
       │
       ▼
  Email Normalizer ──────────────► Common internal schema
       │
       ▼
  Document Chunker ──────────────► Split long emails, add metadata headers
       │
       ▼
  Sentence-Transformers ─────────► Local embedding generation
       │                            (all-MiniLM-L6-v2)
       ▼
  FAISS Vector Index ────────────► Persistent similarity search index
       │
       ▼
  Semantic Retriever ────────────► Top-K retrieval + dedup + filtering
       │
       ▼
  Gemini API ────────────────────► Grounded answer generation
       │
       ├──► FastAPI Backend
       ├──► Streamlit Demo
       └──► CLI Interface
```

## Project Structure

```
MailMind/
├── api/                          # FastAPI backend
│   ├── main.py                   # Application entry point
│   └── routes/                   # API route handlers
│       ├── ask.py                # POST /ask
│       ├── search.py             # POST /search
│       ├── summarize.py          # POST /summarize
│       ├── conversation.py       # POST /conversation/ask
│       ├── index.py              # POST /index/build, GET /index/status
│       └── email.py              # GET /email/{id}
│
├── rag/                          # RAG pipeline core
│   ├── config.py                 # Centralized configuration
│   ├── loader.py                 # Email dataset loading
│   ├── normalizer.py             # Email normalization
│   ├── chunker.py                # Document chunking
│   ├── embedder.py               # Sentence-transformer embeddings
│   ├── vector_store.py           # FAISS index management
│   ├── retriever.py              # Semantic retrieval + dedup
│   ├── prompt.py                 # System prompts for grounding
│   ├── generator.py              # Gemini API integration
│   ├── summarizer.py             # Email summarization
│   ├── conversation.py           # Conversational QA
│   ├── pipeline.py               # Pipeline orchestrator
│   ├── schemas.py                # Pydantic data models
│   ├── utils.py                  # Shared utilities
│   └── run.py                    # CLI demo
│
├── streamlit_app/                # Streamlit UI
│   └── app.py
│
├── evaluation/                   # RAG evaluation
│   ├── eval_queries.json         # Test queries
│   └── evaluate.py               # Evaluation script
│
├── tests/                        # Test suite
│   ├── test_loader.py
│   ├── test_normalizer.py
│   ├── test_chunker.py
│   ├── test_embedder.py
│   ├── test_vector_store.py
│   ├── test_retriever.py
│   └── test_pipeline.py
│
├── ml/                           # ML models (existing)
│   └── categorization/           # BiLSTM email classifier
│
├── datasets/                     # Email datasets
│   ├── raw/                      # Original data
│   └── processed/                # Cleaned data
│
├── data/                         # Generated artifacts (gitignored)
│   └── rag/
│       └── index/                # FAISS index + metadata
│
├── .env.example                  # Environment template
├── requirements.txt              # Python dependencies
└── README.md
```

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Environment Setup

```bash
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY
```

### 3. Build the FAISS Index

```bash
python -m rag.run --build-index
```

This loads emails from `datasets/processed/emails/emails_cleaned.parquet`, normalizes, chunks, embeds, and creates the FAISS index at `data/rag/index/`.

### 4. Interactive CLI

```bash
python -m rag.run
```

Commands:
- `/search <query>` — Semantic email search
- `/ask <query>` — Question answering
- `/summarize <query>` — Summarize matching emails
- `/summary <message_id>` — Summarize a specific email
- `/conversation` — Toggle conversation mode
- `/clear` — Clear conversation history
- `/status` — Show index status
- `/quit` — Exit

Or just type a question directly.

### 5. FastAPI Backend

```bash
uvicorn api.main:app --reload
```

API docs available at: `http://localhost:8000/docs`

#### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/search` | Semantic email search |
| POST | `/ask` | Question answering |
| POST | `/summarize` | Email summarization |
| POST | `/conversation/ask` | Conversational QA |
| POST | `/index/build` | Build FAISS index |
| POST | `/index/update` | Update index with new emails |
| GET | `/index/status` | Index status |
| GET | `/email/{message_id}` | Full email details |

### 6. Streamlit Demo

```bash
streamlit run streamlit_app/app.py
```

### 7. Run Tests

```bash
python -m pytest tests/ -v
```

### 8. Run Evaluation

```bash
python evaluation/evaluate.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Google Gemini API key (required) |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `DATASET_PATH` | `datasets/processed/emails/emails_cleaned.parquet` | Email dataset path |
| `INDEX_DIR` | `data/rag/index` | FAISS index directory |
| `TOP_K` | `5` | Default number of results |
| `SIMILARITY_THRESHOLD` | `0.25` | Minimum similarity score |
| `CHUNK_SIZE` | `400` | Words per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `MAX_CONVERSATION_TURNS` | `5` | Conversation history length |

## Dataset

The primary dataset is the VIT student Gmail export:
- **Source**: Gmail API
- **Records**: 10,771 emails
- **Columns**: Message_ID, Thread_ID, Sender_Name, Sender_Email, Subject, Date, Body, Labels, Gmail_Category, etc.
- **Categories**: Primary, Forums, Updates, Promotions, Social

## Technology Stack

| Component | Technology |
|-----------|------------|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS (faiss-cpu) |
| LLM | Google Gemini API |
| Backend | FastAPI |
| Frontend | Streamlit |
| Data Processing | Pandas, PyArrow |
| Email Classifier | TensorFlow/Keras BiLSTM |

## Grounding & Safety

MailMind uses strict grounding to prevent hallucination:

1. **Source of truth**: Only retrieved email content
2. **Empty retrieval**: Returns "not found" instead of guessing
3. **Citation**: Every answer references source emails
4. **Conversation grounding**: Follow-up questions trigger fresh retrieval
5. **No fabrication**: Dates, venues, names, deadlines are never invented

## Limitations

- Embedding model is English-centric
- FAISS is in-memory (suitable for <100K emails)
- No real-time Gmail sync (batch processing)
- Gemini API requires internet and API key
- Chunking uses simple word-based splitting

## Future Improvements

- Real-time Gmail API integration
- BiLSTM classifier integration for category-filtered retrieval
- Hybrid search (semantic + keyword BM25)
- Re-ranking with cross-encoders
- Email thread reconstruction
- React frontend
- User authentication

---

© 2026 Nayan Sharma. All Rights Reserved.