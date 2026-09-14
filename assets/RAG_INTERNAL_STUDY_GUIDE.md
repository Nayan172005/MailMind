# MAILMIND RAG PIPELINE — INTERNAL TECHNICAL STUDY GUIDE

> **CONFIDENTIAL & INTERNAL USE ONLY**  
> *This document is strictly an internal study guide designed to prepare for faculty/panel technical reviews. It explains the exact Retrieval-Augmented Generation (RAG) architecture implemented in the current MailMind repository.*

---

## SECTION 1 — ONE-PAGE BIG PICTURE

### What Happens When a User Enters a Question?

When a user submits a natural language question (e.g., `/ask "When is my Microsoft assessment?"`), the system processes the request through a 14-step pipeline:

```
[1. User Query] ──► [2. RAGPipeline.ask()] ──► [3. retrieve_with_context()]
                                                       │
[6. Top-20 Candidates] ◄── [5. FAISS k-NN Search] ◄── [4. embed_query()]
         │
         ▼
[7. Query Analysis] ──► [8. Metadata Reranker] ──► [9. Top-5 Reranked Context]
                                                            │
[12. Grounded Answer] ◄── [11. Gemini 3.5 LLM] ◄── [10. RAG System Prompt]
         │
         ▼
[13. Source Citations] ──► [14. AskResponse to CLI]
```

### Complete 14-Step Execution Flow

1. **Query Entry**: The user types `/ask "When is my Microsoft assessment?"` into the CLI (`rag/run.py` -> `main()`).
2. **Pipeline Invocation**: `rag/run.py` routes the query string to `pipeline.ask(query)` in `rag/pipeline.py`.
3. **Retrieval Orchestration**: `pipeline.ask()` calls `retrieve_with_context(query, self.store)` in `rag/retriever.py`.
4. **Query Embedding**: `retriever.py` calls `embed_query(query)` in `rag/embedder.py`. The `sentence-transformers/all-MiniLM-L6-v2` model converts the query text into a 384-dimensional dense vector and L2-normalizes it.
5. **FAISS Vector Search**: `retriever.py` calls `store.search(query_vector, top_k=20)` in `rag/vector_store.py`. `faiss.IndexFlatIP` performs inner product nearest-neighbor search across 13,281 chunk vectors.
6. **Initial Candidate Retrieval**: FAISS returns the Top-20 initial raw candidate chunks (`INITIAL_RETRIEVAL_K = 20`) to prevent premature filtering before reranking.
7. **Query Analysis**: `retriever.py` calls `analyze_query(query)` in `rag/query_analyzer.py`. It normalizes text, extracts keywords, detects temporal intent (e.g., `UPCOMING`, `DATE_SEARCH`), and identifies capitalized entities (`Microsoft`).
8. **Metadata-Aware Reranking & Deduplication**: `retriever.py` calls `rerank_candidates(candidates, analysis)` in `rag/reranker.py`. It computes a composite score incorporating semantic similarity (0.65 weight), subject match (0.15), body match (0.05), sender authority (0.05), entity alignment (0.07), and temporal decay (0.08). Simultaneously, it deduplicates candidate chunks sharing the same `message_id`.
9. **Top-K Context Selection**: Candidates scoring below `MIN_RERANK_SCORE = 0.30` are filtered out. The top 5 distinct email contexts (`FINAL_RETRIEVAL_K = 5`) are selected and formatted into a unified text block with metadata headers.
10. **Prompt Construction**: `retriever.py` returns the Top-5 results and formatted context to `pipeline.ask()`, which calls `build_rag_prompt(query, context)` in `rag/prompt.py`. The system prompt strictly instructs the LLM to answer using ONLY retrieved email context and return an uncertainty response if context is missing.
11. **Grounded LLM Generation**: `pipeline.ask()` passes the prompt to `generate_answer(query, context)` in `rag/generator.py`. The Google GenAI SDK (`from google import genai`) calls Gemini (`gemini-2.0-flash` / `gemini-3.5-flash-lite`) to synthesize a grounded answer.
12. **Uncertainty Fallback Check**: If zero candidates pass retrieval/reranking, `retrieve_with_context()` returns empty context. `pipeline.ask()` immediately returns an uncertainty message ("I couldn't find any relevant emails...") WITHOUT calling the Gemini API.
13. **Source Citation Attachment**: `pipeline.ask()` wraps the generated answer string and retrieved Top-5 `SearchResult` objects into an `AskResponse` object.
14. **CLI Output Display**: `rag/run.py` prints the formatted answer and displays clickable source citations (Message ID, Subject, Sender, Date, Relevance Score).

---

## SECTION 2 — FILE-BY-FILE EXPLANATION

### 1. `rag/config.py`
- **Purpose**: Provides centralized configuration parameters and environment variable management for the entire RAG pipeline.
- **What it actually does**: Defines the Pydantic `Settings` class, loading parameters from `.env` with fallback defaults for dataset paths, embedding models, vector dimensions, retrieval limits, reranking weights, and chunking sizes.
- **Important classes/functions**: `class Settings(BaseSettings)`, `settings` (singleton instance), `faiss_index_path`, `metadata_path`.
- **Important variables**:
  - `EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"`
  - `EMBEDDING_DIMENSION = 384`
  - `INITIAL_RETRIEVAL_K = 20`
  - `FINAL_RETRIEVAL_K = 5`
  - `CHUNK_SIZE = 400`
  - `CHUNK_OVERLAP = 50`
  - `MIN_RERANK_SCORE = 0.30`
- **Key Implementation Snippet**:
```python
class Settings(BaseSettings):
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    INITIAL_RETRIEVAL_K: int = 20
    FINAL_RETRIEVAL_K: int = 5
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 50
```
- **Code Explanation**: Pydantic `BaseSettings` automatically reads `.env` key-value pairs if present; otherwise, it assigns these default attributes.
- **Why we use it**: Prevents hardcoding parameters across multiple files, enabling instant tuning of weights, chunk sizes, and API keys.
- **What happens before it**: Python loads the module and reads `.env` via `dotenv.load_dotenv()`.
- **What happens after it**: Imported by `embedder.py`, `chunker.py`, `vector_store.py`, `retriever.py`, `reranker.py`, `generator.py`, and `pipeline.py`.
- **Panel takeaway**: *"Centralized Pydantic settings instance managing all hyper-parameters, vector paths, and model configurations."*

---

### 2. `rag/schemas.py`
- **Purpose**: Defines strongly typed Pydantic data models for internal records, API requests, and responses.
- **What it actually does**: Guarantees data validation across data loading, chunking, retrieval, reranking, and generation.
- **Important classes/functions**: `EmailRecord`, `DocumentChunk`, `SearchResult`, `SearchResponse`, `AskResponse`, `SummarizeResponse`, `ConversationResponse`.
- **Key Implementation Snippet**:
```python
class DocumentChunk(BaseModel):
    chunk_id: int
    message_id: str
    subject: str
    sender_name: str
    sender_email: str
    date: str
    text: str
    body_snippet: str
```
- **Code Explanation**: Represents a single vector-indexed chunk containing metadata fields + header-prepend body text + short snippet for preview.
- **Why we use it**: Ensures type safety and validation between pipeline components.
- **What happens before it**: Raw data loaded from Parquet/JSON.
- **What happens after it**: Consumed by `chunker.py` and stored in `vector_store.py`.
- **Panel takeaway**: *"Defines data schemas like `DocumentChunk` and `SearchResult` to maintain type safety throughout RAG execution."*

---

### 3. `rag/loader.py`
- **Purpose**: Reads raw email datasets from Parquet, JSON, or CSV files into memory.
- **What it actually does**: Resolves file paths, checks format extensions (`.parquet`, `.json`, `.csv`), and returns a list of raw email dictionaries.
- **Important classes/functions**: `load_emails(path: Optional[str] = None) -> list[dict]`.
- **Key Implementation Snippet**:
```python
def load_emails(path: Optional[str] = None) -> list[dict]:
    target_path = Path(path) if path else settings.dataset_abs_path
    if target_path.suffix == ".parquet":
        df = pd.read_parquet(target_path)
        return df.to_dict(orient="records")
```
- **Code Explanation**: Uses pandas to read `.parquet` files efficiently and converts rows into standard Python dictionary records.
- **Why we use it**: Decouples dataset storage format from pipeline processing logic.
- **What happens before it**: `pipeline.build_index()` is invoked.
- **What happens after it**: Raw dicts are passed to `normalizer.py`.
- **Panel takeaway**: *"File loader using pandas to ingest parquet email datasets into raw dictionaries."*

---

### 4. `rag/normalizer.py`
- **Purpose**: Cleans and standardizes heterogeneous raw email fields into clean `EmailRecord` objects.
- **What it actually does**: Strips HTML tags, normalizes whitespace, parses list fields (recipients), standardizes date formats, and fallback-populates missing fields.
- **Important classes/functions**: `normalize_email(raw: dict) -> EmailRecord`, `normalize_emails(raw_records: list[dict]) -> list[EmailRecord]`.
- **Key Implementation Snippet**:
```python
def normalize_email(raw: dict) -> EmailRecord:
    body = clean_whitespace(safe_str(raw.get("body", "")))
    return EmailRecord(
        message_id=safe_str(raw.get("message_id", "")),
        subject=clean_whitespace(safe_str(raw.get("subject", ""))),
        sender_name=safe_str(raw.get("sender_name", "")),
        sender_email=safe_str(raw.get("sender_email", "")),
        date=safe_str(raw.get("date", "")),
        body=body,
    )
```
- **Code Explanation**: Extracts raw dictionary keys safely, cleans whitespace, and returns validated `EmailRecord` instances.
- **Why we use it**: Prevents clean-up bugs caused by missing values, dirty HTML, or malformed email headers.
- **What happens before it**: `loader.load_emails()`.
- **What happens after it**: Normalized records passed to `chunker.py`.
- **Panel takeaway**: *"Normalizes dirty raw email fields into clean `EmailRecord` instances."*

---

### 5. `rag/chunker.py`
- **Purpose**: Converts normalized emails into searchable document chunks prepended with metadata headers.
- **What it actually does**: Evaluates body word count against `CHUNK_SIZE = 400` words. Short emails produce 1 chunk (`chunk_id=0`). Long emails are split into overlapping word windows of 400 words with 50-word overlap (`CHUNK_OVERLAP = 50`). Prepends Subject, Sender, and Date headers to EVERY chunk text.
- **Important classes/functions**: `_build_metadata_header()`, `_split_text_into_chunks()`, `chunk_email()`, `chunk_emails()`.
- **Key Implementation Snippet**:
```python
def _build_metadata_header(email: EmailRecord) -> str:
    parts = []
    if email.subject:
        parts.append(f"Subject: {email.subject}")
    if email.sender_name or email.sender_email:
        sender = email.sender_name or email.sender_email
        parts.append(f"From: {sender}")
    if email.date:
        parts.append(f"Date: {email.date}")
    return "\n".join(parts)
```
- **Code Explanation**: Constructs a text metadata block prepended to every chunk so dense vector embeddings capture sender/subject intent alongside body text.
- **Why we use it**: Long emails exceed embedding model token limits (512 tokens); chunking preserves local context while metadata prepending prevents loss of email origin.
- **What happens before it**: `normalizer.normalize_emails()`.
- **What happens after it**: `vector_store.build_index()`.
- **Panel takeaway**: *"Splits long emails into 400-word chunks (50-word overlap) prepended with metadata headers."*

---

### 6. `rag/embedder.py`
- **Purpose**: Generates 384-dimensional dense vector embeddings using SentenceTransformers.
- **What it actually does**: Loads `sentence-transformers/all-MiniLM-L6-v2` lazily once in a global singleton `_model`. Provides `embed_texts()` for batch chunk embedding and `embed_query()` for single query embedding, enforcing `normalize_embeddings=True` (L2 normalization).
- **Important classes/functions**: `get_model()`, `embed_texts()`, `embed_query()`.
- **Key Implementation Snippet**:
```python
def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)
```
- **Code Explanation**: Uses SentenceTransformer `encode()` to compute unit-length 384-d vectors (`normalize_embeddings=True`), allowing Inner Product distance to equal Cosine Similarity.
- **Why we use it**: Enables fast semantic vector search over email content.
- **What happens before it**: `chunker.chunk_emails()` or user query submission.
- **What happens after it**: Vectors added to `faiss.IndexFlatIP` in `vector_store.py`.
- **Panel takeaway**: *"Generates L2-normalized 384-d dense embeddings using `all-MiniLM-L6-v2`."*

---

### 7. `rag/vector_store.py`
- **Purpose**: Manages in-memory FAISS similarity search and disk persistence (`emails.faiss` + `metadata.json`).
- **What it actually does**: Initializes `faiss.IndexFlatIP(384)`. Adds normalized chunk vectors, maintains a `metadata` list mapping vector row indices to `DocumentChunk` dicts, and provides disk `save()` / `load()` methods.
- **Important classes/functions**: `class VectorStore`, `build_index()`, `update_index()`, `search()`, `save()`, `load()`.
- **Key Implementation Snippet**:
```python
def build_index(self, chunks: list[DocumentChunk]) -> None:
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)
    self.index = faiss.IndexFlatIP(settings.EMBEDDING_DIMENSION)
    faiss.normalize_L2(vectors)
    self.index.add(vectors)
    self.metadata = [c.model_dump() for c in chunks]
```
- **Code Explanation**: Creates a flat inner-product FAISS index, normalizes vector L2 norms, adds vectors to FAISS, and stores chunk metadata.
- **Why we use it**: Enables sub-20ms k-nearest neighbor similarity search across 13,281 vectors.
- **What happens before it**: `embedder.embed_texts()`.
- **What happens after it**: Consumed by `retriever.py` during query search.
- **Panel takeaway**: *"FAISS `IndexFlatIP` storing 13,281 L2-normalized 384-d vectors with JSON metadata."*

---

### 8. `rag/query_analyzer.py`
- **Purpose**: Extracts structural query signals (keywords, temporal intent, entities) from natural language user queries.
- **What it actually does**: Normalizes text, tokenizes, detects temporal intent (`UPCOMING`, `PAST`, `RECENT`, `DATE_SEARCH`, `MONTH_SEARCH`), and extracts capitalized entity terms.
- **Important classes/functions**: `analyze_query(query: str) -> QueryAnalysis`, `_detect_temporal_intent()`, `_extract_entities()`.
- **Key Implementation Snippet**:
```python
def analyze_query(query: str) -> QueryAnalysis:
    norm = normalize_text(query)
    tokens = tokenize(norm, keep_stopwords=False)
    temp_intent = _detect_temporal_intent(norm)
    entities = _extract_entities(query)
    return QueryAnalysis(
        raw_query=query,
        normalized_query=norm,
        tokens=tokens,
        temporal_intent=temp_intent,
        entities=entities,
    )
```
- **Code Explanation**: Parses raw query string into a structured `QueryAnalysis` object containing tokens, temporal intent enum, and entity strings.
- **Why we use it**: Provides structured features for metadata reranking (e.g., date closeness, sender matching).
- **What happens before it**: Query submitted to `retriever.retrieve()`.
- **What happens after it**: Passed to `reranker.rerank_candidates()`.
- **Panel takeaway**: *"Analyzes user queries to extract keywords, temporal intent, and capitalized entities."*

---

### 9. `rag/reranker.py`
- **Purpose**: Scores and reranks initial candidate chunks using multi-signal metadata weights and deduplicates message IDs.
- **What it actually does**: Calculates composite scores combining semantic cosine similarity (0.65 weight), subject match (0.15), body match (0.05), sender authority (0.05), entity alignment (0.07), and temporal recency/closeness (0.08). Filters candidates below `MIN_RERANK_SCORE = 0.30` and keeps only the highest-scoring chunk per distinct `message_id`.
- **Important classes/functions**: `rerank_candidates()`, `_base_score()`, `_temporal_scores()`, `_lexical_score()`, `_entity_score()`.
- **Key Implementation Snippet**:
```python
final_score = (
    semantic * settings.RERANK_SEMANTIC_WEIGHT
    + subject * settings.RERANK_SUBJECT_WEIGHT
    + body * settings.RERANK_BODY_WEIGHT
    + sender * settings.RERANK_SENDER_WEIGHT
    + entity * settings.RERANK_ENTITY_WEIGHT
    + temporal * settings.RERANK_TEMPORAL_WEIGHT
)
```
- **Code Explanation**: Multiplies normalized individual signal scores (0.0 to 1.0) by their configured weights and sums them to compute a final rerank score.
- **Why we use it**: Pure vector similarity ignores recency and sender authority; reranking boosts relevant emails to top ranks.
- **What happens before it**: `vector_store.search()` retrieves Top-20 initial candidates.
- **What happens after it**: Returns Top-5 reranked candidates to `retriever.py`.
- **Panel takeaway**: *"Multi-signal reranker combining semantic similarity, subject match, sender authority, entity alignment, and temporal signals."*

---

### 10. `rag/retriever.py`
- **Purpose**: High-level retrieval orchestrator connecting embedding search, candidate expansion, deduplication, and reranking.
- **What it actually does**: Retrieves `INITIAL_RETRIEVAL_K = 20` candidates via FAISS vector search, executes query analysis, calls metadata reranker, filters by threshold, and returns Top-5 final `SearchResult` objects + formatted context text block.
- **Important classes/functions**: `retrieve()`, `retrieve_with_context()`, `_initial_k()`.
- **Key Implementation Snippet**:
```python
def retrieve_with_context(query: str, store: VectorStore, top_k: int | None = None):
    results = retrieve(query, store, top_k=top_k)
    if not results:
        return [], ""
    context_blocks = []
    for r in results:
        block = f"--- EMAIL (ID: {r.message_id}) ---\nSubject: {r.subject}\nFrom: {r.sender}\nDate: {r.date}\nContent:\n{r.snippet}"
        context_blocks.append(block)
    return results, "\n\n".join(context_blocks)
```
- **Code Explanation**: Calls `retrieve()` to get reranked results and formats top results into formatted context blocks for LLM consumption.
- **Why we use it**: Provides a clean interface for pipeline and generator modules.
- **What happens before it**: Called by `pipeline.ask()` or `pipeline.search()`.
- **What happens after it**: Context block passed to `prompt.py` and `generator.py`.
- **Panel takeaway**: *"Orchestrates candidate retrieval (Top-20 -> Top-5) and formats email context blocks for Gemini."*

---

### 11. `rag/prompt.py`
- **Purpose**: Builds system prompts for grounded question answering, summarization, and conversation contextualization.
- **What it actually does**: Formats templates that strictly instruct Gemini to answer using ONLY retrieved email context and return uncertainty responses if context lacks evidence.
- **Important classes/functions**: `build_rag_prompt()`, `build_summary_prompt()`, `build_conversation_prompt()`, `build_contextualize_prompt()`.
- **Key Implementation Snippet**:
```python
def build_rag_prompt(query: str, context: str) -> str:
    return f"Answer the user's question using ONLY the provided email context below.\nIf the context does not contain sufficient information to answer, state clearly that the emails do not contain the answer. Do NOT invent facts.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
```
- **Code Explanation**: Injects formatted email context and user question into a strict grounding prompt template.
- **Why we use it**: Prevents LLM hallucination by restricting generation boundaries to retrieved evidence.
- **What happens before it**: `retriever.retrieve_with_context()`.
- **What happens after it**: Prompt string sent to `generator.generate_answer()`.
- **Panel takeaway**: *"Defines strict grounding prompts that constrain Gemini to answer only from retrieved email context."*

---

### 12. `rag/generator.py`
- **Purpose**: Interfaces with the Google GenAI SDK (`google-genai`) to generate answers and summaries via Gemini.
- **What it actually does**: Initializes `genai.Client(api_key=...)` using module singleton `_get_client()`. Sends prompts to Gemini (`gemini-2.0-flash` / `gemini-3.5-flash-lite`) and returns text responses.
- **Important classes/functions**: `_get_client()`, `generate_answer()`, `generate_summary()`.
- **Key Implementation Snippet**:
```python
from google import genai

def generate_answer(query: str, context: str, model: str | None = None) -> str:
    client = _get_client()
    prompt = build_rag_prompt(query, context)
    target_model = model or settings.GEMINI_MODEL
    response = client.models.generate_content(
        model=target_model,
        contents=prompt,
    )
    return response.text.strip()
```
- **Code Explanation**: Uses the official `google-genai` SDK (`genai.Client`) to invoke Gemini's `generate_content()` method with the grounded prompt.
- **Why we use it**: Provides state-of-the-art natural language generation.
- **What happens before it**: `prompt.build_rag_prompt()`.
- **What happens after it**: Answer text wrapped in `AskResponse` and returned to user.
- **Panel takeaway**: *"Invokes Gemini using the modern `google-genai` SDK for grounded text synthesis."*

---

### 13. `rag/conversation.py`
- **Purpose**: Manages multi-turn conversation sessions and contextualizes follow-up queries.
- **What it actually does**: Tracks session history up to `MAX_CONVERSATION_TURNS = 5`. For follow-up queries (e.g., *"What about VISA?"*), it uses Gemini to rephrase the query into a standalone search string (e.g., *"When is my VISA online assessment?"*), then executes FRESH retrieval against FAISS.
- **Important classes/functions**: `class ConversationSession`, `class ConversationManager`, `ask()`.
- **Key Implementation Snippet**:
```python
if session.has_history():
    history_str = session.format_history()
    ctx_prompt = build_contextualize_prompt(query, history_str)
    search_query = generate_raw(ctx_prompt)
else:
    search_query = query
```
- **Code Explanation**: If previous interaction turns exist, asks Gemini to resolve coreferences into a standalone search query before running vector search.
- **Why we use it**: Enables natural multi-turn follow-up questions without corrupting retrieval with old answer text.
- **What happens before it**: User calls `/conversation` in CLI.
- **What happens after it**: Fresh retrieval executed using `search_query`.
- **Panel takeaway**: *"Handles multi-turn follow-ups by contextualizing queries into standalone search strings for fresh FAISS retrieval."*

---

### 14. `rag/summarizer.py`
- **Purpose**: Generates summaries for single emails or groups of emails matching a query.
- **What it actually does**:
  - Mode A (`summarize_single_email`): Looks up email by `message_id` in `_emails_lookup` and calls `generate_summary(context, mode="single")`.
  - Mode B (`summarize_by_query`): Executes retrieval for `query`, formats top matching emails, and calls `generate_summary(context, mode="query")`.
- **Important classes/functions**: `summarize_single_email()`, `summarize_by_query()`.
- **Key Implementation Snippet**:
```python
def summarize_by_query(query: str, store: VectorStore, top_k: int | None = None):
    results, context = retrieve_with_context(query, store, top_k=top_k)
    summary = generate_summary(context, mode="query")
    return SummarizeResponse(summary=summary, sources=results, mode="query")
```
- **Code Explanation**: Retrieves relevant emails for a topic query, builds context, and invokes Gemini summary generation.
- **Why we use it**: Allows users to quickly digest long threads or topic-related email groups.
- **What happens before it**: User calls `/summarize <query>` or `/summary <message_id>`.
- **What happens after it**: Returns `SummarizeResponse` with executive summary bullets + source citations.
- **Panel takeaway**: *"Supports single-email summarization by Message ID and group-email summarization by query."*

---

### 15. `rag/pipeline.py`
- **Purpose**: Main facade orchestrator exposing clean API methods for searching, asking, summarizing, and conversation management.
- **What it actually does**: Instantiates `VectorStore` and `ConversationManager`, loads/builds index, and provides `search()`, `ask()`, `summarize()`, and `conversation_ask()` endpoints.
- **Important classes/functions**: `class RAGPipeline`, `build_index()`, `load_index()`, `search()`, `ask()`, `summarize()`, `conversation_ask()`.
- **Key Implementation Snippet**:
```python
class RAGPipeline:
    def __init__(self):
        self.store = VectorStore()
        self.conversation_manager = ConversationManager()
```
- **Code Explanation**: Encapsulates pipeline components behind a clean facade object used by FastAPI (`api/`) and CLI (`rag/run.py`).
- **Why we use it**: Provides a single unified entry point for all RAG system operations.
- **What happens before it**: Instantiated on system startup.
- **What happens after it**: Directs flow to retriever, vector store, and generator.
- **Panel takeaway**: *"Central facade class connecting vector store, retriever, generator, and conversation manager."*

---

### 16. `rag/run.py`
- **Purpose**: Command-Line Interface (CLI) application for interactive user querying and index management.
- **What it actually does**: Parses command-line flags (`--build-index`, `--force-rebuild`), initializes `RAGPipeline`, loads FAISS index, and runs an interactive shell supporting `/search`, `/ask`, `/summarize`, `/summary`, `/conversation`, `/clear`, `/status`, `/quit`.
- **Important classes/functions**: `main()`, `run_interactive()`.
- **Key Implementation Snippet**:
```python
elif user_input.startswith("/ask "):
    q = user_input[5:].strip()
    res = pipeline.ask(q)
    print(f"\nAnswer:\n{res.answer}\n")
    print_results(res.sources)
```
- **Code Explanation**: Routes CLI commands to corresponding pipeline methods and prints formatted answers + source metadata.
- **Why we use it**: Provides an instant, lightweight terminal interface for testing and reviewing system capabilities.
- **What happens before it**: Executed via `python -m rag.run`.
- **What happens after it**: Calls `pipeline.ask()`, `pipeline.search()`, etc.
- **Panel takeaway**: *"CLI shell routing commands like `/ask` and `/search` to the `RAGPipeline` facade."*

---

### 17. `rag/benchmark_latency.py`
- **Purpose**: Benchmarks warm retrieval latency across 10 sample queries.
- **What it actually does**: Loads FAISS index, performs 3 warm-up queries, measures execution time for 10 benchmark queries using `time.perf_counter()`, and computes Min, Max, Mean, Median (P50), and P95 latency stats.
- **Important classes/functions**: `percentile()`, benchmark loop.
- **Key Implementation Snippet**:
```python
for query in QUERIES:
    start = time.perf_counter()
    response = pipeline.search(query)
    end = time.perf_counter()
    latency_ms = (end - start) * 1000
    latencies.append(latency_ms)
```
- **Code Explanation**: Measures exact warm retrieval execution time (vector search + query analysis + reranking) in milliseconds.
- **Why we use it**: Provides empirical latency benchmarks for technical evaluation.
- **What happens before it**: Index loaded and warm-up queries executed.
- **What happens after it**: Prints summary table (P95 < 200ms verification).
- **Panel takeaway**: *"Retrieval latency benchmark script verifying sub-20ms warm search times."*

---

## SECTION 3 — KEY CODE LINES

### 1. Dataset Loading (`rag/loader.py`)
```python
df = pd.read_parquet(target_path)
return df.to_dict(orient="records")
```
- **What it does**: Reads cleaned emails from a Parquet file into a Pandas DataFrame and converts rows to dictionary records.
- **Why needed**: Parquet is a columnar binary format; converting to dicts allows downstream Pydantic schema validation.
- **If removed/changed**: Dataset cannot be ingested, breaking index construction.

### 2. Email Normalization (`rag/normalizer.py`)
```python
body = clean_whitespace(safe_str(raw.get("body", "")))
return EmailRecord(message_id=..., subject=..., body=body, ...)
```
- **What it does**: Cleans dirty HTML/whitespace and wraps raw dictionary keys into a validated `EmailRecord`.
- **Why needed**: Guarantees consistent string types and non-null fields.
- **If removed/changed**: Missing dictionary keys or HTML markup cause crashes during chunking and embedding.

### 3. Chunking (`rag/chunker.py`)
```python
body_chunks = _split_text_into_chunks(body, chunk_size, chunk_overlap)
full_text = f"{header}\n\nBody:\n{body_chunk}"
```
- **What it does**: Splits long email body into 400-word windows with 50-word overlap, prepending Subject/Sender/Date metadata header to every chunk.
- **Why needed**: Prevents exceeding embedding model token limits and ensures embeddings capture metadata context.
- **If removed/changed**: Truncates long emails or loses sender/subject context during vector search.

### 4. Embedding Generation (`rag/embedder.py`)
```python
embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
```
- **What it does**: Converts chunk text strings into 384-dimensional dense vectors normalized to L2 unit length.
- **Why needed**: Dense vectors capture semantic similarity beyond exact keywords; L2 normalization makes Inner Product equal Cosine Distance.
- **If removed/changed**: Raw text cannot be searched in vector space.

### 5. FAISS Index Creation (`rag/vector_store.py`)
```python
self.index = faiss.IndexFlatIP(settings.EMBEDDING_DIMENSION)
faiss.normalize_L2(vectors)
self.index.add(vectors)
```
- **What it does**: Creates an exact inner product FAISS index and populates it with 13,281 normalized chunk vectors.
- **Why needed**: Enables sub-millisecond similarity search across the entire email corpus.
- **If removed/changed**: Vector similarity queries cannot be executed.

### 6. FAISS Search (`rag/vector_store.py`)
```python
scores, indices = self.index.search(query_vector, top_k)
```
- **What it does**: Performs nearest-neighbor search in FAISS, returning the top-k highest inner product similarity scores and row indices.
- **Why needed**: Fast k-NN retrieval of candidate chunks matching the query vector.
- **If removed/changed**: Search fails or returns random chunks.

### 7. Candidate Retrieval Expansion (`rag/retriever.py`)
```python
raw_results = store.search(query_embedding, top_k=settings.INITIAL_RETRIEVAL_K)
```
- **What it does**: Fetches Top-20 initial candidates (`INITIAL_RETRIEVAL_K = 20`) from FAISS before deduplication and reranking.
- **Why needed**: Prevents premature filtering so the reranker has a diverse pool to evaluate recency and subject match.
- **If removed/changed**: Top candidate pool is too small, lowering Recall@5.

### 8. Deduplication (`rag/reranker.py`)
```python
if message_id in seen_messages:
    continue
seen_messages.add(message_id)
```
- **What it does**: Tracks seen `message_id`s during candidate iteration, keeping only the highest-scoring chunk per unique email.
- **Why needed**: Prevents returning 5 chunks of the exact same email thread in the final Top-5 context.
- **If removed/changed**: Context window is wasted on redundant chunks from a single email.

### 9. Metadata-Aware Reranking (`rag/reranker.py`)
```python
final_score = (
    semantic * 0.65 + subject * 0.15 + body * 0.05 +
    sender * 0.05 + entity * 0.07 + temporal * 0.08
)
```
- **What it does**: Combines semantic vector score with subject, body, sender, entity, and temporal signals into a weighted final score.
- **Why needed**: Vector similarity alone ignores document recency and sender authority.
- **If removed/changed**: Old emails or irrelevant senders rank above recent urgent announcements.

### 10. Top-K Selection (`rag/reranker.py`)
```python
ranked_candidates = [c for c in deduplicated if c["final_score"] >= settings.MIN_RERANK_SCORE]
return ranked_candidates[:top_k]
```
- **What it does**: Filters out candidates scoring below 0.30 threshold and selects the Top-5 reranked results.
- **Why needed**: Restricts context to high-confidence relevant emails.
- **If removed/changed**: Low-quality noise emails are passed to Gemini.

### 11. Context Construction (`rag/retriever.py`)
```python
block = f"--- EMAIL (ID: {r.message_id}) ---\nSubject: {r.subject}\nFrom: {r.sender}\nDate: {r.date}\nContent:\n{r.snippet}"
```
- **What it does**: Formats reranked `SearchResult` objects into clean text context blocks with explicit metadata labels.
- **Why needed**: Provides Gemini with structured, readable context for answering questions.
- **If removed/changed**: Gemini receives unstructured text without metadata context.

### 12. Gemini API Call (`rag/generator.py`)
```python
client = _get_client()
response = client.models.generate_content(model=target_model, contents=prompt)
```
- **What it does**: Sends the grounded prompt to Gemini using the `google-genai` SDK and retrieves generated text.
- **Why needed**: Generates natural language answers backed by email evidence.
- **If removed/changed**: System cannot synthesize natural language answers.

### 13. Grounding System Prompt (`rag/prompt.py`)
```python
return f"Answer the user's question using ONLY the provided email context below. If the context does not contain sufficient information, state clearly that the emails do not contain the answer."
```
- **What it does**: Instructs Gemini to restrict generation to retrieved context and refuse unsupported questions.
- **Why needed**: Mitigates LLM hallucination and enforces evidence-grounded answers.
- **If removed/changed**: LLM generates fabricated personal details or external knowledge.

### 14. Conversation Context (`rag/conversation.py`)
```python
ctx_prompt = build_contextualize_prompt(query, history_str)
search_query = generate_raw(ctx_prompt)
```
- **What it does**: Uses Gemini to rephrase multi-turn follow-up queries (e.g., *"What about VISA?"*) into standalone search strings.
- **Why needed**: Allows fresh FAISS retrieval without polluting search with previous answer text.
- **If removed/changed**: Follow-up questions fail during vector search.

### 15. Summarization (`rag/summarizer.py`)
```python
def summarize_by_query(query: str, store: VectorStore, top_k: int | None = None):
    results, context = retrieve_with_context(query, store, top_k=top_k)
    summary = generate_summary(context, mode="query")
```
- **What it does**: Retrieves matching emails for a topic query and generates executive summary bullet points.
- **Why needed**: Allows users to quickly summarize email groups or threads.
- **If removed/changed**: Group email summarization capability is unavailable.

### 16. Pipeline Orchestration (`rag/pipeline.py`)
```python
def ask(self, query: str, top_k: int | None = None) -> AskResponse:
    results, context = retrieve_with_context(query, self.store, top_k=top_k)
    if not results:
        return AskResponse(query=query, answer="I couldn't find any relevant emails...", sources=[])
    answer = generate_answer(query, context)
    return AskResponse(query=query, answer=answer, sources=results)
```
- **What it does**: Orchestrates retrieval, empty-result checking, answer generation, and response wrapping.
- **Why needed**: Acts as the main facade method used by APIs and CLI.
- **If removed/changed**: External components cannot execute question answering.

---

## SECTION 4 — CHUNKING

### Detailed Chunking Mechanics (`rag/chunker.py`)

- **What constitutes a chunk?**: A `DocumentChunk` object containing metadata header lines + a body snippet of up to 400 words.
- **Fields included in header**:
  - `Subject: <subject>`
  - `From: <sender_name or sender_email>`
  - `Date: <date>`
- **Fields combined**: The metadata header is prepended to the body text string before embedding generation.
- **Chunk Parameters (from `rag/config.py`)**:
  - `CHUNK_SIZE = 400` words (approximate words per chunk).
  - `CHUNK_OVERLAP = 50` words (overlap between consecutive chunks).
- **Short Email Strategy**: If an email body contains <= 400 words, it produces **1 single chunk** (`chunk_id=0`).
- **Long Email Strategy**: If an email body contains > 400 words, it is split into overlapping word windows of 400 words (moving by 400 - 50 = 350 words per step). Every resulting chunk retains the full metadata header prepended at the top.
- **Why this approach was chosen**:
  - Embedding models (`all-MiniLM-L6-v2`) have a 512-token input limit. Chunking prevents text truncation.
  - Prepended headers ensure that even if a body chunk appears halfway through a long email, the vector embedding retains subject and sender intent.
- **Reconnecting chunks to original email**: Every chunk retains the original `message_id`. When retrieved, deduplication groups chunks by `message_id`, and `pipeline.get_email(message_id)` can reassemble full email details.
- **Why 13,281 vectors vs 10,771 unique emails?**:
  - There are 10,771 unique raw emails in the dataset.
  - Short emails produce 1 vector.
  - Long emails produce multiple chunk vectors (e.g., a 1,200-word circular produces 3-4 chunks).
  - Therefore, 10,771 unique emails expand into **13,281 total vector chunks** in the FAISS index.

---

## SECTION 5 — EMBEDDINGS

### Vector Embedding Mechanics (`rag/embedder.py`)

- **Model used**: `sentence-transformers/all-MiniLM-L6-v2`.
- **Why this model?**: Lightweight (22.9 Million parameters, ~90 MB model size), extremely fast CPU inference (~15ms per batch), fine-tuned via Siamese networks on over 1 Billion sentence pairs for semantic similarity tasks.
- **Embedding dimension**: **384 dimensions**. Every text input is mapped to a 384-element float32 vector array.
- **What is embedded?**: The complete `full_text` of each chunk (Metadata Header + Body snippet).
- **Embedding normalization**: `normalize_embeddings=True` in `model.encode()`. Normalizes vectors to L2 unit length (L2 norm = 1.0).
- **Significance of L2 normalization**: For unit vectors, **Inner Product = Cosine Similarity**. This allows `faiss.IndexFlatIP` to compute exact cosine similarity directly.
- **Model loading strategy**: Cached in a module-level singleton `_model` inside `embedder.py`. It is loaded lazily on the first `embed_texts()` or `embed_query()` call and retained in memory for all subsequent queries.
- **Relationship between embedding dimension and FAISS**: `faiss.IndexFlatIP(384)` requires the vector dimension to match `EMBEDDING_DIMENSION = 384` exactly.

---

## SECTION 6 — FAISS

### Vector Index Mechanics (`rag/vector_store.py`)

- **Index type**: `faiss.IndexFlatIP(384)` (Flat Inner Product index).
- **Why selected?**: Performs exact brute-force nearest-neighbor search without approximation loss. For datasets under 100,000 vectors (13,281 vectors here), `IndexFlatIP` provides 100% search accuracy at sub-20ms latency on standard CPU hardware.
- **Similarity metric**: Cosine similarity (achieved via L2-normalized vectors + Inner Product).
- **Index construction**:
  1. `IndexFlatIP(384)` instantiated.
  2. Embeddings generated for 13,281 chunks.
  3. `faiss.normalize_L2(vectors)` called.
  4. `index.add(vectors)` adds vectors to FAISS.
  5. Metadata dictionaries dumped to `metadata.json`.
- **Index persistence**:
  - Binary vector file: `data/rag/index/emails.faiss`
  - JSON metadata file: `data/rag/index/metadata.json`
- **Loading index**: `load()` reads `emails.faiss` using `faiss.read_index()` and loads `metadata.json` into memory.
- **Vector search concept**: Converts query text to 384-d unit vector, performs matrix multiplication against the 13,281 x 384 index matrix, and returns row indices with highest dot-product scores.

---

## SECTION 7 — RETRIEVAL PIPELINE

### Candidate Expansion & Filter Flow (`rag/retriever.py`)

```
User Query ──► embed_query() ──► FAISS Search (Top-20) ──► Query Analysis ──► Reranking & Dedup ──► Top-5 Reranked Context
```

- **Parameters (from `rag/config.py`)**:
  - `INITIAL_RETRIEVAL_K = 20`: Number of initial candidates fetched from FAISS.
  - `FINAL_RETRIEVAL_K = 5`: Number of top reranked chunks returned to context.
  - `SIMILARITY_THRESHOLD = 0.25`: Initial raw vector score cutoff.
  - `MIN_RERANK_SCORE = 0.30`: Final rerank score cutoff.
- **Why retrieve Top-20 first?**: Pure vector search ranks chunks purely by semantic similarity. If we only fetched Top-5 from FAISS, a highly relevant recent announcement might rank at position 8 due to low vocabulary overlap and be missed. By expanding initial retrieval to Top-20, we allow the metadata reranker to evaluate recency, subject match, and sender authority, bringing critical emails into the final Top-5 context.

---

## SECTION 8 — METADATA-AWARE RERANKING

### Multi-Signal Scoring System (`rag/reranker.py`)

The reranker scores each candidate using 6 weighted signals defined in `rag/config.py`:

| Signal Name | Weight | Calculation Method | Why It Helps Email Retrieval |
| :--- | :--- | :--- | :--- |
| **Semantic Score** | `0.65` | Raw FAISS inner product score (max(0.0, min(1.0, score))) | Captures deep semantic meaning and natural language intent. |
| **Subject Match** | `0.15` | Lexical term coverage + phrase presence in `subject` header | Email subjects contain discriminative intent (e.g., *"Microsoft Assessment"*). |
| **Body Match** | `0.05` | Lexical term coverage in `body_snippet` | Reinforces exact keyword matching in email bodies. |
| **Sender Authority** | `0.05` | Matches query tokens against `sender_name` / `sender_email` | Boosts official senders (e.g., placement cell, faculty). |
| **Entity Alignment**| `0.07` | Capitalized entity term overlap in candidate text | Ensures specific company/course names (e.g., *"VISA"*, *"NPTEL"*) align. |
| **Temporal Signal** | `0.08` | Exponential date closeness / recency decay calculation | Prefers recent communications over historical entries when intent demands. |

### Reranking Formula

```
final_score = (semantic * 0.65) + (subject * 0.15) + (body * 0.05) + (sender * 0.05) + (entity * 0.07) + (temporal * 0.08)
```

---

## SECTION 9 — DEDUPLICATION

### Chunk & Thread Deduplication (`rag/reranker.py`)

- **Why duplicates occur**:
  1. Long emails produce multiple chunks (`chunk_id=0`, `chunk_id=1`).
  2. Email reply threads contain quoted text from previous emails.
- **How deduplication works**:
```python
seen_messages = set()
deduplicated = []
for candidate in scored_candidates:
    msg_id = candidate.get("message_id", "")
    if msg_id in seen_messages:
        continue
    seen_messages.add(msg_id)
    deduplicated.append(candidate)
```
- **Stage performed**: Applied immediately after rerank scoring and before final Top-5 slicing.
- **Result**: Guarantees that the Top-5 context passed to Gemini represents **5 distinct emails** rather than redundant chunks from a single message.

---

## SECTION 10 — GEMINI / GENERATION

### Generation Tier Mechanics (`rag/generator.py`)

- **Configured Model**: `GEMINI_MODEL = "gemini-2.0-flash"` in `config.py` (customizable via `.env` or `gemini-3.5-flash-lite`).
- **SDK Import**: `from google import genai` (`google-genai` SDK).
- **Client Initialization**: `genai.Client(api_key=settings.GEMINI_API_KEY)`.
- **API Key Loading**: Loaded lazily from `.env` via `settings.GEMINI_API_KEY`.
- **Information Sent to Gemini**: Grounded system prompt containing query + formatted Top-5 email context blocks.
- **Empty Retrieval Handling**: If `retrieve_with_context()` returns zero candidates, `pipeline.ask()` returns an uncertainty message immediately **without calling the Gemini API**, saving API quota and eliminating hallucination risk.

---

## SECTION 11 — GROUNDED GENERATION

### Grounded Answer vs. Standard LLM QA

- **Standard LLM QA**: Answers from parametric weights, leading to hallucinated personal dates or commitments.
- **MailMind Grounded RAG**: Answers strictly from non-parametric email context.
- **System Prompt Instructions (`rag/prompt.py`)**:
```python
def build_rag_prompt(query: str, context: str) -> str:
    return f"""Answer the user's question using ONLY the provided email context below.
If the context does not contain sufficient information to answer, state clearly that the emails do not contain the answer. Do NOT invent facts.

Context:
{context}

Question: {query}
Answer:"""
```
- **Unsupported Question Example**:
  - Query: *"Is there a VISA online test scheduled in December 2027?"*
  - Context: Emails contain tests for 2025 only.
  - Behavior: Gemini responds: *"Based on the provided emails, there is no VISA online test scheduled in December 2027."*

---

## SECTION 12 — CONVERSATION MODE

### Multi-Turn Dialogue (`rag/conversation.py`)

- **Session Tracking**: `ConversationSession` stores history up to `MAX_CONVERSATION_TURNS = 5`.
- **Turn Flow Example**:
  - **Turn 1 User**: *"When is my Microsoft assessment?"*
  - **Turn 1 Assistant**: *"Your Microsoft assessment is on 21 August 2025 at 7:00 PM."*
  - **Turn 2 User**: *"What about VISA?"*
  - **Contextualization**: `build_contextualize_prompt()` asks Gemini to rephrase Turn 2 into standalone search query: *"When is my VISA online assessment?"*.
  - **Fresh Retrieval**: The standalone query runs FRESH retrieval against FAISS. Previous LLM answers are NEVER indexed or retrieved as evidence.
  - **Session Clear**: `/clear` calls `pipeline.clear_conversation(session_id)`.

---

## SECTION 13 — SUMMARIZATION

### Dual Summarization Modes (`rag/summarizer.py`)

- **Mode A — Specific Email (`/summary <message_id>`)**:
  - Looks up raw body text by `message_id` in `pipeline._emails_lookup`.
  - Invokes `generate_summary(context, mode="single")`.
- **Mode B — Topic Group (`/summarize <query>`)**:
  - Executes `retrieve_with_context(query, store)`.
  - Formats retrieved Top-5 emails into group context.
  - Invokes `generate_summary(context, mode="query")`.
  - Returns executive summary bullet points + source citations.

---

## SECTION 14 — CLI FLOW

### Command Routing Table (`rag/run.py`)

| CLI Command | What It Does | Pipeline Method |
| :--- | :--- | :--- |
| `/search <query>` | Performs semantic retrieval and prints matching emails | `pipeline.search(query)` |
| `/ask <query>` | Grounded QA: retrieves context and generates Gemini answer | `pipeline.ask(query)` |
| `/summarize <query>`| Summarizes group of emails matching topic query | `pipeline.summarize(query=q)` |
| `/summary <id>` | Summarizes a single specific email by Message ID | `pipeline.summarize(message_id=id)` |
| `/conversation` | Starts interactive multi-turn conversation session | `pipeline.conversation_ask()` |
| `/clear` | Clears current conversation history buffer | `pipeline.clear_conversation()` |
| `/status` | Displays FAISS index status, vector count, and paths | `pipeline.get_index_status()` |
| `/quit` | Exits the CLI interactive shell | Exits loop |
| `python -m rag.run --build-index` | Loads dataset, normalizes, chunks, embeds, builds FAISS index | `pipeline.build_index()` |

---

## SECTION 15 — END-TO-END EXAMPLE

### Tracing Query: `"When is my Microsoft assessment?"`

```
1. CLI (rag/run.py) ──► user enters '/ask When is my Microsoft assessment?'
2. RAGPipeline (rag/pipeline.py) ──► ask() calls retrieve_with_context()
3. Retriever (rag/retriever.py) ──► calls embed_query()
4. Embedder (rag/embedder.py) ──► converts query to 384-d L2-normalized vector
5. VectorStore (rag/vector_store.py) ──► faiss.IndexFlatIP.search() retrieves Top-20 candidates
6. QueryAnalyzer (rag/query_analyzer.py) ──► extracts tokens=['microsoft', 'assessment'], entity=['Microsoft']
7. Reranker (rag/reranker.py) ──► applies composite scoring + deduplication by message_id
8. Retriever (rag/retriever.py) ──► selects Top-5 reranked emails and formats context block
9. Prompt Builder (rag/prompt.py) ──► builds grounded system prompt with Context + Query
10. Generator (rag/generator.py) ──► calls client.models.generate_content() via google-genai SDK
11. Gemini API ──► returns grounded text answer citing dates
12. CLI (rag/run.py) ──► prints answer text and lists 5 source citations with Message IDs
```

---

## SECTION 16 — SEARCH VS RAG

| Aspect | `/search <query>` | `/ask <query>` |
| :--- | :--- | :--- |
| **Output Type** | Raw list of matching email records | Synthesized natural language answer + citations |
| **LLM Invocation** | No LLM used (0 API calls) | Calls Gemini API with grounded context |
| **Use Case** | Fast email browsing and inspection | Direct question answering |
| **Execution Path** | `embed_query` -> `FAISS` -> `reranker` | `embed_query` -> `FAISS` -> `reranker` -> `Gemini` |

---

## SECTION 17 — PERFORMANCE

### Benchmark Measurements (`rag/benchmark_latency.py`)

- **Corpus Size**: 13,281 vectors representing 10,771 unique emails.
- **Warm Retrieval Benchmark Results (10 Test Queries)**:
  - **Minimum**: 15.35 ms
  - **Maximum**: 28.00 ms
  - **Mean**: 18.44 ms
  - **Median (P50)**: 17.22 ms
  - **P95**: **20.22 ms** (Sub-200ms P95 warm retrieval verified).
- **Latency Breakdown**:
  - Index loading (one-off startup): ~450 ms
  - Query Embedding (`all-MiniLM-L6-v2`): ~10 ms
  - FAISS Vector Search (`IndexFlatIP`): ~2 ms
  - Query Analysis & Reranking: ~6 ms
  - Gemini LLM API Generation: ~1,200 ms – 1,800 ms

---

## SECTION 18 — CURRENT LIMITATIONS

1. **Word-Level Chunking**:
   - *Problem*: `_split_text_into_chunks()` splits text on word counts (400 words) without sentence boundary detection.
   - *Why it happens*: Uses simple `text.split()` for speed.
   - *Current Mitigation*: 50-word overlap prevents losing context between adjacent chunks.
2. **Regex Entity Extraction**:
   - *Problem*: `_extract_entities()` relies on regex for capitalized words, occasionally capturing sentence starters.
   - *Current Mitigation*: Entity score weight is kept low (0.07).
3. **In-Memory FAISS Persistence**:
   - *Problem*: Updating index requires reloading metadata into memory.
   - *Current Mitigation*: Suitable for current corpus size (13,281 vectors).

---

## SECTION 19 — WHAT IS ACTUALLY "RAG" HERE?

- **RETRIEVAL**: Dense vector search via `embedder.py` + `vector_store.py` (FAISS) + candidate expansion & reranking in `retriever.py` / `reranker.py`.
- **AUGMENTATION**: Formatting retrieved Top-5 email context blocks into a grounded prompt template in `prompt.py`.
- **GENERATION**: Synthesizing the final answer via Gemini 3.5 Flash-Lite in `generator.py`.

---

## SECTION 20 — PANEL QUESTIONS & DEFENSE

### 1. Why did you choose FAISS?
- **Short Panel Answer**: FAISS (`IndexFlatIP`) provides sub-20ms exact nearest-neighbor vector search on CPU without complex cloud database infrastructure.
- **Deeper Understanding**: For 13,281 vectors, flat inner product search has zero recall degradation compared to approximate graph indices (HNSW).
- **Code Reference**: `rag/vector_store.py` -> `build_index()`.

### 2. Why `sentence-transformers/all-MiniLM-L6-v2`?
- **Short Panel Answer**: It is a lightweight 22.9M parameter model optimized for CPU inference that produces high-quality 384-dimensional embeddings.
- **Deeper Understanding**: Fine-tuned on 1B sentence pairs using Siamese networks, outperforming larger models in retrieval speed vs. accuracy trade-offs.
- **Code Reference**: `rag/config.py` -> `EMBEDDING_MODEL`.

### 3. Why 384 dimensions?
- **Short Panel Answer**: 384 is the native output dimension of `all-MiniLM-L6-v2`, balancing dense semantic representation with low memory footprint.
- **Code Reference**: `rag/config.py` -> `EMBEDDING_DIMENSION`.

### 4. Why chunk emails?
- **Short Panel Answer**: Chunking prevents exceeding the 512-token limit of Transformer embedding models and ensures local context is preserved.
- **Code Reference**: `rag/chunker.py` -> `chunk_email()`.

### 5. Why prepending metadata headers to chunks?
- **Short Panel Answer**: Email bodies often lack subject/sender context; prepending headers ensures every vector captures email origin.
- **Code Reference**: `rag/chunker.py` -> `_build_metadata_header()`.

### 6. Why retrieve Top-20 before reranking?
- **Short Panel Answer**: Expanding initial vector search to Top-20 prevents premature filtering of recent or subject-matching emails that rank lower in raw vector score.
- **Code Reference**: `rag/retriever.py` -> `_initial_k()`.

### 7. Why return Top-5 to Gemini?
- **Short Panel Answer**: Top-5 balances context completeness with prompt token cost and LLM attention focus.
- **Code Reference**: `rag/config.py` -> `FINAL_RETRIEVAL_K`.

### 8. How does deduplication work?
- **Short Panel Answer**: Tracks seen `message_id`s during candidate iteration, keeping only the highest-scoring chunk per distinct email.
- **Code Reference**: `rag/reranker.py` -> `rerank_candidates()`.

### 9. Why use metadata in reranking?
- **Short Panel Answer**: Pure vector similarity ignores recency, subject intent, and sender authority. Metadata reranking integrates these signals.
- **Code Reference**: `rag/reranker.py` -> `_base_score()`.

### 10. How does temporal relevance work?
- **Short Panel Answer**: Calculates exponential date closeness decay to prioritize recent or intent-matched emails.
- **Code Reference**: `rag/reranker.py` -> `_temporal_scores()`.

### 11. How do you prevent hallucination?
- **Short Panel Answer**: Through strict system prompt constraints requiring Gemini to answer ONLY from retrieved email context.
- **Code Reference**: `rag/prompt.py` -> `build_rag_prompt()`.

### 12. What happens if the answer isn't in the emails?
- **Short Panel Answer**: If retrieval returns no results, the system returns an uncertainty fallback message without calling Gemini.
- **Code Reference**: `rag/pipeline.py` -> `ask()`.

### 13. How do follow-up questions work?
- **Short Panel Answer**: Uses Gemini to rephrase follow-ups into standalone search queries before running fresh FAISS retrieval.
- **Code Reference**: `rag/conversation.py` -> `ask()`.

### 14. Why Gemini instead of another LLM?
- **Short Panel Answer**: Gemini 3.5 Flash-Lite offers fast response times, high context adherence, and robust free-tier API quotas via `google-genai`.
- **Code Reference**: `rag/generator.py` -> `generate_answer()`.

### 15. Why is retrieval faster than generation?
- **Short Panel Answer**: Retrieval runs local C++ matrix math (FAISS ~18ms), while generation requires network round-trips to Gemini (~1.5s).
- **Code Reference**: `rag/benchmark_latency.py`.

### 16. What does 13,281 vectors represent?
- **Short Panel Answer**: Total 400-word text chunks generated across all 10,771 unique emails.
- **Code Reference**: `rag/vector_store.py` -> `total_vectors`.

### 17. Why are there fewer unique emails than vectors?
- **Short Panel Answer**: Long emails containing $>400$ words are split into multiple chunks, increasing total vector count.
- **Code Reference**: `rag/chunker.py` -> `chunk_emails()`.

### 18. What happens when an email is very long?
- **Short Panel Answer**: It is divided into multiple chunks with 50-word overlap, each prepended with the email's metadata header.
- **Code Reference**: `rag/chunker.py` -> `_split_text_into_chunks()`.

### 19. How do you handle multiple chunks from one email?
- **Short Panel Answer**: Reranking deduplicates chunks by `message_id`, keeping only the single highest-scoring chunk per email.
- **Code Reference**: `rag/reranker.py`.

### 20. How is the FAISS index persisted?
- **Short Panel Answer**: Saved to `emails.faiss` using `faiss.write_index()` and `metadata.json` using standard JSON dumps.
- **Code Reference**: `rag/vector_store.py` -> `save()`.

### 21. What happens when new Gmail emails arrive?
- **Short Panel Answer**: `pipeline.update_index()` chunks new emails, embeds them, and appends vectors to the existing FAISS index.
- **Code Reference**: `rag/pipeline.py` -> `update_index()`.

### 22. How did you measure 20.22 ms P95?
- **Short Panel Answer**: Executed 10 benchmark queries using `time.perf_counter()` after warm-up runs in `benchmark_latency.py`.
- **Code Reference**: `rag/benchmark_latency.py`.

### 23. Does 20.22 ms include Gemini generation?
- **Short Panel Answer**: No, 20.22 ms measures warm vector search and reranking latency. Gemini generation adds ~1.5 seconds.
- **Code Reference**: `rag/benchmark_latency.py`.

### 24. How is L2 normalization applied?
- **Short Panel Answer**: SentenceTransformer `normalize_embeddings=True` and `faiss.normalize_L2()` scale vectors to unit length 1.0.
- **Code Reference**: `rag/embedder.py` -> `embed_texts()`.

### 25. What is the role of `rag/schemas.py`?
- **Short Panel Answer**: Defines typed Pydantic data models ensuring schema validation between pipeline modules.
- **Code Reference**: `rag/schemas.py`.

---

## SECTION 21 — 30-SECOND EXPLANATION

> *"MailMind is an intelligent email assistant built on a grounded RAG architecture. When a user asks a question, we convert the query into a 384-dimensional dense vector using Sentence-Transformers and search a FAISS index housing 13,281 email chunk vectors. We fetch 20 initial candidates, deduplicate message IDs, and rerank them using a multi-signal scorer weighing semantic similarity, subject match, sender authority, and temporal recency. The Top-5 reranked emails are formatted into a grounded system prompt and sent to Gemini 3.5 Flash-Lite via the `google-genai` SDK, producing factual, evidence-backed answers with clickable source citations while preventing AI hallucination."*

---

## SECTION 22 — 2-MINUTE TECHNICAL EXPLANATION

> *"Technically, MailMind combines local dense vector retrieval with cloud LLM generation. In the offline indexing phase, we ingest academic email data via the Gmail API, normalize headers, and split bodies into 400-word chunks with 50-word overlap. Crucially, every chunk is prepended with a metadata header containing the Subject, Sender, and Date. Chunks are embedded into 384-dimensional L2-normalized vectors using `all-MiniLM-L6-v2` and stored in an in-memory FAISS `IndexFlatIP` index alongside JSON metadata.*

> *In the online query phase, a user query is converted into a unit vector and executed against FAISS to pull 20 initial candidates. Our metadata reranker scores candidates across 6 weighted signals: 65% semantic similarity, 15% subject match, 5% body match, 5% sender authority, 7% entity alignment, and 8% temporal recency decay. After filtering out low-confidence scores and deduplicating chunks by Message ID, the Top-5 distinct email contexts are formatted into a strict system prompt.*

> *This prompt is sent to Gemini 3.5 Flash-Lite using the `google-genai` SDK. The prompt explicitly constrains generation to retrieved context and requires an uncertainty fallback if evidence is lacking. Our warm retrieval latency achieves a P95 of 20.22 milliseconds on CPU, providing fast, factual, evidence-grounded answers."*

---

## SECTION 23 — THINGS I MUST NOT CLAIM

1. **Do NOT claim we use an enterprise cloud vector database** (e.g., Pinecone, Weaviate). We use a local in-memory **FAISS** index (`IndexFlatIP`).
2. **Do NOT claim we fine-tuned the Gemini model**. We use pre-trained Gemini via the `google-genai` SDK with prompt engineering.
3. **Do NOT claim retrieval latency includes Gemini API generation**. Retrieval (FAISS + Reranking) takes **~18ms**; Gemini generation takes **~1.5s**.
4. **Do NOT claim priority classification is part of RAG**. Email classification (Spam/General/Important) is a separate BiLSTM component in `ml/categorization`.
5. **Do NOT claim we use `google-generativeai`**. The codebase uses the newer **`google-genai`** SDK (`from google import genai`).
6. **Do NOT claim we re-embed full emails without chunking**. Emails are chunked at 400 words with 50-word overlap.

---

## MAILMIND RAG — IMPLEMENTATION CHEAT SHEET

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
- **FAISS Index Type**: `faiss.IndexFlatIP(384)` (Flat Inner Product, exact cosine search).
- **Corpus Statistics**: 13,281 vectors representing 10,771 unique emails.
- **Chunk Parameters**: 400 words per chunk, 50-word overlap, metadata header prepended.
- **Retrieval Pipeline**: Top-20 Initial Candidates -> Deduplication -> Reranking -> Top-5 Final Context.
- **Reranking Weights**: Semantic (0.65), Subject (0.15), Body (0.05), Sender (0.05), Entity (0.07), Temporal (0.08).
- **LLM Generator**: Gemini 3.5 Flash-Lite / Gemini 2.0 Flash via `google-genai` SDK.
- **Grounded Prompt**: Constrains Gemini to answer ONLY from retrieved email context.
- **Empty Retrieval**: Returns uncertainty fallback message without invoking Gemini API.
- **Warm Retrieval Latency**: Mean 18.44 ms, P50 17.22 ms, **P95 20.22 ms** (verified).

### RAG Component Summary Table

| Component | File | Main Function / Class | One-Line Purpose |
| :--- | :--- | :--- | :--- |
| **Config** | `rag/config.py` | `Settings` | Centralized hyper-parameters and path management. |
| **Schemas** | `rag/schemas.py` | `DocumentChunk`, `SearchResult` | Pydantic type validation for records and API responses. |
| **Loader** | `rag/loader.py` | `load_emails()` | Reads raw email data from Parquet/JSON files. |
| **Normalizer**| `rag/normalizer.py` | `normalize_emails()` | Cleans HTML markup and standardizes email headers. |
| **Chunker** | `rag/chunker.py` | `chunk_emails()` | Splits long emails into 400-word chunks with metadata headers. |
| **Embedder** | `rag/embedder.py` | `embed_texts()`, `embed_query()` | Generates 384-d L2-normalized `all-MiniLM-L6-v2` vectors. |
| **Vector Store**| `rag/vector_store.py`| `VectorStore`, `build_index()` | Manages FAISS `IndexFlatIP` and disk persistence. |
| **Query Analyzer**| `rag/query_analyzer.py`| `analyze_query()` | Extracts keywords, temporal intent, and entities from query. |
| **Reranker** | `rag/reranker.py` | `rerank_candidates()` | Multi-signal 6-weight candidate reranking and deduplication. |
| **Retriever** | `rag/retriever.py` | `retrieve_with_context()` | Orchestrates Top-20 candidate retrieval and context formatting. |
| **Prompt** | `rag/prompt.py` | `build_rag_prompt()` | Builds strict grounding prompts for LLM context constraint. |
| **Generator** | `rag/generator.py` | `generate_answer()` | Interfaces with `google-genai` SDK for Gemini generation. |
| **Conversation**| `rag/conversation.py`| `ConversationManager` | Manages multi-turn history and rephrases follow-up queries. |
| **Summarizer** | `rag/summarizer.py` | `summarize_by_query()` | Generates executive summary bullets for single/group emails. |
| **Pipeline** | `rag/pipeline.py` | `RAGPipeline` | High-level facade connecting all RAG components. |
| **CLI** | `rag/run.py` | `main()` | Interactive terminal shell routing user commands. |
| **Benchmark** | `rag/benchmark_latency.py`| `percentile()` | Measures warm retrieval latency (20.22ms P95). |
