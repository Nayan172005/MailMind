"""
MailMind RAG — Pipeline Orchestrator

High-level API that ties together all RAG components.
This is the main entry point for programmatic usage.
"""

from pathlib import Path
from typing import Optional

from rag.config import settings
from rag.loader import load_emails
from rag.normalizer import normalize_emails
from rag.chunker import chunk_emails
from rag.schemas import (
    EmailRecord,
    SearchResult,
    SearchResponse,
    AskResponse,
    SummarizeResponse,
    ConversationResponse,
    IndexStatusResponse,
    EmailDetailResponse,
)
from rag.vector_store import VectorStore
from rag.retriever import retrieve, retrieve_with_context
from rag.generator import generate_answer
from rag.summarizer import summarize_single_email, summarize_by_query
from rag.conversation import ConversationManager
from rag.utils import get_logger

logger = get_logger("pipeline")


class RAGPipeline:
    """
    Main RAG pipeline orchestrator.

    Usage:
        pipeline = RAGPipeline()
        pipeline.load_index()  # or pipeline.build_index()

        results = pipeline.search("placement interviews")
        answer = pipeline.ask("When is my Microsoft assessment?")
        summary = pipeline.summarize(query="placement emails")
    """

    def __init__(self):
        self.store = VectorStore()
        self.conversation_manager = ConversationManager()
        self._emails: list[EmailRecord] = []
        self._emails_lookup: dict[str, dict] = {}  # message_id -> raw email dict

    @property
    def is_ready(self) -> bool:
        """Whether the index is loaded and ready for queries."""
        return self.store.is_loaded

    def build_index(
        self,
        dataset_path: str | None = None,
        force_rebuild: bool = False,
    ) -> IndexStatusResponse:
        """
        Build the FAISS index from the email dataset.

        Loads emails, normalizes, chunks, embeds, and creates the index.
        Saves the index to disk for persistence.
        """
        # Check if index already exists
        if not force_rebuild and settings.faiss_index_path.exists():
            logger.info("Index already exists. Use force_rebuild=True to rebuild.")
            self.load_index()
            return self.get_index_status()

        # 1. Load raw emails
        raw_emails = load_emails(dataset_path)

        # 2. Normalize
        emails = normalize_emails(raw_emails)
        self._emails = emails

        # Build lookup for summarization
        self._emails_lookup = {}
        for email in emails:
            self._emails_lookup[email.message_id] = {
                "body": email.body,
                "subject": email.subject,
                "sender_name": email.sender_name,
                "sender_email": email.sender_email,
                "date": email.date,
            }

        # 3. Chunk
        chunks = chunk_emails(emails)

        # 4. Build FAISS index (embedding happens inside)
        self.store.build_index(chunks)

        # 5. Save to disk
        self.store.save()

        logger.info("Index built and saved successfully!")
        return self.get_index_status()

    def load_index(self, index_dir: str | None = None) -> None:
        """Load a persisted index from disk."""
        self.store.load(index_dir)
        logger.info("Index loaded and ready for queries.")

    def update_index(
        self,
        dataset_path: str | None = None,
    ) -> IndexStatusResponse:
        """
        Update the index with new emails.

        Loads the current dataset, finds emails not yet indexed,
        and adds them to the existing index.
        """
        if not self.is_ready:
            self.load_index()

        raw_emails = load_emails(dataset_path)
        emails = normalize_emails(raw_emails)

        chunks = chunk_emails(emails)
        added = self.store.update_index(chunks)

        if added > 0:
            self.store.save()
            logger.info(f"Index updated with {added} new chunks.")
        else:
            logger.info("No new emails to add.")

        return self.get_index_status()

    def get_index_status(self) -> IndexStatusResponse:
        """Get current index status."""
        return IndexStatusResponse(
            indexed=self.store.is_loaded,
            total_vectors=self.store.total_vectors,
            total_emails=self.store.total_emails,
            index_path=str(settings.index_abs_dir),
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> SearchResponse:
        """
        Semantic email search.

        Returns ranked email results based on semantic similarity.
        """
        if not self.is_ready:
            raise RuntimeError("Index not loaded. Call load_index() or build_index() first.")

        results = retrieve(query, self.store, top_k=top_k)

        return SearchResponse(
            query=query,
            results=results,
            total=len(results),
        )

    def ask(
        self,
        query: str,
        top_k: int | None = None,
    ) -> AskResponse:
        """
        Question answering with grounded generation.

        Retrieves relevant emails, then generates an answer using Gemini.
        Returns "not found" if no relevant emails are retrieved.
        """
        if not self.is_ready:
            raise RuntimeError("Index not loaded. Call load_index() or build_index() first.")

        results, context = retrieve_with_context(query, self.store, top_k=top_k)

        if not results:
            return AskResponse(
                query=query,
                answer=(
                    "I couldn't find any relevant emails in your mailbox "
                    "for this query. Please try rephrasing your question."
                ),
                sources=[],
            )

        answer = generate_answer(query, context, max_sources=len(results))

        return AskResponse(
            query=query,
            answer=answer,
            sources=results,
        )

    def summarize(
        self,
        message_id: str | None = None,
        query: str | None = None,
        top_k: int | None = None,
    ) -> SummarizeResponse:
        """
        Summarize email(s).

        Mode A: Provide message_id to summarize a single email.
        Mode B: Provide query to summarize matching emails.
        """
        if not self.is_ready:
            raise RuntimeError("Index not loaded. Call load_index() or build_index() first.")

        if message_id:
            return summarize_single_email(
                message_id, self.store, self._emails_lookup
            )
        elif query:
            return summarize_by_query(query, self.store, top_k=top_k)
        else:
            return SummarizeResponse(
                summary="Please provide either a message_id or a search query.",
                sources=[],
                mode="",
            )

    def conversation_ask(
        self,
        session_id: str,
        query: str,
        top_k: int | None = None,
    ) -> ConversationResponse:
        """
        Conversational question answering.

        Maintains session history and contextualizes follow-up queries.
        Each turn performs fresh retrieval.
        """
        if not self.is_ready:
            raise RuntimeError("Index not loaded. Call load_index() or build_index() first.")

        return self.conversation_manager.ask(
            session_id=session_id,
            query=query,
            store=self.store,
            top_k=top_k,
        )

    def get_email(self, message_id: str) -> EmailDetailResponse | None:
        """Retrieve full email details by message_id."""
        # Check the loaded emails first
        if self._emails_lookup and message_id in self._emails_lookup:
            data = self._emails_lookup[message_id]
            return EmailDetailResponse(
                message_id=message_id,
                subject=data.get("subject", ""),
                sender_name=data.get("sender_name", ""),
                sender_email=data.get("sender_email", ""),
                date=data.get("date", ""),
                body=data.get("body", ""),
            )

        # Fall back to store metadata
        chunks = self.store.get_chunks_by_message_id(message_id)
        if chunks:
            first = chunks[0]
            body_parts = [c.get("body_snippet", "") for c in sorted(chunks, key=lambda x: x.get("chunk_id", 0))]
            return EmailDetailResponse(
                message_id=message_id,
                subject=first.get("subject", ""),
                sender_name=first.get("sender_name", ""),
                sender_email=first.get("sender_email", ""),
                date=first.get("date", ""),
                body="\n\n".join(body_parts),
            )

        return None

    def clear_conversation(self, session_id: str) -> None:
        """Clear a conversation session."""
        self.conversation_manager.clear_session(session_id)
