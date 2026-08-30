"""
MailMind RAG — Conversation Manager

Maintains session-level conversation history and contextualizes
follow-up queries. Each turn performs FRESH retrieval — previous
assistant answers are NEVER the source of truth.

Flow:
  conversation history → contextualized query → fresh retrieval → grounded answer
"""

from dataclasses import dataclass, field

from rag.config import settings
from rag.schemas import (
    SearchResult,
    ConversationMessage,
    ConversationResponse,
)
from rag.vector_store import VectorStore
from rag.retriever import retrieve_with_context
from rag.prompt import build_conversation_prompt, build_contextualize_prompt
from rag.generator import generate_raw
from rag.utils import get_logger

logger = get_logger("conversation")


@dataclass
class ConversationSession:
    """A single conversation session."""
    session_id: str
    history: list[ConversationMessage] = field(default_factory=list)
    max_turns: int = 5

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        """Add a user-assistant turn to history."""
        self.history.append(ConversationMessage(role="user", content=user_msg))
        self.history.append(ConversationMessage(role="assistant", content=assistant_msg))

        # Trim to max_turns (each turn = 2 messages)
        max_messages = self.max_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def format_history(self) -> str:
        """Format history for prompt injection."""
        if not self.history:
            return "(No previous conversation)"

        lines = []
        for msg in self.history:
            role = "User" if msg.role == "user" else "Assistant"
            # Truncate long messages in history to save tokens
            content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @property
    def has_history(self) -> bool:
        return len(self.history) > 0


class ConversationManager:
    """Manages multiple conversation sessions."""

    def __init__(self):
        self._sessions: dict[str, ConversationSession] = {}

    def get_session(self, session_id: str) -> ConversationSession:
        """Get or create a conversation session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationSession(
                session_id=session_id,
                max_turns=settings.MAX_CONVERSATION_TURNS,
            )
        return self._sessions[session_id]

    def clear_session(self, session_id: str) -> None:
        """Clear a conversation session."""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def ask(
        self,
        session_id: str,
        query: str,
        store: VectorStore,
        top_k: int | None = None,
    ) -> ConversationResponse:
        """
        Handle a conversational query.

        1. If there's history, contextualize the query
        2. Perform fresh retrieval with the contextualized query
        3. Generate a grounded answer using conversation context
        4. Store the turn in history
        """
        session = self.get_session(session_id)
        top_k = top_k or settings.TOP_K

        # 1. Contextualize query if there's conversation history
        search_query = query
        if session.has_history:
            history_text = session.format_history()
            contextualize_prompt = build_contextualize_prompt(query, history_text)
            contextualized = generate_raw(contextualize_prompt)
            if contextualized.strip():
                search_query = contextualized.strip()
                logger.info(f"Contextualized query: '{query}' -> '{search_query}'")

        # 2. Fresh retrieval
        results, context = retrieve_with_context(
            search_query, store, top_k=top_k
        )

        # 3. Generate answer
        if not results:
            answer = (
                "I couldn't find any relevant emails for this question. "
                "Please try rephrasing or asking about a different topic."
            )
        else:
            # Build conversational prompt with history
            history_text = session.format_history()

            from rag.prompt import build_conversation_prompt
            from google import genai

            prompt = build_conversation_prompt(query, context, history_text)

            try:
                client = genai.Client(api_key=settings.GEMINI_API_KEY)
                response = client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=prompt,
                )
                answer = (response.text or "").strip()
                if not answer:
                    answer = "I received an empty response. Please try again."
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                answer = f"An error occurred during generation: {e}"

        # 4. Store turn
        session.add_turn(query, answer)

        return ConversationResponse(
            session_id=session_id,
            query=query,
            answer=answer,
            sources=results,
            history=session.history.copy(),
        )
