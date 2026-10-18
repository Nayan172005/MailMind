"""
MailMind RAG — Gemini Generator

Integrates with the Google Gemini API for grounded answer generation.
Handles empty retrieval gracefully — never calls Gemini without context.
"""

from google import genai

from rag.config import settings
from rag.prompt import build_rag_prompt, build_summary_prompt
from rag.utils import get_logger

logger = get_logger("generator")

# Module-level client cache
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Get or create the Gemini API client."""
    global _client
    if _client is None:
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Please set it in your .env file or environment variables."
            )
        _client = genai.Client(api_key=api_key)
        logger.info(f"Gemini client initialized (model: {settings.GEMINI_MODEL})")
    return _client


def validate_citations(answer: str, max_sources: int) -> str:
    """
    Validate and clean citation references in generated answers.
    Ensures that any [Source X] references correspond to real sources in range 1..max_sources.
    Removes non-existent citation tags to avoid hallucinated references.
    """
    import re

    def _check_cite(match: re.Match) -> str:
        idx_str = match.group(1)
        try:
            idx = int(idx_str)
            if 1 <= idx <= max_sources:
                return f"[Source {idx}]"
            else:
                return ""  # drop invalid citation
        except ValueError:
            return ""

    cleaned = re.sub(r"\[Source\s*(\d+)\]", _check_cite, answer, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def generate_answer(
    query: str,
    context: str,
    model: str | None = None,
    max_sources: int = 5,
) -> str:
    """
    Generate a grounded answer using Gemini.

    Args:
        query: The user's question.
        context: The retrieved email context (formatted text).
        model: Override the default Gemini model.
        max_sources: Number of sources provided in context for citation validation.

    Returns:
        Generated answer string.

    Raises:
        RuntimeError if API key is missing or API call fails.
    """
    if not context.strip():
        return (
            "I couldn't find any relevant emails in your mailbox for this query. "
            "Please try rephrasing your question or searching for different terms."
        )

    client = _get_client()
    model_name = model or settings.GEMINI_MODEL
    prompt = build_rag_prompt(query, context)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        answer = response.text
        if not answer or not answer.strip():
            return "I received an empty response. Please try again."

        cleaned_answer = answer.strip()
        if max_sources > 0:
            cleaned_answer = validate_citations(cleaned_answer, max_sources)

        return cleaned_answer

    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise RuntimeError(f"Failed to generate answer: {e}")


def generate_summary(
    context: str,
    mode: str = "single",
    model: str | None = None,
) -> str:
    """
    Generate an email summary using Gemini.

    Args:
        context: Email content to summarize.
        mode: "single" for one email, "multi" for multiple.
        model: Override the default Gemini model.

    Returns:
        Summary string.
    """
    if not context.strip():
        return "No email content to summarize."

    client = _get_client()
    model_name = model or settings.GEMINI_MODEL
    prompt = build_summary_prompt(context, mode)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        summary = response.text
        if not summary or not summary.strip():
            return "I received an empty response. Please try again."
        return summary.strip()

    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise RuntimeError(f"Failed to generate summary: {e}")


def generate_raw(
    prompt: str,
    model: str | None = None,
) -> str:
    """
    Generate a raw response from Gemini (used for query contextualization).

    Args:
        prompt: Full prompt text.
        model: Override the default Gemini model.

    Returns:
        Generated text.
    """
    client = _get_client()
    model_name = model or settings.GEMINI_MODEL

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return (response.text or "").strip()

    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return ""
