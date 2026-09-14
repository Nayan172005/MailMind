"""
MailMind RAG - Lightweight query analysis.

This module intentionally avoids LLM calls. It extracts enough structure from a
user query to guide candidate reranking while leaving FAISS semantic retrieval as
the primary retrieval signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re


class QueryType(str, Enum):
    GENERAL_SEARCH = "GENERAL_SEARCH"
    QUESTION = "QUESTION"
    SUMMARY = "SUMMARY"
    TEMPORAL_SEARCH = "TEMPORAL_SEARCH"
    ENTITY_SEARCH = "ENTITY_SEARCH"


class TemporalIntent(str, Enum):
    RECENT = "RECENT"
    UPCOMING = "UPCOMING"
    TODAY = "TODAY"
    TOMORROW = "TOMORROW"
    YESTERDAY = "YESTERDAY"
    EARLIEST = "EARLIEST"


STOPWORDS = {
    "a", "about", "all", "am", "an", "and", "any", "are", "as", "at",
    "be", "by", "can", "could", "did", "do", "does", "email", "emails",
    "find", "for", "from", "give", "has", "have", "i", "in", "is", "it",
    "me", "my", "of", "on", "or", "please", "related", "show", "tell",
    "that", "the", "their", "there", "these", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with", "you", "your",
}

QUESTION_WORDS = {"what", "when", "where", "who", "why", "how", "which"}
SUMMARY_WORDS = {"summarize", "summary", "summarise", "digest", "overview"}

TEMPORAL_PATTERNS: tuple[tuple[TemporalIntent, tuple[str, ...]], ...] = (
    (TemporalIntent.RECENT, ("most recent", "recent", "latest", "newest", "last")),
    (TemporalIntent.UPCOMING, ("upcoming", "next")),
    (TemporalIntent.TODAY, ("today",)),
    (TemporalIntent.TOMORROW, ("tomorrow",)),
    (TemporalIntent.YESTERDAY, ("yesterday",)),
    (TemporalIntent.EARLIEST, ("earliest", "oldest", "first")),
)


@dataclass(frozen=True)
class QueryAnalysis:
    query: str
    normalized_query: str
    query_type: QueryType
    terms: list[str] = field(default_factory=list)
    subject_terms: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    temporal_intent: TemporalIntent | None = None


def normalize_text(text: str) -> str:
    """Lowercase text and collapse non-alphanumeric runs to spaces."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Tokenize text for lightweight lexical scoring."""
    tokens = re.findall(r"[a-z0-9][a-z0-9.+_-]*", text.lower())
    if keep_stopwords:
        return tokens
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        key = normalize_text(value)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value.strip())
    return result


def _detect_temporal_intent(normalized_query: str) -> TemporalIntent | None:
    padded = f" {normalized_query} "
    for intent, patterns in TEMPORAL_PATTERNS:
        for pattern in patterns:
            if f" {pattern} " in padded:
                return intent
    return None


def _extract_entities(query: str) -> list[str]:
    """Extract explicit entity-like phrases without maintaining company lists."""
    entities: list[str] = []

    # User-quoted phrases are usually intentional entities/topics.
    entities.extend(re.findall(r'"([^"]{2,80})"|\'([^\']{2,80})\'', query))
    entities = [item for match in entities for item in (match if isinstance(match, tuple) else (match,)) if item]

    # Capitalized or acronym phrases: Microsoft, NPTEL, Tata Electronics, VISA.
    entity_pattern = re.compile(
        r"\b(?:[A-Z][A-Za-z0-9&.+-]*|[A-Z0-9&.+-]{2,})"
        r"(?:\s+(?:[A-Z][A-Za-z0-9&.+-]*|[A-Z0-9&.+-]{2,}))*\b"
    )
    for match in entity_pattern.finditer(query):
        phrase = match.group(0).strip()
        first = phrase.split()[0].lower()
        if first in STOPWORDS or first in QUESTION_WORDS:
            continue
        entities.append(phrase)

    return _dedupe_preserve_order(entities)


def analyze_query(query: str) -> QueryAnalysis:
    """Analyze a user query for deterministic retrieval signals."""
    stripped = query.strip()
    normalized = normalize_text(stripped)
    all_terms = tokenize(stripped)
    temporal_intent = _detect_temporal_intent(normalized)
    entities = _extract_entities(stripped)

    temporal_terms = {
        term
        for _, patterns in TEMPORAL_PATTERNS
        for pattern in patterns
        for term in pattern.split()
    }
    subject_terms = [
        term for term in all_terms
        if term not in temporal_terms and term not in SUMMARY_WORDS
    ]

    first_token = tokenize(stripped, keep_stopwords=True)[:1]
    if temporal_intent:
        query_type = QueryType.TEMPORAL_SEARCH
    elif entities:
        query_type = QueryType.ENTITY_SEARCH
    elif any(term in SUMMARY_WORDS for term in all_terms):
        query_type = QueryType.SUMMARY
    elif stripped.endswith("?") or (first_token and first_token[0] in QUESTION_WORDS):
        query_type = QueryType.QUESTION
    else:
        query_type = QueryType.GENERAL_SEARCH

    return QueryAnalysis(
        query=stripped,
        normalized_query=normalized,
        query_type=query_type,
        terms=_dedupe_preserve_order(all_terms),
        subject_terms=_dedupe_preserve_order(subject_terms),
        entities=entities,
        temporal_intent=temporal_intent,
    )
