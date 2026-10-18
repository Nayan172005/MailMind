"""
MailMind RAG - Lightweight query analysis.

This module intentionally avoids external LLM calls. It extracts structured
intent, entities, senders, venues, temporal constraints, and broad collection signals
from user queries using deterministic, explainable extraction.
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

CONVERSATIONAL_VERBS = {
    "find", "summarize", "summarise", "list", "show", "search", "give", "tell",
    "get", "display", "check", "fetch", "provide", "share", "what", "when",
    "where", "which", "who", "why", "how", "email", "emails", "message",
    "messages", "circular", "circulars", "notice", "notices", "details",
    "update", "updates", "info", "information", "regarding", "about",
}

QUESTION_WORDS = {"what", "when", "where", "who", "why", "how", "which"}
SUMMARY_WORDS = {"summarize", "summary", "summarise", "digest", "overview"}

BROAD_PATTERNS = (
    "all emails", "all messages", "all updates", "all notices", "all circulars",
    "find all", "list all", "show all", "give all", "fetch all", "every email",
    "circular regarding", "circulars regarding",
)

TEMPORAL_PATTERNS: tuple[tuple[TemporalIntent, tuple[str, ...]], ...] = (
    (TemporalIntent.RECENT, ("most recent", "recent", "latest", "newest", "last")),
    (TemporalIntent.UPCOMING, ("upcoming", "next")),
    (TemporalIntent.TODAY, ("today",)),
    (TemporalIntent.TOMORROW, ("tomorrow",)),
    (TemporalIntent.YESTERDAY, ("yesterday",)),
    (TemporalIntent.EARLIEST, ("earliest", "oldest", "first")),
)

MONTH_NAMES = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
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
    senders: list[str] = field(default_factory=list)
    venues: list[str] = field(default_factory=list)
    date_tokens: list[str] = field(default_factory=list)
    is_broad_search: bool = False


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


def _detect_broad_search(query: str) -> bool:
    norm = normalize_text(query)
    padded = f" {norm} "
    return any(p in padded for p in BROAD_PATTERNS)


def _extract_senders(query: str) -> list[str]:
    """Extract explicit sender mentions or email addresses from query."""
    senders: list[str] = []

    # Direct email address pattern
    email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    for match in email_pattern.finditer(query):
        senders.append(match.group(0))

    # Sender expressions like: "emails from Helpdesk CDC regarding...", "from Google", "sent by CDC"
    sender_expr = re.compile(
        r"(?:emails?\s+from|from|sent\s+by)\s+([A-Za-z0-9@._\s\-]+?)"
        r"(?:\s+(?:regarding|about|for|on|with|mentioning|in\s+regard\s+to|batch)|\?|$|[.,;])",
        re.IGNORECASE,
    )
    for match in sender_expr.finditer(query):
        raw_sender = match.group(1).strip()
        # Clean leading words like 'the', 'my'
        cleaned = re.sub(r"^(?:the|my)\s+", "", raw_sender, flags=re.IGNORECASE).strip()
        tokens = tokenize(cleaned)
        if tokens and not all(t in STOPWORDS or t in CONVERSATIONAL_VERBS for t in tokens):
            senders.append(cleaned)

    return _dedupe_preserve_order(senders)


def _extract_venues(query: str) -> list[str]:
    """Extract venue or room code patterns (e.g., PRP 717, Anna Auditorium)."""
    venues: list[str] = []

    # Room codes e.g. PRP 717, SJT 101, TT 202
    room_pattern = re.compile(r"\b(?:PRP|SJT|TT|SMV|GDN|MB|CB|VJT|ALM)\s*[-]?\s*\d{2,4}[A-Za-z]?\b", re.IGNORECASE)
    for match in room_pattern.finditer(query):
        venues.append(match.group(0))

    # Named venue locations e.g. "Anna Auditorium", "Main Hall", "Lab 3"
    named_pattern = re.compile(
        r"\b(?:[A-Z][a-zA-Z\s]{1,20}(?:Auditorium|Hall|Auditorium|Seminar\s+Hall|Conference\s+Room))\b"
    )
    for match in named_pattern.finditer(query):
        venues.append(match.group(0))

    return _dedupe_preserve_order(venues)


def _extract_date_tokens(query: str) -> list[str]:
    """Extract explicit date mentions, year references, or batch tokens."""
    tokens: list[str] = []

    # Year + batch e.g. "2027 batch", "2026 batch"
    batch_matches = re.findall(r"\b(20\d\d\s+batch)\b", query, re.IGNORECASE)
    tokens.extend(batch_matches)

    # Month + year e.g. "July 2026", "Jan 2025"
    month_year = re.findall(
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December|"
        r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+20\d\d)\b",
        query,
        re.IGNORECASE,
    )
    tokens.extend(month_year)

    # Standalone 4-digit years (e.g., "1994", "2026", "2027")
    year_matches = re.findall(r"\b(19\d\d|20\d\d)\b", query)
    tokens.extend(year_matches)

    return _dedupe_preserve_order(tokens)


def _extract_entities(query: str) -> list[str]:
    """
    Extract explicit entity-like phrases without maintaining hardcoded company lists.
    Extracts quoted phrases, parenthesized abbreviations, acronyms, and capitalized
    or preposition-bound proper nouns.
    """
    entities: list[str] = []

    # 1. User-quoted phrases: "Optum", 'Pixel Compute'
    entities.extend(re.findall(r'"([^"]{2,80})"|\'([^\']{2,80})\'', query))
    entities = [item for match in entities for item in (match if isinstance(match, tuple) else (match,)) if item]

    # 2. Acronyms in parentheses: "School of Computer Science and Engineering (SCOPE)"
    paren_matches = re.findall(r"\(([A-Z0-9&.+-]{2,10})\)", query)
    entities.extend(paren_matches)

    # Full phrase before parenthesis if applicable: "School of Computer Science and Engineering"
    pre_paren = re.findall(r"([A-Z][a-zA-Z\s]{5,60})\s*\([A-Z0-9&.+-]{2,10}\)", query)
    for p in pre_paren:
        p_clean = p.strip()
        if p_clean:
            entities.append(p_clean)

    # 3. Capitalized or acronym phrases: Microsoft, NPTEL, Tata Electronics, VISA, Optum
    entity_pattern = re.compile(
        r"\b(?:[A-Z][A-Za-z0-9&.+-]*|[A-Z0-9&.+-]{2,})"
        r"(?:\s+(?:[A-Z][A-Za-z0-9&.+-]*|[A-Z0-9&.+-]{2,}))*\b"
    )
    for match in entity_pattern.finditer(query):
        phrase = match.group(0).strip()
        words = phrase.split()
        # Drop leading conversational verbs (e.g. "Find all...", "Summarize...")
        while words and words[0].lower() in CONVERSATIONAL_VERBS:
            words = words[1:]
        if not words:
            continue
        cleaned_phrase = " ".join(words)
        if len(cleaned_phrase) > 1 and cleaned_phrase.lower() not in STOPWORDS:
            entities.append(cleaned_phrase)
            if len(words) > 1:
                for w in words:
                    if len(w) > 1 and w.lower() not in STOPWORDS and w.lower() not in CONVERSATIONAL_VERBS:
                        entities.append(w)

    # 4. Contextual entity extraction for lowercase or mixed queries:
    # Targets entities following "for", "at", "from", "with", "regarding", "about"
    preposition_pattern = re.compile(
        r"(?:for|at|from|with|regarding|about)\s+([A-Za-z0-9&.+-]+(?:\s+[A-Za-z0-9&.+-]+)?)\b",
        re.IGNORECASE,
    )
    for match in preposition_pattern.finditer(query):
        phrase = match.group(1).strip()
        p_tokens = tokenize(phrase)
        # Skip if composed purely of stopwords or generic conversational nouns
        if not p_tokens or all(t in STOPWORDS or t in CONVERSATIONAL_VERBS for t in p_tokens):
            continue
        # Skip generic query words like "the company", "the test", "the interview"
        generic = {"company", "interview", "test", "internship", "exam", "placement", "online assessment", "queries"}
        if phrase.lower() in generic or all(t in generic for t in p_tokens):
            continue
        entities.append(phrase)
        if len(p_tokens) > 1:
            for tok in p_tokens:
                if tok.lower() not in STOPWORDS and tok.lower() not in CONVERSATIONAL_VERBS and tok.lower() not in generic:
                    entities.append(tok)

    # 5. Entity modifier before key event nouns (e.g. "<entity> exam", "<entity> test", "<entity> internship")
    event_noun_pattern = re.compile(
        r"(?:the|a|an)?\s*([A-Za-z0-9&.+-]+(?:\s+[A-Za-z0-9&.+-]+)?)\s+(?:exam|test|interview|assessment|drive|shortlist|course|registration|session|workshop|hackathon|internship)\b",
        re.IGNORECASE,
    )
    for match in event_noun_pattern.finditer(query):
        phrase = match.group(1).strip()
        p_tokens = tokenize(phrase)
        if not p_tokens or all(t in STOPWORDS or t in CONVERSATIONAL_VERBS for t in p_tokens):
            continue
        generic = {"online", "upcoming", "recent", "latest", "first", "final", "next", "new", "mock"}
        if phrase.lower() in generic or all(t in generic for t in p_tokens):
            continue
        entities.append(phrase)
        if len(p_tokens) > 1:
            for tok in p_tokens:
                if tok.lower() not in STOPWORDS and tok.lower() not in CONVERSATIONAL_VERBS and tok.lower() not in generic:
                    entities.append(tok)

    return _dedupe_preserve_order(entities)


def analyze_query(query: str) -> QueryAnalysis:
    """Analyze a user query for deterministic retrieval and reranking signals."""
    stripped = query.strip()
    normalized = normalize_text(stripped)
    all_terms = tokenize(stripped)
    temporal_intent = _detect_temporal_intent(normalized)
    entities = _extract_entities(stripped)
    senders = _extract_senders(stripped)
    venues = _extract_venues(stripped)
    date_tokens = _extract_date_tokens(stripped)
    is_broad = _detect_broad_search(stripped)

    temporal_terms = {
        term
        for _, patterns in TEMPORAL_PATTERNS
        for pattern in patterns
        for term in pattern.split()
    }
    date_term_set = {t for dt in date_tokens for t in tokenize(dt)}
    sender_term_set = {t for s in senders for t in tokenize(s)}

    subject_terms = [
        term for term in all_terms
        if term not in temporal_terms
        and term not in SUMMARY_WORDS
        and term not in CONVERSATIONAL_VERBS
        and term not in date_term_set
    ]

    # Fallback: if subject_terms became empty, keep all non-stopwords
    if not subject_terms:
        subject_terms = [t for t in all_terms if t not in temporal_terms]
    if not subject_terms:
        subject_terms = list(all_terms)

    first_token = tokenize(stripped, keep_stopwords=True)[:1]
    if temporal_intent:
        query_type = QueryType.TEMPORAL_SEARCH
    elif entities or senders:
        query_type = QueryType.ENTITY_SEARCH
    elif is_broad or any(term in SUMMARY_WORDS for term in all_terms):
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
        senders=senders,
        venues=venues,
        date_tokens=date_tokens,
        is_broad_search=is_broad,
    )
