"""
MailMind RAG - metadata-aware candidate reranking.

FAISS similarity remains the dominant signal. Metadata signals are conservative
secondary adjustments used after deeper FAISS retrieval.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
import math
from typing import Any

from rag.config import settings
from rag.query_analyzer import (
    QueryAnalysis,
    TemporalIntent,
    normalize_text,
    tokenize,
)
from rag.utils import get_logger, truncate_text

logger = get_logger("reranker")


def parse_email_date(value: Any) -> datetime | None:
    """Parse common email date formats without raising on malformed input."""
    if value is None:
        return None

    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None

        dt = None
        parsers = (
            lambda s: parsedate_to_datetime(s),
            lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")),
        )
        for parser in parsers:
            try:
                dt = parser(text)
                break
            except (TypeError, ValueError, IndexError, AttributeError):
                continue

        if dt is None:
            for fmt in (
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
                "%d-%m-%Y",
                "%d/%m/%Y",
                "%m/%d/%Y",
                "%a, %d %b %Y %H:%M:%S %z",
                "%d %b %Y %H:%M:%S %z",
            ):
                try:
                    dt = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue

        if dt is None:
            try:
                from dateutil import parser as dateutil_parser

                dt = dateutil_parser.parse(text, fuzzy=True)
            except Exception:
                return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _semantic_score(raw_score: float) -> float:
    # FAISS IndexFlatIP over normalized embeddings behaves like cosine
    # similarity. Clamp to keep weighted scores interpretable.
    return max(0.0, min(1.0, raw_score))


def _lexical_score(query_terms: list[str], text: str) -> float:
    if not query_terms or not text:
        return 0.0

    text_tokens = set(tokenize(text))
    if not text_tokens:
        return 0.0

    matched = 0
    for term in query_terms:
        term_norm = term.rstrip("s")
        if term in text_tokens or term_norm in {t.rstrip("s") for t in text_tokens}:
            matched += 1

    coverage = matched / len(query_terms)
    density = matched / max(len(text_tokens), 1)
    return min(1.0, (0.85 * coverage) + (0.15 * min(1.0, density * 4)))


def _phrase_present(phrase: str, text: str) -> bool:
    if not phrase or not text:
        return False
    return normalize_text(phrase) in normalize_text(text)


def _entity_score(entities: list[str], candidate: dict) -> float:
    if not entities:
        return 0.0

    subject = candidate.get("subject", "")
    sender = f"{candidate.get('sender_name', '')} {candidate.get('sender_email', '')}"
    body = f"{candidate.get('body_snippet', '')} {candidate.get('text', '')}"

    scores = []
    for entity in entities:
        if _phrase_present(entity, subject):
            scores.append(1.0)
        elif _phrase_present(entity, sender):
            scores.append(0.85)
        elif _phrase_present(entity, body):
            scores.append(0.65)
        else:
            entity_terms = tokenize(entity)
            combined = f"{subject} {sender} {body}"
            scores.append(_lexical_score(entity_terms, combined) * 0.5)

    return sum(scores) / len(scores)


def _base_score(candidate: dict, analysis: QueryAnalysis) -> dict:
    semantic = _semantic_score(float(candidate.get("score", 0.0)))
    subject = _lexical_score(analysis.subject_terms or analysis.terms, candidate.get("subject", ""))
    body_text = f"{candidate.get('body_snippet', '')} {candidate.get('text', '')}"
    body = _lexical_score(analysis.subject_terms or analysis.terms, body_text)
    sender_text = f"{candidate.get('sender_name', '')} {candidate.get('sender_email', '')}"
    sender = _lexical_score(analysis.terms, sender_text)
    entity = _entity_score(analysis.entities, candidate)

    base = (
        semantic * settings.RERANK_SEMANTIC_WEIGHT
        + subject * settings.RERANK_SUBJECT_WEIGHT
        + body * settings.RERANK_BODY_WEIGHT
        + sender * settings.RERANK_SENDER_WEIGHT
        + entity * settings.RERANK_ENTITY_WEIGHT
    )

    return {
        "semantic_score": semantic,
        "subject_score": subject,
        "body_score": body,
        "sender_score": sender,
        "entity_score": entity,
        "temporal_score": 0.0,
        "base_score": base,
        "final_score": base,
        "parsed_date": parse_email_date(candidate.get("date")),
    }


def _temporal_scores(scored: list[dict], intent: TemporalIntent) -> dict[str, float]:
    dated = [(item["message_id"], item.get("parsed_date")) for item in scored if item.get("parsed_date")]
    if not dated:
        return {}

    timestamps = [dt.timestamp() for _, dt in dated]
    min_ts = min(timestamps)
    max_ts = max(timestamps)
    span = max(max_ts - min_ts, 1.0)
    now = datetime.now(timezone.utc)

    scores: dict[str, float] = {}
    for message_id, dt in dated:
        recency = (dt.timestamp() - min_ts) / span

        if intent == TemporalIntent.RECENT:
            score = recency
        elif intent == TemporalIntent.EARLIEST:
            score = 1.0 - recency
        elif intent == TemporalIntent.UPCOMING:
            if dt >= now:
                days_until = max((dt - now).total_seconds() / 86400, 0.0)
                score = math.exp(-days_until / 45.0)
            else:
                score = 0.45 * recency
        elif intent == TemporalIntent.TODAY:
            score = _day_closeness(dt, now)
        elif intent == TemporalIntent.TOMORROW:
            score = _day_closeness(dt, now + timedelta(days=1))
        elif intent == TemporalIntent.YESTERDAY:
            score = _day_closeness(dt, now - timedelta(days=1))
        else:
            score = 0.0

        scores[message_id] = max(0.0, min(1.0, score))

    return scores


def _day_closeness(dt: datetime, target: datetime) -> float:
    days = abs((dt.date() - target.date()).days)
    return math.exp(-days / 7.0)


def rerank_candidates(
    query: str,
    analysis: QueryAnalysis,
    candidates: list[dict],
    final_k: int,
    *,
    deduplicate: bool = True,
) -> list[dict]:
    """Deduplicate and rerank raw FAISS candidates."""
    if not candidates:
        return []

    scored_chunks = []
    for candidate in candidates:
        scored = dict(candidate)
        scored.update(_base_score(candidate, analysis))
        scored_chunks.append(scored)

    if deduplicate:
        grouped: dict[str, dict] = {}
        for scored in scored_chunks:
            message_id = scored.get("message_id", "")
            if not message_id:
                continue
            current = grouped.get(message_id)
            if current is None or scored["base_score"] > current["base_score"]:
                grouped[message_id] = scored
        scored_emails = list(grouped.values())
    else:
        scored_emails = scored_chunks

    if analysis.temporal_intent:
        temporal_by_message = _temporal_scores(scored_emails, analysis.temporal_intent)
        for scored in scored_emails:
            temporal = temporal_by_message.get(scored.get("message_id", ""), 0.0)
            scored["temporal_score"] = temporal
            scored["final_score"] = scored["base_score"] + (
                temporal * settings.RERANK_TEMPORAL_WEIGHT
            )

    ranked = sorted(
        scored_emails,
        key=lambda item: (
            item["final_score"],
            item["semantic_score"],
            item.get("parsed_date") or datetime.min.replace(tzinfo=timezone.utc),
        ),
        reverse=True,
    )
    ranked = [
        item for item in ranked
        if item["final_score"] >= settings.MIN_RERANK_SCORE
    ]

    _log_debug_ranking(query, analysis, candidates, ranked, final_k)
    return ranked[:final_k]


def _log_debug_ranking(
    query: str,
    analysis: QueryAnalysis,
    candidates: list[dict],
    ranked: list[dict],
    final_k: int,
) -> None:
    logger.debug(
        "Query: %s | type=%s entities=%s temporal=%s",
        truncate_text(query, 120),
        analysis.query_type.value,
        analysis.entities,
        analysis.temporal_intent.value if analysis.temporal_intent else None,
    )
    logger.debug(
        "Initial candidates: %d | Unique emails after deduplication: %d",
        len(candidates),
        len({item.get("message_id") for item in ranked}),
    )
    logger.debug("Final results:")
    for i, item in enumerate(ranked[:final_k], 1):
        logger.debug(
            "%d. %s | semantic=%.4f subject=%.4f body=%.4f sender=%.4f entity=%.4f "
            "temporal=%.4f final=%.4f message_id=%s",
            i,
            truncate_text(item.get("subject", "(No Subject)"), 120),
            item.get("semantic_score", 0.0),
            item.get("subject_score", 0.0),
            item.get("body_score", 0.0),
            item.get("sender_score", 0.0),
            item.get("entity_score", 0.0),
            item.get("temporal_score", 0.0),
            item.get("final_score", 0.0),
            item.get("message_id", ""),
        )
