"""
MailMind RAG - metadata-aware candidate reranking.

FAISS semantic similarity serves as the primary anchor. High-precision secondary
signals (entity coverage, explicit sender matching, venue tokens, date constraints,
and controllable temporal intent) calibrate ordering and eliminate false positives.
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
    """Calculate multi-entity coverage across subject, sender, and body."""
    if not entities:
        return 0.0

    subject = candidate.get("subject", "")
    sender = f"{candidate.get('sender_name', '')} {candidate.get('sender_email', '')}"
    body = f"{candidate.get('body_snippet', '')} {candidate.get('text', '')}"

    single_ents = [e for e in entities if " " not in e and len(e) > 1]
    target_ents = single_ents if single_ents else entities

    matched_val = 0.0
    for entity in target_ents:
        if _phrase_present(entity, subject):
            matched_val += 1.0
        elif _phrase_present(entity, sender):
            matched_val += 0.85
        elif _phrase_present(entity, body):
            matched_val += 0.60
        else:
            entity_terms = tokenize(entity)
            combined = f"{subject} {sender} {body}"
            lex = _lexical_score(entity_terms, combined)
            if lex > 0.4:
                matched_val += 0.4

    return min(1.0, matched_val / max(len(target_ents), 1))


def _sender_score(senders: list[str], candidate: dict, fallback_terms: list[str]) -> float:
    """Calculate sender match prioritizing explicit sender query constraints."""
    sender_name = candidate.get("sender_name", "")
    sender_email = candidate.get("sender_email", "")
    sender_text = f"{sender_name} {sender_email}".lower()

    if senders:
        best_score = 0.0
        for snd in senders:
            snd_lower = snd.lower()
            if snd_lower in sender_text:
                best_score = max(best_score, 1.0)
            else:
                snd_tokens = tokenize(snd)
                if snd_tokens:
                    matched = sum(1 for t in snd_tokens if t in sender_text)
                    token_score = matched / len(snd_tokens)
                    if token_score > 0.5:
                        best_score = max(best_score, 0.85 * token_score)
        return best_score

    return _lexical_score(fallback_terms, sender_text)


def _venue_score(venues: list[str], candidate: dict) -> float:
    """Score matches against room/building/venue identifiers."""
    if not venues:
        return 0.0

    subject = candidate.get("subject", "")
    body = f"{candidate.get('body_snippet', '')} {candidate.get('text', '')}"

    for v in venues:
        if _phrase_present(v, subject):
            return 1.0
        if _phrase_present(v, body):
            return 0.80
    return 0.0


def _date_constraint_score(date_tokens: list[str], candidate: dict, semantic: float) -> float:
    """
    Reward matching explicit date/batch tokens, and penalize candidates lacking
    the requested date token when semantic confidence is low.
    """
    if not date_tokens:
        return 0.0

    subject = candidate.get("subject", "")
    body = f"{candidate.get('body_snippet', '')} {candidate.get('text', '')}"

    has_match = False
    for dt in date_tokens:
        if _phrase_present(dt, subject):
            has_match = True
            return 0.15
        elif _phrase_present(dt, body):
            has_match = True
            return 0.08

    # Date token explicitly requested in query but completely missing in low-scoring candidate
    if not has_match and semantic < 0.50:
        return -0.15

    return 0.0


def _base_score(candidate: dict, analysis: QueryAnalysis) -> dict:
    semantic = _semantic_score(float(candidate.get("score", 0.0)))
    subject = _lexical_score(analysis.subject_terms or analysis.terms, candidate.get("subject", ""))
    body_text = f"{candidate.get('body_snippet', '')} {candidate.get('text', '')}"
    body = _lexical_score(analysis.subject_terms or analysis.terms, body_text)
    sender = _sender_score(analysis.senders, candidate, analysis.terms)
    entity = _entity_score(analysis.entities, candidate)
    venue = _venue_score(analysis.venues, candidate)
    date_constraint = _date_constraint_score(analysis.date_tokens, candidate, semantic)

    base = (
        semantic * settings.RERANK_SEMANTIC_WEIGHT
        + subject * settings.RERANK_SUBJECT_WEIGHT
        + body * settings.RERANK_BODY_WEIGHT
        + sender * settings.RERANK_SENDER_WEIGHT
        + entity * settings.RERANK_ENTITY_WEIGHT
        + (venue * 0.10 if analysis.venues else 0.0)
        + date_constraint
    )

    return {
        "semantic_score": semantic,
        "subject_score": subject,
        "body_score": body,
        "sender_score": sender,
        "entity_score": entity,
        "venue_score": venue,
        "date_constraint": date_constraint,
        "temporal_score": 0.0,
        "base_score": base,
        "final_score": base,
        "parsed_date": parse_email_date(candidate.get("date")),
    }


def _temporal_scores(
    scored: list[dict],
    intent: TemporalIntent,
    reference_time: datetime | None = None,
) -> dict[str, float]:
    dated = [(item["message_id"], item.get("parsed_date")) for item in scored if item.get("parsed_date")]
    if not dated:
        return {}

    timestamps = [dt.timestamp() for _, dt in dated]
    min_ts = min(timestamps)
    max_ts = max(timestamps)
    span = max(max_ts - min_ts, 1.0)
    now = reference_time or datetime.now(timezone.utc)

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
    reference_time: datetime | None = None,
) -> list[dict]:
    """Deduplicate and rerank raw candidates using multi-signal scoring."""
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
        temporal_by_message = _temporal_scores(
            scored_emails,
            analysis.temporal_intent,
            reference_time=reference_time,
        )
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
        "Query: %s | type=%s entities=%s senders=%s date_tokens=%s temporal=%s",
        truncate_text(query, 120),
        analysis.query_type.value,
        analysis.entities,
        analysis.senders,
        analysis.date_tokens,
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
