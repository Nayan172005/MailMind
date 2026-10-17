"""
MailMind Gmail — Gmail API Client

Wraps the Gmail API for fetching messages, watching mailboxes,
and querying history. All operations use the backend-held credentials.
"""

from typing import Optional

from googleapiclient.discovery import build

from api.gmail.auth import get_credentials
from rag.config import settings
from rag.utils import get_logger

logger = get_logger("gmail.client")

# Cached Gmail service instance
_service = None


def get_service():
    """
    Get or create the Gmail API service.

    Raises RuntimeError if not authenticated.
    """
    global _service

    creds = get_credentials()
    if creds is None:
        raise RuntimeError(
            "Gmail is not authenticated. "
            "Visit /gmail/auth to authorize Gmail access."
        )

    # Rebuild service if credentials changed
    if _service is None:
        _service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        logger.info("Gmail API service created")

    return _service


def reset_service() -> None:
    """Force re-creation of Gmail service (e.g. after re-auth)."""
    global _service
    _service = None


def get_profile() -> dict:
    """Get the authenticated Gmail user's profile."""
    service = get_service()
    return service.users().getProfile(userId="me").execute()


def get_message(message_id: str, format: str = "full") -> dict:
    """
    Fetch a single Gmail message by ID.

    Args:
        message_id: Gmail message ID.
        format: 'full', 'metadata', 'minimal', or 'raw'.

    Returns:
        Gmail message resource dict.
    """
    service = get_service()
    return (
        service.users()
        .messages()
        .get(userId="me", id=message_id, format=format)
        .execute()
    )


def list_messages(
    query: str = "",
    max_results: int = 50,
    page_token: Optional[str] = None,
) -> dict:
    """
    List messages matching a query.

    Args:
        query: Gmail search query (e.g. 'is:unread', 'newer_than:7d').
        max_results: Maximum messages to return.
        page_token: Pagination token.

    Returns:
        Gmail listMessages response.
    """
    service = get_service()
    params = {
        "userId": "me",
        "maxResults": max_results,
    }
    if query:
        params["q"] = query
    if page_token:
        params["pageToken"] = page_token

    return service.users().messages().list(**params).execute()


def get_history(
    start_history_id: str,
    history_types: Optional[list[str]] = None,
    max_results: int = 100,
) -> dict:
    """
    Get mailbox history changes since a given historyId.

    Args:
        start_history_id: The historyId to start from.
        history_types: Types of history to return (e.g. ['messageAdded']).
        max_results: Maximum history records.

    Returns:
        Gmail history response.
    """
    service = get_service()
    params = {
        "userId": "me",
        "startHistoryId": start_history_id,
        "maxResults": max_results,
    }
    if history_types:
        params["historyTypes"] = history_types

    return service.users().history().list(**params).execute()


def setup_watch(
    topic_name: Optional[str] = None,
    label_ids: Optional[list[str]] = None,
) -> dict:
    """
    Set up Gmail push notifications via Pub/Sub.

    Args:
        topic_name: Pub/Sub topic (defaults to settings).
        label_ids: Label IDs to watch (defaults to INBOX).

    Returns:
        Gmail watch response with historyId and expiration.
    """
    service = get_service()
    topic = topic_name or settings.GMAIL_PUBSUB_TOPIC

    if not topic:
        raise RuntimeError(
            "GMAIL_PUBSUB_TOPIC is not configured. "
            "Set it to your Google Cloud Pub/Sub topic."
        )

    body = {
        "topicName": topic,
        "labelIds": label_ids or ["INBOX"],
    }

    result = service.users().watch(userId="me", body=body).execute()
    logger.info(
        f"Gmail watch established. historyId={result.get('historyId')}, "
        f"expiration={result.get('expiration')}"
    )
    return result


def stop_watch() -> None:
    """Stop receiving Gmail push notifications."""
    service = get_service()
    service.users().stop(userId="me").execute()
    logger.info("Gmail watch stopped")
