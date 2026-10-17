"""
MailMind Gmail — Watch & History Synchronization

Manages Gmail Watch (Pub/Sub push notifications) and History API
synchronization. Provides both push-based and polling fallback.

State (historyId, watch expiration) is persisted to disk so it
survives application restarts.
"""

import asyncio
import base64
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from api.gmail import client as gmail_client
from api.gmail.ingestion import ingest_gmail_messages
from rag.config import settings
from rag.vector_store import VectorStore
from rag.utils import get_logger

logger = get_logger("gmail.watcher")


# ─── Watch State Persistence ─────────────────────────────────────────

class WatchState:
    """Persistent state for Gmail watch/history tracking."""

    def __init__(self):
        self.history_id: Optional[str] = None
        self.watch_expiration: Optional[int] = None  # Unix ms
        self.last_notification_time: Optional[str] = None
        self.is_watching: bool = False

    def to_dict(self) -> dict:
        return {
            "history_id": self.history_id,
            "watch_expiration": self.watch_expiration,
            "last_notification_time": self.last_notification_time,
            "is_watching": self.is_watching,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WatchState":
        state = cls()
        state.history_id = data.get("history_id")
        state.watch_expiration = data.get("watch_expiration")
        state.last_notification_time = data.get("last_notification_time")
        state.is_watching = data.get("is_watching", False)
        return state

    def save(self) -> None:
        """Persist state to disk."""
        state_path = settings.gmail_state_abs_path
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls) -> "WatchState":
        """Load state from disk, or return empty state."""
        state_path = settings.gmail_state_abs_path
        if not state_path.exists():
            return cls()
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load watch state: {e}")
            return cls()


# Global state
watch_state = WatchState.load()

# Background polling task handle
_poll_task: Optional[asyncio.Task] = None


# ─── Gmail Watch (Push Notifications) ────────────────────────────────

def setup_watch() -> dict:
    """
    Set up Gmail push notifications via Pub/Sub.

    Returns:
        Watch result dict with historyId and expiration.
    """
    result = gmail_client.setup_watch()

    watch_state.history_id = str(result.get("historyId", ""))
    watch_state.watch_expiration = int(result.get("expiration", 0))
    watch_state.is_watching = True
    watch_state.save()

    logger.info(
        f"Gmail watch set up. historyId={watch_state.history_id}, "
        f"expiration={watch_state.watch_expiration}"
    )
    return result


def stop_watch() -> None:
    """Stop Gmail push notifications."""
    try:
        gmail_client.stop_watch()
    except Exception as e:
        logger.warning(f"Error stopping watch: {e}")

    watch_state.is_watching = False
    watch_state.save()


def renew_watch_if_needed() -> Optional[dict]:
    """
    Renew the Gmail watch if it's about to expire.

    Returns:
        New watch result if renewed, None otherwise.
    """
    if not watch_state.watch_expiration:
        return None

    now_ms = int(time.time() * 1000)
    buffer_ms = settings.GMAIL_WATCH_EXPIRATION_BUFFER * 1000

    if now_ms >= (watch_state.watch_expiration - buffer_ms):
        logger.info("Gmail watch expiring soon, renewing...")
        return setup_watch()

    return None


# ─── Pub/Sub Notification Processing ─────────────────────────────────

def process_pubsub_notification(
    notification_data: dict,
    store: VectorStore,
) -> list[dict]:
    """
    Process a Pub/Sub notification from Gmail.

    Steps:
    1. Validate and decode notification to get email address and historyId
    2. Check idempotency against stored historyId
    3. Use History API to find newly added messages
    4. Fetch and ingest each new message
    5. Update stored historyId on success

    Args:
        notification_data: Pub/Sub push message data.
        store: Loaded VectorStore.

    Returns:
        List of ingestion results.

    Raises:
        ValueError: If notification envelope or data is malformed.
        Exception: If transient Gmail API / ingestion failure occurs.
    """
    watch_state.last_notification_time = datetime.now(timezone.utc).isoformat()

    if not isinstance(notification_data, dict):
        raise ValueError("Invalid notification envelope: payload must be a JSON object")

    message = notification_data.get("message")
    if not isinstance(message, dict):
        raise ValueError("Invalid notification envelope: missing 'message' object")

    encoded_data = message.get("data")
    if not encoded_data or not isinstance(encoded_data, str) or not encoded_data.strip():
        raise ValueError("Invalid notification envelope: missing or empty 'message.data'")

    try:
        decoded_bytes = base64.b64decode(encoded_data, validate=True)
        decoded = decoded_bytes.decode("utf-8")
        notification = json.loads(decoded)
    except Exception as e:
        logger.warning(f"Failed to decode Pub/Sub notification payload: {e}")
        raise ValueError(f"Malformed Pub/Sub message data: {e}")

    if not isinstance(notification, dict):
        raise ValueError("Decoded Pub/Sub payload is not a JSON object")

    new_history_id = str(notification.get("historyId", "")).strip()
    email_address = notification.get("emailAddress", "")

    if not new_history_id:
        raise ValueError("Pub/Sub notification missing required 'historyId'")

    logger.info(
        f"Pub/Sub notification: email={email_address}, "
        f"historyId={new_history_id}"
    )

    if not watch_state.history_id:
        logger.warning("No stored historyId. Setting from notification.")
        watch_state.history_id = new_history_id
        watch_state.save()
        return []

    # Idempotency check: ignore duplicate or past historyId notifications
    try:
        curr_id = int(watch_state.history_id)
        incoming_id = int(new_history_id)
        if incoming_id <= curr_id:
            logger.info(
                f"Notification historyId ({incoming_id}) <= current historyId ({curr_id}); "
                "skipping duplicate or past notification."
            )
            return []
    except (ValueError, TypeError):
        if new_history_id == watch_state.history_id:
            logger.info(
                f"Duplicate notification historyId ({new_history_id}); skipping."
            )
            return []

    # Fetch and process history
    results = _process_history(watch_state.history_id, store)

    # Update stored historyId only on successful processing
    try:
        if int(new_history_id) > int(watch_state.history_id):
            watch_state.history_id = new_history_id
            watch_state.save()
    except (ValueError, TypeError):
        watch_state.history_id = new_history_id
        watch_state.save()

    return results


def _process_history(
    start_history_id: str,
    store: VectorStore,
) -> list[dict]:
    """
    Fetch Gmail history since a given historyId and ingest new messages.

    Handles history expiration by falling back to recent message listing.
    """
    try:
        history_response = gmail_client.get_history(
            start_history_id=start_history_id,
            history_types=["messageAdded"],
        )
    except Exception as e:
        error_str = str(e)
        if "404" in error_str or "historyId" in error_str.lower():
            logger.warning(
                f"History ID {start_history_id} expired or invalid. "
                "Falling back to recent messages sync."
            )
            return _fallback_recent_sync(store)
        raise

    history_records = history_response.get("history", [])
    if not history_records:
        logger.info("No new history records")
        return []

    # Collect all newly added message IDs
    new_message_ids = set()
    for record in history_records:
        for msg_added in record.get("messagesAdded", []):
            msg = msg_added.get("message", {})
            msg_id = msg.get("id", "")
            if msg_id:
                new_message_ids.add(msg_id)

    if not new_message_ids:
        logger.info("No new messages in history")
        return []

    logger.info(f"Found {len(new_message_ids)} new message(s) in history")

    # Fetch full messages
    full_messages = []
    for msg_id in new_message_ids:
        try:
            full_msg = gmail_client.get_message(msg_id)
            full_messages.append(full_msg)
        except Exception as e:
            logger.error(f"Failed to fetch message {msg_id}: {e}")

    if not full_messages:
        return []

    # Ingest
    return ingest_gmail_messages(full_messages, store)


def _fallback_recent_sync(store: VectorStore) -> list[dict]:
    """
    Fallback synchronization when history ID is invalid.

    Fetches recent messages and processes them (dedup handles overlap).
    """
    logger.info("Running fallback recent sync...")
    return sync_recent_messages(store, max_results=20)


# ─── Initial / Manual Synchronization ────────────────────────────────

def sync_recent_messages(
    store: VectorStore,
    max_results: int = 50,
    days: Optional[int] = None,
) -> list[dict]:
    """
    Manually sync recent Gmail messages into the FAISS index.

    Args:
        store: Loaded VectorStore.
        max_results: Maximum messages to fetch.
        days: Only fetch messages from the last N days.

    Returns:
        List of ingestion results.
    """
    query = ""
    sync_days = days or settings.GMAIL_INITIAL_SYNC_DAYS
    if sync_days > 0:
        query = f"newer_than:{sync_days}d"

    max_results = min(max_results, settings.GMAIL_INITIAL_SYNC_MAX)

    logger.info(f"Syncing recent messages: query='{query}', max={max_results}")

    try:
        list_result = gmail_client.list_messages(
            query=query, max_results=max_results
        )
    except Exception as e:
        logger.error(f"Failed to list messages: {e}")
        return []

    message_stubs = list_result.get("messages", [])
    if not message_stubs:
        logger.info("No messages found for sync")
        return []

    logger.info(f"Found {len(message_stubs)} messages to sync")

    # Fetch full messages
    full_messages = []
    for stub in message_stubs:
        msg_id = stub.get("id", "")
        try:
            full_msg = gmail_client.get_message(msg_id)
            full_messages.append(full_msg)
        except Exception as e:
            logger.error(f"Failed to fetch message {msg_id}: {e}")

    if not full_messages:
        return []

    # Update history ID from profile
    try:
        profile = gmail_client.get_profile()
        watch_state.history_id = str(profile.get("historyId", ""))
        watch_state.save()
    except Exception as e:
        logger.warning(f"Failed to update historyId from profile: {e}")

    return ingest_gmail_messages(full_messages, store)


# ─── Polling Fallback ────────────────────────────────────────────────

async def start_polling(store: VectorStore) -> None:
    """
    Start background polling for new Gmail messages.

    Use this as a fallback when Pub/Sub is not configured.
    """
    global _poll_task

    if _poll_task and not _poll_task.done():
        logger.warning("Polling task already running")
        return

    _poll_task = asyncio.create_task(_polling_loop(store))
    logger.info(
        f"Gmail polling started (interval={settings.GMAIL_POLL_INTERVAL}s)"
    )


async def stop_polling() -> None:
    """Stop the background polling task."""
    global _poll_task

    if _poll_task and not _poll_task.done():
        _poll_task.cancel()
        try:
            await _poll_task
        except asyncio.CancelledError:
            pass
        logger.info("Gmail polling stopped")

    _poll_task = None


async def _polling_loop(store: VectorStore) -> None:
    """Background loop that periodically checks for new messages."""
    while True:
        try:
            await asyncio.sleep(settings.GMAIL_POLL_INTERVAL)

            if watch_state.history_id:
                _process_history(watch_state.history_id, store)

                # Update history ID
                try:
                    profile = gmail_client.get_profile()
                    watch_state.history_id = str(profile.get("historyId", ""))
                    watch_state.save()
                except Exception as e:
                    logger.warning(f"Polling: failed to update historyId: {e}")
            else:
                # No history ID yet, do a recent sync
                sync_recent_messages(store, max_results=10)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Polling error: {e}", exc_info=True)
            await asyncio.sleep(30)  # Back off on error


def get_watch_status() -> dict:
    """Get the current watch/polling status."""
    now_ms = int(time.time() * 1000)
    expiration = watch_state.watch_expiration

    status = watch_state.to_dict()
    status["watch_active"] = (
        watch_state.is_watching
        and expiration is not None
        and now_ms < expiration
    )
    status["polling_active"] = _poll_task is not None and not _poll_task.done()

    if expiration:
        remaining_s = max(0, (expiration - now_ms) / 1000)
        status["watch_expires_in_seconds"] = int(remaining_s)

    return status
