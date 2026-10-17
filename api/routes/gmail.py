"""
MailMind API — Gmail Integration Routes

Endpoints:
    GET  /gmail/auth          — Get OAuth authorization URL
    GET  /gmail/oauth/callback — OAuth callback handler
    POST /gmail/webhook       — Pub/Sub push notification receiver
    POST /gmail/sync          — Manual message sync
    POST /gmail/watch         — Create/renew Gmail watch
    POST /gmail/watch/stop    — Stop Gmail watch
    GET  /gmail/status        — Connection and watch status
    GET  /gmail/recent        — Recently processed emails
    POST /gmail/poll/start    — Start polling fallback
    POST /gmail/poll/stop     — Stop polling fallback
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional

from rag.config import settings
from rag.utils import get_logger

logger = get_logger("api.gmail")

router = APIRouter(prefix="/gmail", tags=["Gmail Integration"])


# ─── Request/Response Models ─────────────────────────────────────────

class SyncRequest(BaseModel):
    max_results: int = Field(default=50, ge=1, le=200)
    days: Optional[int] = Field(default=None, ge=1, le=365)


class GmailStatusResponse(BaseModel):
    authenticated: bool = False
    email: str = ""
    gmail_enabled: bool = False
    watch_status: dict = Field(default_factory=dict)
    ingestion_stats: dict = Field(default_factory=dict)
    index_vectors: int = 0
    index_emails: int = 0


# ─── OAuth Endpoints ─────────────────────────────────────────────────

@router.get("/auth")
async def gmail_auth():
    """
    Get the Gmail OAuth authorization URL.

    The user should visit this URL to grant Gmail access.
    """
    try:
        from api.gmail.auth import get_auth_url
        url = get_auth_url()
        return {"auth_url": url}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/oauth/callback")
async def gmail_oauth_callback(
    code: str = "",
    state: str = "",
    error: Optional[str] = None,
):
    """
    OAuth callback endpoint.

    Google redirects here after the user authorizes access.
    The state parameter is validated against the persisted PKCE state.
    """
    if error:
        raise HTTPException(status_code=400, detail=f"OAuth error: {error}")

    if not code:
        raise HTTPException(status_code=400, detail="No authorization code received.")

    if not state:
        raise HTTPException(
            status_code=400,
            detail="Missing OAuth state parameter. Please restart authorization at /gmail/auth.",
        )

    try:
        from api.gmail.auth import handle_callback
        from api.gmail.client import reset_service
        handle_callback(code, state)
        reset_service()  # Force service re-creation with new creds
        return {
            "status": "success",
            "message": "Gmail authenticated successfully. You can now sync emails.",
        }
    except RuntimeError as e:
        # State mismatch or expired session — tell user to restart
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"OAuth callback failed: {e}")
        raise HTTPException(status_code=500, detail=f"OAuth failed: {e}")


# ─── Pub/Sub Webhook ─────────────────────────────────────────────────

def verify_pubsub_jwt(request: Request) -> dict:
    """
    Validate the Google Cloud Pub/Sub push notification OIDC JWT token.

    Checks:
    - Authorization: Bearer <token> header present and well-formed.
    - Signature, expiration, and issuer (accounts.google.com).
    - Exact configured audience (settings.GMAIL_PUBSUB_AUDIENCE).
    - Service-account email matches settings.GMAIL_PUBSUB_SERVICE_ACCOUNT.
    - email_verified is True.

    Raises:
        HTTPException(401): Missing or malformed Authorization header.
        HTTPException(403): Invalid/expired token, wrong audience, wrong service account, or unverified email.
        HTTPException(500): Server authentication settings not configured.
    """
    if not settings.GMAIL_PUBSUB_VERIFY_TOKEN:
        logger.warning(
            "Pub/Sub JWT token verification is disabled via GMAIL_PUBSUB_VERIFY_TOKEN=False."
        )
        return {}

    auth_header = request.headers.get("Authorization")
    if not auth_header:
        logger.warning("Pub/Sub webhook rejected: Missing Authorization header")
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        logger.warning("Pub/Sub webhook rejected: Malformed Authorization header")
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header format. Expected 'Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1].strip()

    if not settings.GMAIL_PUBSUB_AUDIENCE:
        logger.error(
            "Pub/Sub webhook rejected: GMAIL_PUBSUB_AUDIENCE is not configured on the server."
        )
        raise HTTPException(
            status_code=500,
            detail="Pub/Sub audience verification is not configured on the server",
        )

    if not settings.GMAIL_PUBSUB_SERVICE_ACCOUNT:
        logger.error(
            "Pub/Sub webhook rejected: GMAIL_PUBSUB_SERVICE_ACCOUNT is not configured on the server."
        )
        raise HTTPException(
            status_code=500,
            detail="Pub/Sub service-account verification is not configured on the server",
        )

    from google.oauth2 import id_token
    from google.auth.transport import requests as google_requests

    try:
        request_adapter = google_requests.Request()
        claims = id_token.verify_oauth2_token(
            token,
            request_adapter,
            audience=settings.GMAIL_PUBSUB_AUDIENCE,
        )
    except ValueError as e:
        logger.warning(f"Pub/Sub JWT verification failed: {e}")
        raise HTTPException(
            status_code=403,
            detail=f"Token verification failed: {e}",
        )
    except Exception as e:
        logger.error(f"Unexpected error during Pub/Sub JWT verification: {e}")
        raise HTTPException(
            status_code=403,
            detail="Token verification failed",
        )

    # Validate issuer
    issuer = claims.get("iss")
    if issuer not in ("accounts.google.com", "https://accounts.google.com"):
        logger.warning(f"Pub/Sub JWT issuer mismatch: {issuer}")
        raise HTTPException(
            status_code=403,
            detail=f"Invalid token issuer: {issuer}",
        )

    # Validate service account email
    token_email = claims.get("email")
    if not token_email or token_email.lower() != settings.GMAIL_PUBSUB_SERVICE_ACCOUNT.lower():
        logger.warning(
            f"Pub/Sub JWT service account mismatch: got {token_email}, "
            f"expected {settings.GMAIL_PUBSUB_SERVICE_ACCOUNT}"
        )
        raise HTTPException(
            status_code=403,
            detail="Unexpected service-account email",
        )

    # Validate email_verified
    if claims.get("email_verified") is not True:
        logger.warning("Pub/Sub JWT service account email is not verified")
        raise HTTPException(
            status_code=403,
            detail="Service-account email is not verified",
        )

    return claims


@router.post("/webhook")
async def gmail_webhook(request: Request):
    """
    Receive Gmail push notifications via Google Cloud Pub/Sub.

    Validates Google OIDC JWT bearer authentication.
    Processes notification history idempotently and acknowledges with 200 on success.
    Returns retryable non-2xx for transient failures.
    """
    # 1. Authenticate Google Pub/Sub OIDC JWT
    verify_pubsub_jwt(request)

    # 2. Check if FAISS index is loaded
    from api.main import pipeline

    if not pipeline.is_ready:
        logger.warning("Webhook received but FAISS index is not loaded")
        raise HTTPException(
            status_code=503,
            detail="Index not ready; please retry",
        )

    # 3. Parse and validate JSON envelope
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400,
            detail="Invalid Pub/Sub notification format: expected JSON object",
        )

    message = body.get("message")
    if not isinstance(message, dict):
        raise HTTPException(
            status_code=400,
            detail="Invalid Pub/Sub notification format: missing 'message' object",
        )

    encoded_data = message.get("data")
    if not encoded_data or not isinstance(encoded_data, str) or not encoded_data.strip():
        raise HTTPException(
            status_code=400,
            detail="Invalid Pub/Sub notification format: missing or empty 'message.data'",
        )

    # 4. Process notification
    try:
        from api.gmail.watcher import process_pubsub_notification

        results = process_pubsub_notification(body, pipeline.store)

        summary = {}
        for r in results:
            s = r.get("status", "unknown")
            summary[s] = summary.get(s, 0) + 1

        logger.info(f"Webhook processed successfully: {summary} (total={len(results)})")
        return {"status": "ok", "processed": len(results), "summary": summary}

    except ValueError as e:
        # Invalid Base64, missing fields, or malformed JSON in message.data
        logger.warning(f"Malformed Pub/Sub message data: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transient webhook processing error: {e}", exc_info=True)
        # Return HTTP 500 so Pub/Sub will retry
        raise HTTPException(
            status_code=500,
            detail="Transient processing error; please retry",
        )


# ─── Sync Endpoints ──────────────────────────────────────────────────

@router.post("/sync")
async def manual_sync(request: SyncRequest = SyncRequest()):
    """
    Manually trigger Gmail synchronization.

    Fetches recent messages and processes them through the
    classifier → embedding → FAISS pipeline.
    """
    from api.gmail.auth import is_authenticated
    if not is_authenticated():
        raise HTTPException(
            status_code=401,
            detail="Gmail not authenticated. Visit /gmail/auth first."
        )

    from api.main import pipeline
    if not pipeline.is_ready:
        raise HTTPException(status_code=503, detail="Index not loaded.")

    try:
        from api.gmail.watcher import sync_recent_messages
        results = sync_recent_messages(
            pipeline.store,
            max_results=request.max_results,
            days=request.days,
        )

        summary = {}
        for r in results:
            summary[r["status"]] = summary.get(r["status"], 0) + 1

        return {
            "status": "ok",
            "total_processed": len(results),
            "summary": summary,
            "results": results,
        }
    except Exception as e:
        logger.error(f"Manual sync failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Watch Endpoints ─────────────────────────────────────────────────

@router.post("/watch")
async def create_watch():
    """
    Create or renew a Gmail watch for push notifications.

    Requires GMAIL_PUBSUB_TOPIC to be configured.
    """
    from api.gmail.auth import is_authenticated
    if not is_authenticated():
        raise HTTPException(status_code=401, detail="Gmail not authenticated.")

    try:
        from api.gmail.watcher import setup_watch
        result = setup_watch()
        return {
            "status": "ok",
            "history_id": result.get("historyId"),
            "expiration": result.get("expiration"),
        }
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Watch setup failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/watch/stop")
async def stop_watch_endpoint():
    """Stop Gmail push notifications."""
    try:
        from api.gmail.watcher import stop_watch
        stop_watch()
        return {"status": "ok", "message": "Gmail watch stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Status / Debug Endpoints ────────────────────────────────────────

@router.get("/status", response_model=GmailStatusResponse)
async def gmail_status():
    """
    Get Gmail integration status.

    Shows authentication status, watch status, and ingestion statistics.
    """
    from api.gmail.auth import is_authenticated, get_credentials
    from api.gmail.ingestion import stats
    from api.gmail.watcher import get_watch_status
    from api.main import pipeline
    from rag.config import settings

    email = ""
    authenticated = is_authenticated()
    if authenticated:
        try:
            profile = gmail_client_profile()
            email = profile.get("emailAddress", "")
        except Exception:
            pass

    return GmailStatusResponse(
        authenticated=authenticated,
        email=email,
        gmail_enabled=settings.GMAIL_ENABLED,
        watch_status=get_watch_status(),
        ingestion_stats=stats.to_dict(),
        index_vectors=pipeline.store.total_vectors if pipeline.is_ready else 0,
        index_emails=pipeline.store.total_emails if pipeline.is_ready else 0,
    )


@router.get("/recent")
async def gmail_recent():
    """
    Show recently processed emails.

    Returns the last 50 processed emails with their classification
    and ingestion status.
    """
    from api.gmail.ingestion import stats

    return {
        "recent_emails": stats.recent_emails[:20],
        "stats": stats.to_dict(),
    }


# ─── Polling Endpoints ───────────────────────────────────────────────

@router.post("/poll/start")
async def start_polling():
    """
    Start background polling for new Gmail messages.

    Use this when Pub/Sub push notifications are not configured.
    """
    from api.gmail.auth import is_authenticated
    if not is_authenticated():
        raise HTTPException(status_code=401, detail="Gmail not authenticated.")

    from api.main import pipeline
    if not pipeline.is_ready:
        raise HTTPException(status_code=503, detail="Index not loaded.")

    from api.gmail.watcher import start_polling as _start_polling
    await _start_polling(pipeline.store)

    return {
        "status": "ok",
        "message": f"Polling started (interval={settings.GMAIL_POLL_INTERVAL}s)",
    }


@router.post("/poll/stop")
async def stop_polling_endpoint():
    """Stop background polling."""
    from api.gmail.watcher import stop_polling
    await stop_polling()
    return {"status": "ok", "message": "Polling stopped"}


# ─── Helper ──────────────────────────────────────────────────────────

def gmail_client_profile() -> dict:
    """Get Gmail profile (helper to avoid circular imports)."""
    from api.gmail.client import get_profile
    return get_profile()
