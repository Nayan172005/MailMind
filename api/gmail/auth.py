"""
MailMind Gmail — OAuth 2.0 Authentication

Backend-only OAuth handling for Gmail API.
Tokens are stored on disk and never exposed to the frontend.

PKCE flow:
    1. get_auth_url() creates a Flow, calls authorization_url() which
       generates a code_verifier + code_challenge, and persists
       {state, code_verifier} to disk.
    2. handle_callback(code, state) loads the persisted state, rebuilds
       the Flow with the original code_verifier, validates the state
       parameter, then calls fetch_token().
"""

import json
import os
from pathlib import Path
from typing import Optional

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow

from rag.config import settings
from rag.utils import get_logger

logger = get_logger("gmail.auth")

# Cached credentials instance
_credentials: Optional[Credentials] = None


def _get_client_config() -> dict:
    """Build OAuth client config from environment variables."""
    if not settings.GMAIL_CLIENT_ID or not settings.GMAIL_CLIENT_SECRET:
        raise RuntimeError(
            "GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET must be set. "
            "See .env.example for required configuration."
        )
    return {
        "web": {
            "client_id": settings.GMAIL_CLIENT_ID,
            "client_secret": settings.GMAIL_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [settings.GMAIL_REDIRECT_URI],
        }
    }


def _oauth_state_path() -> Path:
    """Path to the temporary file that stores PKCE state between requests."""
    return settings.gmail_token_abs_path.parent / "oauth_pending.json"


def _save_oauth_state(state: str, code_verifier: str) -> None:
    """Persist the OAuth state + PKCE code_verifier to disk."""
    path = _oauth_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"state": state, "code_verifier": code_verifier}, f)


def _load_and_clear_oauth_state() -> Optional[dict]:
    """
    Load the persisted OAuth state and delete the file.

    Returns:
        Dict with 'state' and 'code_verifier', or None if missing.
    """
    path = _oauth_state_path()
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Remove the file so it can't be replayed
        path.unlink(missing_ok=True)
        return data
    except Exception as e:
        logger.error(f"Failed to load OAuth state: {e}")
        path.unlink(missing_ok=True)
        return None


def get_auth_url() -> str:
    """
    Generate the OAuth 2.0 authorization URL.

    Creates a PKCE code_verifier, persists it along with the state
    parameter, and returns the authorization URL for the user to visit.

    Returns:
        The URL the user should visit to authorize Gmail access.
    """
    flow = Flow.from_client_config(
        _get_client_config(),
        scopes=settings.GMAIL_SCOPES.split(","),
        redirect_uri=settings.GMAIL_REDIRECT_URI,
    )
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )

    # Persist state + PKCE code_verifier so handle_callback can restore them
    _save_oauth_state(state, flow.code_verifier)

    logger.info("Generated Gmail OAuth URL (PKCE state persisted)")
    return auth_url


def handle_callback(authorization_code: str, state: str) -> Credentials:
    """
    Exchange the authorization code for credentials and persist tokens.

    Restores the PKCE code_verifier from the persisted state and validates
    the state parameter before exchanging the code.

    Args:
        authorization_code: Code received from OAuth callback.
        state: OAuth state parameter from the callback URL.

    Returns:
        Authenticated Credentials object.

    Raises:
        RuntimeError: If state is missing, expired, or doesn't match.
    """
    global _credentials

    # Load the persisted state + code_verifier
    saved = _load_and_clear_oauth_state()
    if saved is None:
        raise RuntimeError(
            "OAuth session expired or not found. "
            "Please restart authorization by visiting /gmail/auth."
        )

    # Validate the state parameter (CSRF protection)
    if state != saved["state"]:
        raise RuntimeError(
            "OAuth state mismatch (possible CSRF). "
            "Please restart authorization by visiting /gmail/auth."
        )

    # Rebuild the Flow with the ORIGINAL code_verifier
    flow = Flow.from_client_config(
        _get_client_config(),
        scopes=settings.GMAIL_SCOPES.split(","),
        redirect_uri=settings.GMAIL_REDIRECT_URI,
        state=saved["state"],
    )
    flow.code_verifier = saved["code_verifier"]

    flow.fetch_token(code=authorization_code)
    _credentials = flow.credentials

    _save_token(_credentials)
    logger.info("Gmail OAuth tokens saved successfully")
    return _credentials


def get_credentials() -> Optional[Credentials]:
    """
    Get valid Gmail credentials.

    Loads from cache, then disk. Refreshes expired tokens automatically.
    Returns None if no credentials are available.
    """
    global _credentials

    if _credentials and _credentials.valid:
        return _credentials

    # Try loading from disk
    if _credentials is None:
        _credentials = _load_token()

    if _credentials is None:
        return None

    # Refresh if expired
    if _credentials.expired and _credentials.refresh_token:
        try:
            from google.auth.transport.requests import Request
            _credentials.refresh(Request())
            _save_token(_credentials)
            logger.info("Gmail credentials refreshed")
        except Exception as e:
            logger.error(f"Failed to refresh Gmail credentials: {e}")
            _credentials = None
            return None

    return _credentials


def is_authenticated() -> bool:
    """Check if valid Gmail credentials are available."""
    creds = get_credentials()
    return creds is not None and creds.valid


def revoke_credentials() -> None:
    """Revoke stored credentials and clear the token file."""
    global _credentials
    _credentials = None

    token_path = settings.gmail_token_abs_path
    if token_path.exists():
        token_path.unlink()
        logger.info("Gmail credentials revoked and token file removed")

    # Also clean up any pending OAuth state
    pending = _oauth_state_path()
    if pending.exists():
        pending.unlink(missing_ok=True)


def _save_token(credentials: Credentials) -> None:
    """Persist credentials to disk as JSON."""
    token_path = settings.gmail_token_abs_path
    token_path.parent.mkdir(parents=True, exist_ok=True)

    token_data = {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": list(credentials.scopes) if credentials.scopes else [],
    }

    with open(token_path, "w", encoding="utf-8") as f:
        json.dump(token_data, f, indent=2)


def _load_token() -> Optional[Credentials]:
    """Load credentials from disk."""
    token_path = settings.gmail_token_abs_path
    if not token_path.exists():
        return None

    try:
        with open(token_path, "r", encoding="utf-8") as f:
            token_data = json.load(f)

        creds = Credentials(
            token=token_data.get("token"),
            refresh_token=token_data.get("refresh_token"),
            token_uri=token_data.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=token_data.get("client_id", settings.GMAIL_CLIENT_ID),
            client_secret=token_data.get("client_secret", settings.GMAIL_CLIENT_SECRET),
            scopes=token_data.get("scopes"),
        )
        logger.info("Gmail credentials loaded from disk")
        return creds

    except Exception as e:
        logger.error(f"Failed to load Gmail token: {e}")
        return None
