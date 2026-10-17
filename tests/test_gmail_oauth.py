"""
Tests for Gmail OAuth PKCE flow.

Verifies that the authorization URL generation and callback token exchange
properly persist and restore the PKCE code_verifier and state parameter.
"""

import json
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest


# ─── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def mock_settings(tmp_path):
    """Patch settings to use a temp directory for token/state files."""
    token_path = tmp_path / "gmail" / "token.json"
    state_path = tmp_path / "gmail" / "state.json"

    with patch("api.gmail.auth.settings") as mock_s:
        mock_s.GMAIL_CLIENT_ID = "test-client-id"
        mock_s.GMAIL_CLIENT_SECRET = "test-client-secret"
        mock_s.GMAIL_REDIRECT_URI = "http://localhost:8000/gmail/oauth/callback"
        mock_s.GMAIL_SCOPES = "https://www.googleapis.com/auth/gmail.readonly"
        mock_s.gmail_token_abs_path = token_path

        yield mock_s


@pytest.fixture(autouse=True)
def reset_credentials():
    """Reset the module-level credentials cache between tests."""
    import api.gmail.auth as auth_mod
    auth_mod._credentials = None
    yield
    auth_mod._credentials = None


# ─── Tests: PKCE State Persistence ───────────────────────────────────

class TestPKCEStatePersistence:
    def test_get_auth_url_persists_state(self, mock_settings, tmp_path):
        """get_auth_url() should persist code_verifier and state to disk."""
        from api.gmail.auth import get_auth_url, _oauth_state_path

        with patch("api.gmail.auth.Flow") as MockFlow:
            mock_flow_instance = MagicMock()
            mock_flow_instance.authorization_url.return_value = (
                "https://accounts.google.com/o/oauth2/auth?state=test_state_123",
                "test_state_123",
            )
            mock_flow_instance.code_verifier = "test_code_verifier_abc"
            MockFlow.from_client_config.return_value = mock_flow_instance

            auth_url = get_auth_url()

        # Verify the URL was returned
        assert "test_state_123" in auth_url

        # Verify state file was written
        state_path = _oauth_state_path()
        assert state_path.exists()

        with open(state_path, "r") as f:
            saved = json.load(f)

        assert saved["state"] == "test_state_123"
        assert saved["code_verifier"] == "test_code_verifier_abc"

    def test_handle_callback_restores_verifier(self, mock_settings, tmp_path):
        """handle_callback() should restore the code_verifier before fetch_token()."""
        from api.gmail.auth import _save_oauth_state

        # Pre-persist the state as if get_auth_url() was called
        _save_oauth_state("correct_state", "correct_verifier")

        with patch("api.gmail.auth.Flow") as MockFlow:
            mock_flow_instance = MagicMock()
            mock_creds = MagicMock()
            mock_creds.token = "access_token"
            mock_creds.refresh_token = "refresh_token"
            mock_creds.token_uri = "https://oauth2.googleapis.com/token"
            mock_creds.client_id = "test-client-id"
            mock_creds.client_secret = "test-client-secret"
            mock_creds.scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
            mock_flow_instance.credentials = mock_creds
            MockFlow.from_client_config.return_value = mock_flow_instance

            from api.gmail.auth import handle_callback
            handle_callback("auth_code_xyz", "correct_state")

        # Verify code_verifier was restored on the Flow
        assert mock_flow_instance.code_verifier == "correct_verifier"

        # Verify fetch_token was called with the auth code
        mock_flow_instance.fetch_token.assert_called_once_with(code="auth_code_xyz")

        # Verify the Flow was created with the correct state
        call_kwargs = MockFlow.from_client_config.call_args
        assert call_kwargs[1]["state"] == "correct_state"

    def test_state_file_deleted_after_callback(self, mock_settings, tmp_path):
        """The state file should be deleted after successful callback (no replay)."""
        from api.gmail.auth import _save_oauth_state, _oauth_state_path

        _save_oauth_state("state_abc", "verifier_abc")
        assert _oauth_state_path().exists()

        with patch("api.gmail.auth.Flow") as MockFlow:
            mock_flow_instance = MagicMock()
            mock_creds = MagicMock()
            mock_creds.token = "tok"
            mock_creds.refresh_token = "ref"
            mock_creds.token_uri = "uri"
            mock_creds.client_id = "cid"
            mock_creds.client_secret = "sec"
            mock_creds.scopes = []
            mock_flow_instance.credentials = mock_creds
            MockFlow.from_client_config.return_value = mock_flow_instance

            from api.gmail.auth import handle_callback
            handle_callback("code", "state_abc")

        # State file should be gone
        assert not _oauth_state_path().exists()


# ─── Tests: State Validation (CSRF) ─────────────────────────────────

class TestStateValidation:
    def test_mismatched_state_raises(self, mock_settings, tmp_path):
        """Mismatched state parameter should raise RuntimeError."""
        from api.gmail.auth import _save_oauth_state, handle_callback

        _save_oauth_state("original_state", "verifier")

        with pytest.raises(RuntimeError, match="state mismatch"):
            handle_callback("code", "wrong_state")

    def test_missing_state_file_raises(self, mock_settings, tmp_path):
        """Missing state file (expired session) should raise RuntimeError."""
        from api.gmail.auth import handle_callback

        # Don't save any state — simulates expired/missing session
        with pytest.raises(RuntimeError, match="expired or not found"):
            handle_callback("code", "some_state")

    def test_double_callback_raises(self, mock_settings, tmp_path):
        """Using the same callback twice should fail (state file consumed)."""
        from api.gmail.auth import _save_oauth_state, handle_callback

        _save_oauth_state("state_once", "verifier_once")

        with patch("api.gmail.auth.Flow") as MockFlow:
            mock_flow_instance = MagicMock()
            mock_creds = MagicMock()
            mock_creds.token = "t"
            mock_creds.refresh_token = "r"
            mock_creds.token_uri = "u"
            mock_creds.client_id = "c"
            mock_creds.client_secret = "s"
            mock_creds.scopes = []
            mock_flow_instance.credentials = mock_creds
            MockFlow.from_client_config.return_value = mock_flow_instance

            # First call succeeds
            handle_callback("code", "state_once")

        # Second call with same state should fail
        with pytest.raises(RuntimeError, match="expired or not found"):
            handle_callback("code", "state_once")


# ─── Tests: Full Auth → Callback Round-Trip ──────────────────────────

class TestFullRoundTrip:
    def test_auth_then_callback_preserves_verifier(self, mock_settings, tmp_path):
        """
        End-to-end test: get_auth_url() → callback with correct state
        should result in fetch_token() being called with the original
        code_verifier restored on the Flow.
        """
        from api.gmail.auth import get_auth_url, handle_callback

        captured_verifier = None
        captured_state = None

        # Phase 1: Generate auth URL
        with patch("api.gmail.auth.Flow") as MockFlow:
            mock_flow_auth = MagicMock()
            mock_flow_auth.authorization_url.return_value = (
                "https://example.com/auth?state=roundtrip_state",
                "roundtrip_state",
            )
            mock_flow_auth.code_verifier = "roundtrip_verifier"
            MockFlow.from_client_config.return_value = mock_flow_auth

            url = get_auth_url()

        assert "roundtrip_state" in url

        # Phase 2: Handle callback
        with patch("api.gmail.auth.Flow") as MockFlow:
            mock_flow_cb = MagicMock()
            mock_creds = MagicMock()
            mock_creds.token = "access"
            mock_creds.refresh_token = "refresh"
            mock_creds.token_uri = "https://oauth2.googleapis.com/token"
            mock_creds.client_id = "test-client-id"
            mock_creds.client_secret = "test-client-secret"
            mock_creds.scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
            mock_flow_cb.credentials = mock_creds
            MockFlow.from_client_config.return_value = mock_flow_cb

            handle_callback("authorization_code_here", "roundtrip_state")

            # Verify the code_verifier from Phase 1 was restored
            assert mock_flow_cb.code_verifier == "roundtrip_verifier"

            # Verify Flow was constructed with the correct state
            call_kwargs = MockFlow.from_client_config.call_args
            assert call_kwargs[1]["state"] == "roundtrip_state"

            # Verify fetch_token was called
            mock_flow_cb.fetch_token.assert_called_once()
