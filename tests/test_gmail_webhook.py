"""
Tests for Gmail Pub/Sub Webhook Security and Processing.

Verifies:
1. OIDC Bearer JWT authentication:
   - Missing Authorization header (401)
   - Malformed Authorization header (401)
   - Invalid JWT signature / expired token (403)
   - Wrong audience (403)
   - Invalid issuer (403)
   - Unexpected service-account email (403)
   - Unverified service-account email (403)
   - Unconfigured server authentication settings (500)
2. Pub/Sub envelope validation:
   - Malformed JSON body (400)
   - Missing 'message' field (400)
   - Missing or empty 'message.data' (400)
   - Invalid Base64 in 'message.data' (400)
   - Non-JSON decoded payload (400)
   - Missing historyId in payload (400)
3. Notification processing and retries:
   - Successful authenticated notification (200)
   - Transient processing failure returns retryable HTTP 500
   - FAISS index not ready returns retryable HTTP 503
4. Watcher idempotency:
   - Duplicate or older historyId is safely acknowledged without duplicate ingestion
   - State is only advanced when processing succeeds
"""

import base64
import json
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes.gmail import verify_pubsub_jwt
from api.gmail.watcher import process_pubsub_notification, watch_state
from rag.config import settings


# ─── Test Fixtures & Constants ───────────────────────────────────────

TEST_AUDIENCE = "https://mailmind.example.com/gmail/webhook"
TEST_SERVICE_ACCOUNT = "pubsub-push-sa@mailmind-project.iam.gserviceaccount.com"
VALID_BEARER_TOKEN = "valid.jwt.token"


def make_pubsub_envelope(email: str = "user@gmail.com", history_id: str = "1005") -> dict:
    """Helper to create a standard Pub/Sub push notification envelope."""
    payload = json.dumps({"emailAddress": email, "historyId": history_id})
    b64_data = base64.b64encode(payload.encode("utf-8")).decode("utf-8")
    return {
        "message": {
            "data": b64_data,
            "messageId": "msg-pubsub-12345",
            "publishTime": "2026-10-14T12:00:00.000Z",
        },
        "subscription": "projects/mailmind/subscriptions/gmail-push-sub",
    }


@pytest.fixture
def configured_settings():
    """Ensure Pub/Sub auth settings are configured for test duration."""
    orig_audience = settings.GMAIL_PUBSUB_AUDIENCE
    orig_sa = settings.GMAIL_PUBSUB_SERVICE_ACCOUNT
    orig_verify = settings.GMAIL_PUBSUB_VERIFY_TOKEN

    settings.GMAIL_PUBSUB_AUDIENCE = TEST_AUDIENCE
    settings.GMAIL_PUBSUB_SERVICE_ACCOUNT = TEST_SERVICE_ACCOUNT
    settings.GMAIL_PUBSUB_VERIFY_TOKEN = True

    yield settings

    settings.GMAIL_PUBSUB_AUDIENCE = orig_audience
    settings.GMAIL_PUBSUB_SERVICE_ACCOUNT = orig_sa
    settings.GMAIL_PUBSUB_VERIFY_TOKEN = orig_verify


@pytest.fixture
def mock_valid_jwt():
    """Mock verify_oauth2_token returning valid claims matching test settings."""
    valid_claims = {
        "iss": "https://accounts.google.com",
        "aud": TEST_AUDIENCE,
        "email": TEST_SERVICE_ACCOUNT,
        "email_verified": True,
        "sub": "1234567890",
    }
    with patch("google.oauth2.id_token.verify_oauth2_token", return_value=valid_claims) as m:
        yield m


@pytest.fixture
def ready_pipeline():
    """Mock api.main.pipeline as ready with a mock VectorStore."""
    with patch("api.main.pipeline") as mock_pipeline:
        mock_pipeline.is_ready = True
        mock_pipeline.store = MagicMock()
        mock_pipeline.store._indexed_message_ids = set()
        yield mock_pipeline


@pytest.fixture
def client():
    return TestClient(app)


# ─── 1. Authentication Tests ─────────────────────────────────────────

class TestWebhookAuthentication:

    def test_missing_authorization_header(self, client, configured_settings):
        """Requests without Authorization header must be rejected with 401."""
        envelope = make_pubsub_envelope()
        response = client.post("/gmail/webhook", json=envelope)
        assert response.status_code == 401
        assert "Missing Authorization header" in response.json()["detail"]

    def test_malformed_authorization_header_no_bearer(self, client, configured_settings):
        """Header without 'Bearer' prefix must be rejected with 401."""
        envelope = make_pubsub_envelope()
        response = client.post(
            "/gmail/webhook",
            json=envelope,
            headers={"Authorization": f"Basic {VALID_BEARER_TOKEN}"},
        )
        assert response.status_code == 401
        assert "Invalid Authorization header format" in response.json()["detail"]

    def test_malformed_authorization_header_empty_token(self, client, configured_settings):
        """'Bearer ' without token must be rejected with 401."""
        envelope = make_pubsub_envelope()
        response = client.post(
            "/gmail/webhook",
            json=envelope,
            headers={"Authorization": "Bearer   "},
        )
        assert response.status_code == 401

    def test_server_unconfigured_audience(self, client):
        """If server has no GMAIL_PUBSUB_AUDIENCE set, return 500 rather than allowing access."""
        with patch.object(settings, "GMAIL_PUBSUB_AUDIENCE", ""), \
             patch.object(settings, "GMAIL_PUBSUB_SERVICE_ACCOUNT", TEST_SERVICE_ACCOUNT):
            envelope = make_pubsub_envelope()
            response = client.post(
                "/gmail/webhook",
                json=envelope,
                headers={"Authorization": f"Bearer {VALID_BEARER_TOKEN}"},
            )
            assert response.status_code == 500
            assert "audience verification is not configured" in response.json()["detail"]

    def test_server_unconfigured_service_account(self, client):
        """If server has no GMAIL_PUBSUB_SERVICE_ACCOUNT set, return 500."""
        with patch.object(settings, "GMAIL_PUBSUB_AUDIENCE", TEST_AUDIENCE), \
             patch.object(settings, "GMAIL_PUBSUB_SERVICE_ACCOUNT", ""):
            envelope = make_pubsub_envelope()
            response = client.post(
                "/gmail/webhook",
                json=envelope,
                headers={"Authorization": f"Bearer {VALID_BEARER_TOKEN}"},
            )
            assert response.status_code == 500
            assert "service-account verification is not configured" in response.json()["detail"]

    def test_invalid_jwt_signature(self, client, configured_settings):
        """Invalid JWT signature must be rejected with 403."""
        with patch(
            "google.oauth2.id_token.verify_oauth2_token",
            side_effect=ValueError("Invalid crypto padding / signature"),
        ):
            envelope = make_pubsub_envelope()
            response = client.post(
                "/gmail/webhook",
                json=envelope,
                headers={"Authorization": "Bearer bad-signature-token"},
            )
            assert response.status_code == 403
            assert "Token verification failed" in response.json()["detail"]

    def test_expired_jwt(self, client, configured_settings):
        """Expired JWT must be rejected with 403."""
        with patch(
            "google.oauth2.id_token.verify_oauth2_token",
            side_effect=ValueError("Token expired: 1600000000 < 1700000000"),
        ):
            envelope = make_pubsub_envelope()
            response = client.post(
                "/gmail/webhook",
                json=envelope,
                headers={"Authorization": "Bearer expired-token"},
            )
            assert response.status_code == 403
            assert "Token verification failed" in response.json()["detail"]

    def test_wrong_audience(self, client, configured_settings):
        """JWT with wrong audience must be rejected with 403."""
        with patch(
            "google.oauth2.id_token.verify_oauth2_token",
            side_effect=ValueError("Wrong audience; expected https://mailmind..."),
        ):
            envelope = make_pubsub_envelope()
            response = client.post(
                "/gmail/webhook",
                json=envelope,
                headers={"Authorization": f"Bearer {VALID_BEARER_TOKEN}"},
            )
            assert response.status_code == 403
            assert "Token verification failed" in response.json()["detail"]

    def test_invalid_token_issuer(self, client, configured_settings):
        """JWT from untrusted issuer must be rejected with 403."""
        untrusted_claims = {
            "iss": "https://untrusted-issuer.attacker.com",
            "aud": TEST_AUDIENCE,
            "email": TEST_SERVICE_ACCOUNT,
            "email_verified": True,
        }
        with patch("google.oauth2.id_token.verify_oauth2_token", return_value=untrusted_claims):
            envelope = make_pubsub_envelope()
            response = client.post(
                "/gmail/webhook",
                json=envelope,
                headers={"Authorization": f"Bearer {VALID_BEARER_TOKEN}"},
            )
            assert response.status_code == 403
            assert "Invalid token issuer" in response.json()["detail"]

    def test_unexpected_service_account_email(self, client, configured_settings):
        """Token from unexpected service account must be rejected with 403."""
        attacker_claims = {
            "iss": "https://accounts.google.com",
            "aud": TEST_AUDIENCE,
            "email": "attacker@evil-domain.iam.gserviceaccount.com",
            "email_verified": True,
        }
        with patch("google.oauth2.id_token.verify_oauth2_token", return_value=attacker_claims):
            envelope = make_pubsub_envelope()
            response = client.post(
                "/gmail/webhook",
                json=envelope,
                headers={"Authorization": f"Bearer {VALID_BEARER_TOKEN}"},
            )
            assert response.status_code == 403
            assert "Unexpected service-account email" in response.json()["detail"]

    def test_unverified_service_account_email(self, client, configured_settings):
        """Token with email_verified=False must be rejected with 403."""
        unverified_claims = {
            "iss": "https://accounts.google.com",
            "aud": TEST_AUDIENCE,
            "email": TEST_SERVICE_ACCOUNT,
            "email_verified": False,
        }
        with patch("google.oauth2.id_token.verify_oauth2_token", return_value=unverified_claims):
            envelope = make_pubsub_envelope()
            response = client.post(
                "/gmail/webhook",
                json=envelope,
                headers={"Authorization": f"Bearer {VALID_BEARER_TOKEN}"},
            )
            assert response.status_code == 403
            assert "Service-account email is not verified" in response.json()["detail"]


# ─── 2. Pub/Sub Envelope Validation Tests ────────────────────────────

class TestWebhookEnvelopeValidation:

    def test_malformed_json_body(self, client, configured_settings, mock_valid_jwt, ready_pipeline):
        """Non-JSON body must return 400 Bad Request."""
        response = client.post(
            "/gmail/webhook",
            content=b"this is not json",
            headers={
                "Authorization": f"Bearer {VALID_BEARER_TOKEN}",
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 400
        assert "Invalid JSON body" in response.json()["detail"]

    def test_missing_message_field(self, client, configured_settings, mock_valid_jwt, ready_pipeline):
        """Envelope missing 'message' object must return 400."""
        response = client.post(
            "/gmail/webhook",
            json={"subscription": "projects/p/subscriptions/s"},
            headers={"Authorization": f"Bearer {VALID_BEARER_TOKEN}"},
        )
        assert response.status_code == 400
        assert "missing 'message' object" in response.json()["detail"]

    def test_missing_message_data(self, client, configured_settings, mock_valid_jwt, ready_pipeline):
        """'message' without 'data' must return 400."""
        response = client.post(
            "/gmail/webhook",
            json={"message": {"messageId": "12345"}},
            headers={"Authorization": f"Bearer {VALID_BEARER_TOKEN}"},
        )
        assert response.status_code == 400
        assert "missing or empty 'message.data'" in response.json()["detail"]

    def test_empty_message_data(self, client, configured_settings, mock_valid_jwt, ready_pipeline):
        """'message.data' being empty string must return 400."""
        response = client.post(
            "/gmail/webhook",
            json={"message": {"data": "   "}},
            headers={"Authorization": f"Bearer {VALID_BEARER_TOKEN}"},
        )
        assert response.status_code == 400
        assert "missing or empty 'message.data'" in response.json()["detail"]

    def test_invalid_base64_in_data(self, client, configured_settings, mock_valid_jwt, ready_pipeline):
        """'message.data' containing invalid Base64 characters must return 400."""
        response = client.post(
            "/gmail/webhook",
            json={"message": {"data": "not!valid!base64==="}},
            headers={"Authorization": f"Bearer {VALID_BEARER_TOKEN}"},
        )
        assert response.status_code == 400
        assert "Malformed" in response.json()["detail"] or "Base64" in response.json()["detail"]

    def test_decoded_data_not_json(self, client, configured_settings, mock_valid_jwt, ready_pipeline):
        """'message.data' base64 decoded into non-JSON string must return 400."""
        b64_plain = base64.b64encode(b"plain non-json text").decode("utf-8")
        response = client.post(
            "/gmail/webhook",
            json={"message": {"data": b64_plain}},
            headers={"Authorization": f"Bearer {VALID_BEARER_TOKEN}"},
        )
        assert response.status_code == 400

    def test_decoded_data_missing_history_id(self, client, configured_settings, mock_valid_jwt, ready_pipeline):
        """Decoded JSON missing 'historyId' must return 400."""
        b64_no_history = base64.b64encode(json.dumps({"emailAddress": "user@gmail.com"}).encode()).decode()
        response = client.post(
            "/gmail/webhook",
            json={"message": {"data": b64_no_history}},
            headers={"Authorization": f"Bearer {VALID_BEARER_TOKEN}"},
        )
        assert response.status_code == 400
        assert "historyId" in response.json()["detail"]


# ─── 3. Processing and Retries Tests ─────────────────────────────────

class TestWebhookProcessingAndRetries:

    def test_successful_notification(self, client, configured_settings, mock_valid_jwt, ready_pipeline):
        """Valid authenticated notification returns 200 with processing summary."""
        mock_results = [
            {"status": "ingested", "message_id": "msg_001", "chunks_added": 2},
            {"status": "duplicate", "message_id": "msg_002", "chunks_added": 0},
        ]
        with patch("api.gmail.watcher.process_pubsub_notification", return_value=mock_results):
            envelope = make_pubsub_envelope(history_id="1005")
            response = client.post(
                "/gmail/webhook",
                json=envelope,
                headers={"Authorization": f"Bearer {VALID_BEARER_TOKEN}"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["processed"] == 2
            assert data["summary"] == {"ingested": 1, "duplicate": 1}

    def test_transient_failure_returns_500_for_retry(
        self, client, configured_settings, mock_valid_jwt, ready_pipeline
    ):
        """Transient processing failure must return HTTP 500 so Pub/Sub will retry."""
        with patch(
            "api.gmail.watcher.process_pubsub_notification",
            side_effect=RuntimeError("Gmail API network socket timeout"),
        ):
            envelope = make_pubsub_envelope(history_id="1005")
            response = client.post(
                "/gmail/webhook",
                json=envelope,
                headers={"Authorization": f"Bearer {VALID_BEARER_TOKEN}"},
            )
            assert response.status_code == 500
            assert "Transient processing error" in response.json()["detail"]

    def test_index_not_ready_returns_503_for_retry(
        self, client, configured_settings, mock_valid_jwt
    ):
        """When vector index is not loaded, return 503 so Pub/Sub retries after startup."""
        with patch("api.main.pipeline") as mock_pipeline:
            mock_pipeline.is_ready = False
            envelope = make_pubsub_envelope(history_id="1005")
            response = client.post(
                "/gmail/webhook",
                json=envelope,
                headers={"Authorization": f"Bearer {VALID_BEARER_TOKEN}"},
            )
            assert response.status_code == 503
            assert "Index not ready" in response.json()["detail"]


# ─── 4. Watcher Idempotency Tests ────────────────────────────────────

class TestWatcherIdempotency:

    def test_duplicate_history_id_ignored(self, tmp_path):
        """Duplicate notification with same or older historyId returns empty list without reprocessing."""
        mock_store = MagicMock()
        watch_state.history_id = "1005"

        with patch("api.gmail.watcher._process_history") as mock_process:
            # Send duplicate notification with historyId "1005"
            envelope = make_pubsub_envelope(history_id="1005")
            results = process_pubsub_notification(envelope, mock_store)

            assert results == []
            mock_process.assert_not_called()

            # Send older notification with historyId "1000"
            older_envelope = make_pubsub_envelope(history_id="1000")
            older_results = process_pubsub_notification(older_envelope, mock_store)

            assert older_results == []
            mock_process.assert_not_called()

    def test_new_history_id_processes_and_updates_state(self):
        """New higher historyId triggers _process_history and updates state on success."""
        mock_store = MagicMock()
        watch_state.history_id = "1005"

        mock_ingestion_results = [{"status": "ingested", "message_id": "m1"}]
        with patch("api.gmail.watcher._process_history", return_value=mock_ingestion_results) as mock_process, \
             patch.object(watch_state, "save") as mock_save:

            envelope = make_pubsub_envelope(history_id="1010")
            results = process_pubsub_notification(envelope, mock_store)

            assert results == mock_ingestion_results
            mock_process.assert_called_once_with("1005", mock_store)
            assert watch_state.history_id == "1010"
            mock_save.assert_called()

    def test_failure_preserves_stored_history_id(self):
        """When processing fails, stored history_id must NOT be advanced, enabling clean retry."""
        mock_store = MagicMock()
        watch_state.history_id = "1005"

        with patch("api.gmail.watcher._process_history", side_effect=IOError("Disk full or network down")), \
             patch.object(watch_state, "save") as mock_save:

            envelope = make_pubsub_envelope(history_id="1010")

            with pytest.raises(IOError):
                process_pubsub_notification(envelope, mock_store)

            # History ID remains at 1005 so Pub/Sub retry starts from 1005
            assert watch_state.history_id == "1005"
