"""
MailMind Gmail — BiLSTM Classifier Wrapper

Loads the existing trained BiLSTM model and provides a clean
predict() interface for the ingestion pipeline.

Reuses the exact model/tokenizer/label-encoder artifacts from
ml/categorization/ — does NOT train a new model.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from rag.config import settings
from rag.utils import get_logger

logger = get_logger("gmail.classifier")

# Module-level caches
_model = None
_sender_tokenizer = None
_subject_tokenizer = None
_body_tokenizer = None
_label_encoder = None
_loaded = False

# Model input dimensions (from the training notebook)
SENDER_MAX_LEN = 50
SUBJECT_MAX_LEN = 100
BODY_MAX_LEN = 400


def _get_artifacts_dir() -> Path:
    """Locate the classifier artifacts directory."""
    return settings.PROJECT_ROOT / "ml" / "categorization"


def load_classifier() -> None:
    """
    Load the BiLSTM model, tokenizers, and label encoder.

    Call this once at startup. Subsequent calls are no-ops.
    """
    global _model, _sender_tokenizer, _subject_tokenizer
    global _body_tokenizer, _label_encoder, _loaded

    if _loaded:
        return

    artifacts_dir = _get_artifacts_dir()
    model_path = artifacts_dir / "models" / "mailmind_email_bilstm_v1.keras"
    sender_tok_path = artifacts_dir / "tokenizers" / "sender_tokenizer.pkl"
    subject_tok_path = artifacts_dir / "tokenizers" / "subject_tokenizer.pkl"
    body_tok_path = artifacts_dir / "tokenizers" / "body_tokenizer.pkl"
    label_enc_path = artifacts_dir / "tokenizers" / "email_label_encoder.pkl"

    # Verify all artifacts exist
    for path in [model_path, sender_tok_path, subject_tok_path, body_tok_path, label_enc_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Classifier artifact not found: {path}\n"
                f"Ensure the trained model exists in ml/categorization/"
            )

    # Load TensorFlow/Keras model
    logger.info(f"Loading BiLSTM classifier from {model_path.name}...")
    from tensorflow.keras.models import load_model
    _model = load_model(str(model_path))

    # Load tokenizers
    with open(sender_tok_path, "rb") as f:
        _sender_tokenizer = pickle.load(f)
    with open(subject_tok_path, "rb") as f:
        _subject_tokenizer = pickle.load(f)
    with open(body_tok_path, "rb") as f:
        _body_tokenizer = pickle.load(f)

    # Load label encoder
    with open(label_enc_path, "rb") as f:
        _label_encoder = pickle.load(f)

    _loaded = True
    logger.info(
        f"Classifier loaded. Classes: {list(_label_encoder.classes_)}"
    )


def predict_email(
    sender: str,
    subject: str,
    body: str,
) -> dict:
    """
    Classify a single email using the BiLSTM model.

    Args:
        sender: Sender email/name string.
        subject: Email subject line.
        body: Email body text.

    Returns:
        Dict with keys:
            label: str — predicted class ('Spam', 'General', 'Important')
            confidence: float — prediction confidence
            probabilities: dict — class -> probability mapping
    """
    if not _loaded:
        load_classifier()

    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # Tokenize and pad sender
    sender_seq = _sender_tokenizer.texts_to_sequences([sender])
    sender_pad = pad_sequences(
        sender_seq, maxlen=SENDER_MAX_LEN,
        padding="post", truncating="post"
    )

    # Tokenize and pad subject
    subject_seq = _subject_tokenizer.texts_to_sequences([subject])
    subject_pad = pad_sequences(
        subject_seq, maxlen=SUBJECT_MAX_LEN,
        padding="post", truncating="post"
    )

    # Tokenize and pad body
    body_seq = _body_tokenizer.texts_to_sequences([body])
    body_pad = pad_sequences(
        body_seq, maxlen=BODY_MAX_LEN,
        padding="post", truncating="post"
    )

    # Predict
    probabilities = _model.predict(
        [sender_pad, subject_pad, body_pad],
        verbose=0,
    )[0]

    predicted_index = np.argmax(probabilities)
    predicted_label = _label_encoder.inverse_transform([predicted_index])[0]
    confidence = float(probabilities[predicted_index])

    return {
        "label": predicted_label,
        "confidence": confidence,
        "probabilities": {
            _label_encoder.classes_[i]: float(probabilities[i])
            for i in range(len(_label_encoder.classes_))
        },
    }


def is_spam(classification: dict) -> bool:
    """Check if a classification result indicates spam."""
    return classification.get("label", "").lower() == "spam"


def get_classes() -> Optional[list[str]]:
    """Get the classifier's class labels, or None if not loaded."""
    if _label_encoder is None:
        return None
    return list(_label_encoder.classes_)
