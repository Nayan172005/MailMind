"""
MailMind Evaluation — BiLSTM Email Classifier Evaluation.

Evaluates the BiLSTM classifier strictly on the held-out test split (542 samples)
reproduced from `ml/categorization/EmailClassifier.ipynb` using `SEED = 42`.
Never evaluates on or mixes training data.

Reports:
- 3x3 Confusion Matrix
- Per-class Precision, Recall, F1, Support for [Spam, General, Important]
- Macro-averaged F1, Precision, Recall
- Weighted-averaged F1, Precision, Recall
- Overall Accuracy
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from rag.config import settings
from rag.utils import get_logger

logger = get_logger("evaluation.classifier")

SEED = 42
DATASET_PATH = settings.PROJECT_ROOT / "ml" / "categorization" / "datasets" / "email_classification_dataset.parquet"
CSV_BACKUP_PATH = settings.PROJECT_ROOT / "ml" / "categorization" / "datasets" / "email_classification_dataset.csv"
ARTIFACTS_DIR = settings.PROJECT_ROOT / "ml" / "categorization"


@dataclass
class ClassifierEvaluationResult:
    total_test_samples: int
    class_names: List[str]
    confusion_matrix: List[List[int]]  # rows: actual, cols: predicted
    per_class_metrics: Dict[str, Dict[str, float]]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    accuracy: float

    def to_dict(self) -> dict:
        return asdict(self)


def load_held_out_test_split() -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """
    Reproduces the exact held-out test split (10% test = 542 rows)
    from `EmailClassifier.ipynb` with SEED=42 and stratification.
    """
    if DATASET_PATH.exists():
        df = pd.read_parquet(DATASET_PATH)
    elif CSV_BACKUP_PATH.exists():
        df = pd.read_csv(CSV_BACKUP_PATH)
    else:
        raise FileNotFoundError(f"Classification dataset not found at {DATASET_PATH} or {CSV_BACKUP_PATH}")

    # Standardize text columns
    for col in ["sender", "subject", "body"]:
        df[col] = df[col].fillna("").astype(str)
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    # Encode labels
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])

    X = df[["sender", "subject", "body"]]
    y = df["label_id"]

    # Step 1: 80% train, 20% temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    # Step 2: 10% val, 10% test from temp
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=SEED,
        stratify=y_temp,
    )

    logger.info(
        f"Dataset split: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}"
    )
    return X_test, y_test, le


def run_classifier_evaluation() -> ClassifierEvaluationResult:
    """
    Run evaluation on the held-out test dataset using the saved BiLSTM model.
    """
    print("\nEvaluating BiLSTM Classifier on held-out test set (542 emails)...")

    # 1. Load test data
    X_test, y_test, le = load_held_out_test_split()

    # 2. Load model and tokenizers
    model_path = ARTIFACTS_DIR / "models" / "mailmind_email_bilstm_v1.keras"
    sender_tok_path = ARTIFACTS_DIR / "tokenizers" / "sender_tokenizer.pkl"
    subject_tok_path = ARTIFACTS_DIR / "tokenizers" / "subject_tokenizer.pkl"
    body_tok_path = ARTIFACTS_DIR / "tokenizers" / "body_tokenizer.pkl"
    label_enc_path = ARTIFACTS_DIR / "tokenizers" / "email_label_encoder.pkl"

    for p in [model_path, sender_tok_path, subject_tok_path, body_tok_path, label_enc_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing classifier artifact: {p}")

    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model = load_model(str(model_path))
    with open(sender_tok_path, "rb") as f:
        sender_tok = pickle.load(f)
    with open(subject_tok_path, "rb") as f:
        subject_tok = pickle.load(f)
    with open(body_tok_path, "rb") as f:
        body_tok = pickle.load(f)
    with open(label_enc_path, "rb") as f:
        trained_le = pickle.load(f)

    # 3. Preprocess test inputs in batch for speed
    SENDER_MAX = 50
    SUBJECT_MAX = 100
    BODY_MAX = 400

    sender_seqs = sender_tok.texts_to_sequences(X_test["sender"].tolist())
    subject_seqs = subject_tok.texts_to_sequences(X_test["subject"].tolist())
    body_seqs = body_tok.texts_to_sequences(X_test["body"].tolist())

    sender_pad = pad_sequences(sender_seqs, maxlen=SENDER_MAX, padding="post", truncating="post")
    subject_pad = pad_sequences(subject_seqs, maxlen=SUBJECT_MAX, padding="post", truncating="post")
    body_pad = pad_sequences(body_seqs, maxlen=BODY_MAX, padding="post", truncating="post")

    # 4. Predict
    probabilities = model.predict([sender_pad, subject_pad, body_pad], batch_size=64, verbose=0)
    pred_indices = np.argmax(probabilities, axis=1)

    # Map labels to trained classes (e.g. general, important, spam)
    trained_classes = [c.lower() for c in trained_le.classes_]
    test_true_labels = [le.classes_[i].lower() for i in y_test.values]
    test_pred_labels = [trained_classes[i] for i in pred_indices]

    # Compute metrics
    unique_classes = sorted(list(set(trained_classes) | set(test_true_labels)))
    cm = confusion_matrix(test_true_labels, test_pred_labels, labels=unique_classes)
    acc = accuracy_score(test_true_labels, test_pred_labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        test_true_labels, test_pred_labels, labels=unique_classes, zero_division=0
    )

    per_class = {}
    for i, c in enumerate(unique_classes):
        per_class[c] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        test_true_labels, test_pred_labels, average="macro", zero_division=0
    )
    weight_p, weight_r, weight_f, _ = precision_recall_fscore_support(
        test_true_labels, test_pred_labels, average="weighted", zero_division=0
    )

    return ClassifierEvaluationResult(
        total_test_samples=len(y_test),
        class_names=unique_classes,
        confusion_matrix=cm.tolist(),
        per_class_metrics=per_class,
        macro_precision=round(float(macro_p), 4),
        macro_recall=round(float(macro_r), 4),
        macro_f1=round(float(macro_f), 4),
        weighted_precision=round(float(weight_p), 4),
        weighted_recall=round(float(weight_r), 4),
        weighted_f1=round(float(weight_f), 4),
        accuracy=round(float(acc), 4),
    )
