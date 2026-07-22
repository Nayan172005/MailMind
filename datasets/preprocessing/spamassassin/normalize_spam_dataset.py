"""Schema normalization and header cleanup for SpamAssassin processed dataset.

Transforms the SpamAssassin processed dataset to match the canonical Enron schema:
[file, message_id, date, sender, receiver, cc, bcc, subject, body, body_length, word_count, has_subject, is_reply, is_forward, label]
"""

from __future__ import annotations

import random
import re
from pathlib import Path
import pandas as pd

TARGET_COLUMNS = [
    "file",
    "message_id",
    "date",
    "sender",
    "receiver",
    "cc",
    "bcc",
    "subject",
    "body",
    "body_length",
    "word_count",
    "has_subject",
    "is_reply",
    "is_forward",
    "label",
]

# Set of standard/known header names for matching
KNOWN_HEADER_NAMES = {
    "from", "to", "cc", "bcc", "subject", "date", "message-id", "message-id",
    "reply-to", "reply-by", "sender", "errors-to", "return-path", "delivered-to",
    "received", "mime-version", "content-type", "content-transfer-encoding",
    "content-disposition", "content-class", "precedence", "status", "in-reply-to",
    "references", "user-agent", "organization", "comments", "thread-index",
    "lines", "envelope-to", "x-originalarrivaltime", "x-mimeole", "x-mimesent",
    "resent-from", "resent-to", "resent-date", "resent-message-id", "resent-cc", "resent-bcc",
    "x-mailer", "x-priority", "x-msmail-priority", "x-mime-autoconverted", "x-beenthere",
    "x-mailman-version", "x-original-date", "x-loop", "x-remote-host", "x_id", "x-keywords",
    "x-operating-system", "x-message-flag", "x-mailscanner", "x-library", "x-header",
    "list-id", "list-unsubscribe", "list-subscribe", "list-help", "list-post", "list-archive"
}


def normalize_body_text(text: object) -> str:
    """Normalize body whitespace while preserving paragraph breaks, URLs, punctuation."""

    val = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not val:
        return ""

    lines = []
    prev_blank = False
    for line in val.split("\n"):
        cleaned = re.sub(r"[ \t\f\v]+", " ", line).strip()
        if not cleaned:
            if lines and not prev_blank:
                lines.append("")
            prev_blank = True
            continue
        lines.append(cleaned)
        prev_blank = False

    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)


def is_header_line(line: str) -> tuple[bool, str, str]:
    """Check if a line is an email header line or mbox envelope line.

    Returns:
        (is_header, header_name_lower, header_value)
    """

    stripped = line.strip()
    if not stripped:
        return False, "", ""

    # Envelope From line e.g. "From foo@bar.com Sat Sep 7 22:05:58 2002"
    if stripped.startswith("From ") and not stripped.startswith("From:"):
        return True, "mbox_from", stripped[5:].strip()

    # Standard "Header-Name: Value"
    if ":" in stripped:
        parts = stripped.split(":", 1)
        key = parts[0].strip()
        val = parts[1].strip()
        key_lower = key.lower()

        # Check if key is a valid identifier (no spaces, control chars)
        if re.match(r"^[a-zA-Z0-9\-_]+$", key) and len(key) <= 50:
            if (
                key_lower in KNOWN_HEADER_NAMES
                or key_lower.startswith(("x-", "list-", "resent-", "content-", "mail-", "envelope-"))
            ):
                return True, key_lower, val

    return False, "", ""


def extract_headers_and_clean_body(
    raw_body: str,
    existing_sender: str = "",
    existing_subject: str = "",
) -> tuple[dict[str, str], str]:
    """Extract metadata headers from body text and return cleaned body without headers."""

    if not raw_body:
        return {}, ""

    normalized = raw_body.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")

    extracted = {
        "sender": existing_sender,
        "subject": existing_subject,
        "message_id": "",
        "date": "",
        "receiver": "",
        "cc": "",
        "bcc": "",
    }

    body_lines = []
    in_header_block = True
    last_header_key = None

    for line in lines:
        stripped = line.strip()

        if in_header_block:
            if not stripped:
                # Blank line encountered in header block -> transition to body
                in_header_block = False
                last_header_key = None
                continue

            # Check for header line
            is_hdr, h_key, h_val = is_header_line(line)
            if is_hdr:
                last_header_key = h_key
                if h_key == "mbox_from":
                    # Mbox line e.g. "foo@bar.com Sat Sep 7 22:05:58 2002"
                    if not extracted["sender"] and "@" in h_val:
                        sender_match = re.search(r"[^\s<>]+@[^\s<>]+", h_val)
                        if sender_match:
                            extracted["sender"] = sender_match.group(0)
                elif h_key in ("from", "sender") and not extracted["sender"]:
                    extracted["sender"] = h_val
                elif h_key == "subject" and not extracted["subject"]:
                    extracted["subject"] = h_val
                elif h_key in ("message-id", "message-id") and not extracted["message_id"]:
                    extracted["message_id"] = h_val
                elif h_key == "date" and not extracted["date"]:
                    extracted["date"] = h_val
                elif h_key == "to" and not extracted["receiver"]:
                    extracted["receiver"] = h_val
                elif h_key == "cc" and not extracted["cc"]:
                    extracted["cc"] = h_val
                elif h_key == "bcc" and not extracted["bcc"]:
                    extracted["bcc"] = h_val
                continue

            # Check for folded header continuation line
            if (line.startswith(" ") or line.startswith("\t")) and last_header_key:
                # Continuation of previous header
                continue

            # First line that is not a header line and not blank -> header block ends!
            in_header_block = False
            last_header_key = None

        # Outside initial header block: check if line is an isolated lingering header
        is_hdr, h_key, h_val = is_header_line(line)
        if is_hdr:
            if h_key in ("from", "sender") and not extracted["sender"]:
                extracted["sender"] = h_val
            elif h_key == "subject" and not extracted["subject"]:
                extracted["subject"] = h_val
            elif h_key in ("message-id", "message-id") and not extracted["message_id"]:
                extracted["message_id"] = h_val
            elif h_key == "date" and not extracted["date"]:
                extracted["date"] = h_val
            elif h_key == "to" and not extracted["receiver"]:
                extracted["receiver"] = h_val
            elif h_key == "cc" and not extracted["cc"]:
                extracted["cc"] = h_val
            elif h_key == "bcc" and not extracted["bcc"]:
                extracted["bcc"] = h_val
            continue

        body_lines.append(line)

    cleaned_body = normalize_body_text("\n".join(body_lines))
    return extracted, cleaned_body


def transform_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Transform SpamAssassin dataset to canonical Enron schema."""

    transformed_rows = []
    stats = {
        "emails_processed": len(df),
        "headers_extracted": 0,
        "bodies_cleaned": 0,
        "missing_subject": 0,
        "missing_sender": 0,
        "missing_receiver": 0,
    }

    for _, row in df.iterrows():
        raw_sender = str(row.get("sender", "") or "").strip()
        raw_subject = str(row.get("subject", "") or "").strip()
        raw_body = str(row.get("body", "") or "")
        label = row["label"]

        extracted, cleaned_body = extract_headers_and_clean_body(
            raw_body,
            existing_sender=raw_sender,
            existing_subject=raw_subject,
        )

        sender = (extracted.get("sender") or raw_sender).strip()
        subject = (extracted.get("subject") or raw_subject).strip()
        message_id = (extracted.get("message_id") or "").strip()
        date = (extracted.get("date") or "").strip()
        receiver = (extracted.get("receiver") or "").strip()
        cc = (extracted.get("cc") or "").strip()
        bcc = (extracted.get("bcc") or "").strip()

        if message_id or date or receiver or cc or bcc:
            stats["headers_extracted"] += 1
        if cleaned_body != raw_body:
            stats["bodies_cleaned"] += 1
        if not subject:
            stats["missing_subject"] += 1
        if not sender:
            stats["missing_sender"] += 1
        if not receiver:
            stats["missing_receiver"] += 1

        body_length = len(cleaned_body)
        word_count = len(cleaned_body.split())
        has_subject = bool(subject != "")

        subj_lower = subject.lower().lstrip()
        is_reply = subj_lower.startswith("re:")
        is_forward = subj_lower.startswith(("fw:", "fwd:"))

        transformed_rows.append({
            "file": "",
            "message_id": message_id,
            "date": date,
            "sender": sender,
            "receiver": receiver,
            "cc": cc,
            "bcc": bcc,
            "subject": subject,
            "body": cleaned_body,
            "body_length": body_length,
            "word_count": word_count,
            "has_subject": has_subject,
            "is_reply": is_reply,
            "is_forward": is_forward,
            "label": int(label),
        })

    out_df = pd.DataFrame(transformed_rows, columns=TARGET_COLUMNS)
    return out_df, stats


def main():
    project_root = Path(__file__).resolve().parents[3]
    spam_dir = project_root / "datasets" / "processed" / "spamassassin"
    processed_dir = project_root / "datasets" / "processed"

    csv_paths = []
    if (spam_dir / "spam_processed.csv").exists():
        csv_paths.append(spam_dir / "spam_processed.csv")
    if (processed_dir / "spam_processed.csv").exists():
        csv_paths.append(processed_dir / "spam_processed.csv")

    if not csv_paths:
        raise FileNotFoundError("Could not locate spam_processed.csv in datasets/processed or datasets/processed/spamassassin")

    primary_csv = csv_paths[0]
    print(f"Reading {primary_csv}...")
    df = pd.read_csv(primary_csv, keep_default_na=False)
    print(f"Initial rows: {len(df)}")

    transformed_df, stats = transform_dataset(df)
    print(f"Transformed rows: {len(transformed_df)}")

    # Overwrite CSV and Parquet files
    for p_csv in csv_paths:
        print(f"Writing transformed CSV to {p_csv}...")
        transformed_df.to_csv(p_csv, index=False, encoding="utf-8")
        p_parquet = p_csv.with_suffix(".parquet")
        print(f"Writing Parquet to {p_parquet}...")
        transformed_df.to_parquet(p_parquet, index=False, engine="pyarrow")

    # Also create/update parquet if missing in primary location
    primary_parquet = primary_csv.with_suffix(".parquet")
    transformed_df.to_parquet(primary_parquet, index=False, engine="pyarrow")

    # Randomly inspect 100 emails for validation
    random.seed(42)
    sample_indices = random.sample(range(len(transformed_df)), min(100, len(transformed_df)))
    inspect_samples = transformed_df.iloc[sample_indices]

    no_headers_in_body = True
    for idx, row in inspect_samples.iterrows():
        b = row["body"]
        for line in b.split("\n")[:5]:
            is_hdr, _, _ = is_header_line(line)
            if is_hdr:
                no_headers_in_body = False
                print(f"[WARNING] Lingering header in row {idx}: {line}")

    avg_body_len = transformed_df["body_length"].mean()
    avg_word_cnt = transformed_df["word_count"].mean()

    print("\n==========================================")
    print("SpamAssassin Schema Normalization Report")
    print("==========================================")
    print(f"Emails processed             : {stats['emails_processed']:,}")
    print(f"Headers extracted            : {stats['headers_extracted']:,}")
    print(f"Bodies cleaned               : {stats['bodies_cleaned']:,}")
    print(f"Emails with missing subject  : {stats['missing_subject']:,}")
    print(f"Emails with missing sender   : {stats['missing_sender']:,}")
    print(f"Emails with missing receiver : {stats['missing_receiver']:,}")
    print(f"Average body length          : {avg_body_len:.1f} chars")
    print(f"Average word count           : {avg_word_cnt:.1f} words")
    print(f"Validation sample (100)      : {'PASSED (No headers in body)' if no_headers_in_body else 'WARNING'}")
    print("==========================================")


if __name__ == "__main__":
    main()
