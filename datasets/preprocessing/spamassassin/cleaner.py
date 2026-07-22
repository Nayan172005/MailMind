"""Cleaning and feature generation for SpamAssassin classifier records."""

from __future__ import annotations

from dataclasses import dataclass
import re
import pandas as pd

from .utils import count_urls, normalize_model_text


PUBLIC_COLUMNS = [
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

KNOWN_HEADER_NAMES = {
    "from", "to", "cc", "bcc", "subject", "date", "message-id",
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


def is_header_line(line: str) -> tuple[bool, str, str]:
    stripped = line.strip()
    if not stripped:
        return False, "", ""
    if stripped.startswith("From ") and not stripped.startswith("From:"):
        return True, "mbox_from", stripped[5:].strip()
    if ":" in stripped:
        parts = stripped.split(":", 1)
        key = parts[0].strip()
        val = parts[1].strip()
        key_lower = key.lower()
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
                in_header_block = False
                last_header_key = None
                continue
            is_hdr, h_key, h_val = is_header_line(line)
            if is_hdr:
                last_header_key = h_key
                if h_key == "mbox_from":
                    if not extracted["sender"] and "@" in h_val:
                        sender_match = re.search(r"[^\s<>]+@[^\s<>]+", h_val)
                        if sender_match:
                            extracted["sender"] = sender_match.group(0)
                elif h_key in ("from", "sender") and not extracted["sender"]:
                    extracted["sender"] = h_val
                elif h_key == "subject" and not extracted["subject"]:
                    extracted["subject"] = h_val
                elif h_key == "message-id" and not extracted["message_id"]:
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
            if (line.startswith(" ") or line.startswith("\t")) and last_header_key:
                continue
            in_header_block = False
            last_header_key = None

        is_hdr, h_key, h_val = is_header_line(line)
        if is_hdr:
            if h_key in ("from", "sender") and not extracted["sender"]:
                extracted["sender"] = h_val
            elif h_key == "subject" and not extracted["subject"]:
                extracted["subject"] = h_val
            elif h_key == "message-id" and not extracted["message_id"]:
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

    val = "\n".join(body_lines).strip()
    if not val:
        return extracted, ""

    norm_lines = []
    prev_blank = False
    for line in val.split("\n"):
        cleaned_l = re.sub(r"[ \t\f\v]+", " ", line).strip()
        if not cleaned_l:
            if norm_lines and not prev_blank:
                norm_lines.append("")
            prev_blank = True
            continue
        norm_lines.append(cleaned_l)
        prev_blank = False

    while norm_lines and norm_lines[-1] == "":
        norm_lines.pop()

    cleaned_body = "\n".join(norm_lines)
    return extracted, cleaned_body


@dataclass(slots=True)
class CleaningReport:
    """Per-chunk cleaning counters."""

    input_count: int = 0
    failed_parses_removed: int = 0
    empty_bodies_removed: int = 0
    output_count: int = 0


class SpamAssassinCleaner:
    """Produce model-ready text matching Enron canonical schema."""

    def clean_chunk(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
        """Clean chunk and derive canonical features."""

        report = CleaningReport(input_count=len(frame))
        if frame.empty:
            return pd.DataFrame(columns=PUBLIC_COLUMNS), report

        cleaned = frame.copy()
        failed_mask = ~cleaned["_parse_success"].fillna(False)
        report.failed_parses_removed = int(failed_mask.sum())
        cleaned = cleaned.loc[~failed_mask].copy()

        if cleaned.empty:
            return pd.DataFrame(columns=PUBLIC_COLUMNS), report

        files = []
        message_ids = []
        dates = []
        senders = []
        receivers = []
        ccs = []
        bccs = []
        subjects = []
        bodies = []

        for _, row in cleaned.iterrows():
            raw_sender = str(row.get("sender", "") or "").strip()
            raw_subject = str(row.get("subject", "") or "").strip()
            raw_body = str(row.get("body", "") or "")

            extracted, cleaned_b = extract_headers_and_clean_body(
                raw_body,
                existing_sender=raw_sender,
                existing_subject=raw_subject,
            )

            files.append("")
            senders.append((extracted.get("sender") or raw_sender).strip())
            subjects.append((extracted.get("subject") or raw_subject).strip())
            message_ids.append((extracted.get("message_id") or row.get("message_id", "") or "").strip())
            dates.append((extracted.get("date") or row.get("date", "") or "").strip())
            receivers.append((extracted.get("receiver") or row.get("receiver", "") or "").strip())
            ccs.append((extracted.get("cc") or row.get("cc", "") or "").strip())
            bccs.append((extracted.get("bcc") or row.get("bcc", "") or "").strip())
            bodies.append(cleaned_b)

        cleaned["file"] = files
        cleaned["message_id"] = message_ids
        cleaned["date"] = dates
        cleaned["sender"] = senders
        cleaned["receiver"] = receivers
        cleaned["cc"] = ccs
        cleaned["bcc"] = bccs
        cleaned["subject"] = subjects
        cleaned["body"] = bodies

        empty_mask = cleaned["body"].eq("")
        report.empty_bodies_removed = int(empty_mask.sum())
        cleaned = cleaned.loc[~empty_mask].copy()
        if cleaned.empty:
            return pd.DataFrame(columns=PUBLIC_COLUMNS), report

        cleaned["body_length"] = cleaned["body"].str.len().astype("int64")
        cleaned["word_count"] = cleaned["body"].map(lambda b: len(b.split())).astype("int64")
        cleaned["has_subject"] = cleaned["subject"].ne("")

        subj_lower = cleaned["subject"].str.lower().str.strip()
        cleaned["is_reply"] = subj_lower.str.startswith("re:", na=False)
        cleaned["is_forward"] = subj_lower.str.startswith(("fw:", "fwd:"), na=False)
        cleaned["label"] = cleaned["label"].astype("int8")

        report.output_count = len(cleaned)
        return cleaned.loc[:, PUBLIC_COLUMNS], report
