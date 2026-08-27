# Imports --------------------------------------------------

import json
import re
import pandas as pd


# Load Dataset ---------------------------------------------

with open("emails.json", "r", encoding="utf-8") as f:
    emails = json.load(f)


# Cleaning Functions ---------------------------------------

def remove_reply_chain(text):

    patterns = [

        r"\nOn .*?wrote:\s*$",
        r"\nFrom:.*$",
        r"\nSent:.*$",
        r"\nTo:.*$",
        r"\nSubject:.*$"

    ]

    earliest = len(text)

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
        )

        if match:
            earliest = min(earliest, match.start())

    return text[:earliest]


def remove_forwarded(text):

    patterns = [

        r"-{5,}\s*Forwarded message\s*-{5,}.*",

        r"Begin forwarded message:.*",

        r"Forwarded message:.*"

    ]

    for pattern in patterns:

        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

    return text


def remove_disclaimer(text):

    patterns = [

        r"--\s*\n\*\*Disclaimer:.*",

        r"\*\*Disclaimer:.*",

        r"This message was sent from Vellore Institute of Technology.*",

        r"The contents of this email may contain legally protected.*",

        r"You have received this email in error.*",

        r"©\s*\d{4}\s*Google LLC.*",

        r"You received this email to let you know about important changes.*"

    ]

    for pattern in patterns:

        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

    return text


def remove_images(text):

    return re.sub(
        r"\[image:.*?\]",
        "",
        text,
        flags=re.IGNORECASE
    )


def clean_markdown(text):

    # remove markdown bold/italic markers

    text = re.sub(r"\*+", "", text)

    return text


def clean_urls(text):

    # <https://abc.com>  --> https://abc.com

    text = re.sub(
        r"<(https?://[^>]+)>",
        r"\1",
        text
    )

    return text


def remove_unicode_artifacts(text):

    text = text.replace("\u200b", "")

    text = text.replace("\u00ad", "")

    text = text.replace("\ufeff", "")

    return text


def remove_signature(text):

    patterns = [

        r"\nWarm regards.*",

        r"\nThanks & Regards.*",

        r"\nRegards.*",

        r"\nBest Regards.*",

        r"\nBest,\s*$"

    ]

    earliest = len(text)

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        if match:
            earliest = min(earliest, match.start())

    return text[:earliest]


def normalize(text):

    text = text.replace("\r", "\n")

    text = re.sub(r"\n{3,}", "\n\n", text)

    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


# Cleaning --------------------------------------------------

for email in emails:

    body = email.get("Body", "")

    body = remove_unicode_artifacts(body)

    body = remove_images(body)

    body = remove_reply_chain(body)

    body = remove_forwarded(body)

    body = remove_signature(body)

    body = remove_disclaimer(body)

    body = clean_markdown(body)

    body = clean_urls(body)

    body = normalize(body)

    email["Body"] = body

    email["Attachment_Names"] = sorted(
        list(
            set(
                email.get("Attachment_Names", [])
            )
        )
    )


# Save JSON -------------------------------------------------

with open(
    "emails_cleaned.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        emails,
        f,
        indent=4,
        ensure_ascii=False
    )


# Save CSV --------------------------------------------------

df = pd.DataFrame(emails)

df.to_csv(

    "emails_cleaned.csv",

    index=False,

    encoding="utf-8-sig"

)


print("\nCleaning Complete!\n")

print("Saved:")
print("emails_cleaned.json")
print("emails_cleaned.csv")


# Save Parquet ---------------------------------

df.to_parquet(
    "emails_cleaned.parquet",
    engine="pyarrow",
    index=False
)

print("Parquet Saved Successfully!")