"""Masking of structured PII fields and redaction of PII embedded in free text."""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

import pandas as pd

from safedata.domain.models import Redaction
from safedata.domain.primitives import digits_only
from safedata.domain.schema import (
    LINEAGE_COLUMN,
    OUTPUT_COLUMNS,
    PROTECTED_ANALYTIC_COLUMNS,
    STRUCTURED_PII_COLUMNS,
    PiiEntity,
)
from safedata.privacy.detection import NameLexicon, find_name_spans
from safedata.privacy.patterns import (
    CARD_RE,
    EMAIL_FULL_RE,
    EMAIL_RE,
    PHONE_RE,
    REDACTED_CARD,
    REDACTED_EMAIL,
    REDACTED_NAME,
    REDACTED_PHONE,
)
from safedata.privacy.tokenization import Tokenizer

logger = logging.getLogger(__name__)


def mask_email(value: str | None) -> str:
    """``jane.doe@gmail.com`` -> ``j***@gmail.com``; malformed values are fully redacted."""
    value = (value or "").strip()
    if not value:
        return ""
    if not EMAIL_FULL_RE.fullmatch(value):
        return REDACTED_EMAIL
    local, domain = value.split("@", 1)
    return f"{local[0]}***@{domain}"


def mask_phone(value: str | None) -> str:
    """Keep only the last four digits (support-verification pattern)."""
    digits = digits_only(value)
    if not digits:
        return ""
    return f"***-***-{digits[-4:]}" if len(digits) >= 10 else REDACTED_PHONE


def mask_card(value: str | None) -> str:
    """PCI DSS-style truncation: retain at most the last four digits."""
    digits = digits_only(value)
    if not digits:
        return ""
    return f"****-****-****-{digits[-4:]}" if 13 <= len(digits) <= 19 else REDACTED_CARD


def redact_free_text(text: str, lexicon: NameLexicon, own_name: str = "") -> Redaction:
    """
    Remove PII embedded in unstructured text. Order matters: emails first (their
    digits must not read as phones), then cards (longest digit runs), phones, names.
    Card/phone matching is fail-closed: any 13-19 digit run is redacted, Luhn-valid or not.
    """
    if not text:
        return Redaction(text or "", frozenset())
    found: set[PiiEntity] = set()

    def _mask_email_match(match: re.Match[str]) -> str:
        found.add(PiiEntity.EMAIL)
        return mask_email(match.group(0))

    text = EMAIL_RE.sub(_mask_email_match, text)
    text, n_cards = CARD_RE.subn(REDACTED_CARD, text)
    text, n_phones = PHONE_RE.subn(REDACTED_PHONE, text)
    if n_cards:
        found.add(PiiEntity.CARD)
    if n_phones:
        found.add(PiiEntity.PHONE)

    if spans := find_name_spans(text, lexicon, own_name):
        found.add(PiiEntity.NAME)
        for start, end in reversed(spans):
            text = f"{text[:start]}{REDACTED_NAME}{text[end:]}"
    return Redaction(text, frozenset(found))


def mask_dataset(df: pd.DataFrame, tokenizer: Tokenizer, lexicon: NameLexicon) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the masked frame (``OUTPUT_COLUMNS``) and masking statistics."""
    out = pd.DataFrame(index=df.index)
    out[LINEAGE_COLUMN] = df[LINEAGE_COLUMN]
    out["Transaction_ID"] = df["Transaction_ID"]
    out["Customer_Token"] = df["Customer_Name"].map(lambda v: tokenizer.token(v, "CUST"))
    out["Email_Address"] = df["Email_Address"].map(mask_email)
    out["Phone_Number"] = df["Phone_Number"].map(mask_phone)
    out["Credit_Card_Number"] = df["Credit_Card_Number"].map(mask_card)
    out["Card_Token"] = df["Credit_Card_Number"].map(lambda v: tokenizer.token(digits_only(v), "CARD"))

    # Analytic fields are copied, never transformed: integrity by construction.
    for col in PROTECTED_ANALYTIC_COLUMNS:
        out[col] = df[col]
    out["Merchant_Category"] = df["Merchant_Category"]

    redactions = [
        redact_free_text(note, lexicon, own_name)
        for note, own_name in zip(df["Transaction_Notes"], df["Customer_Name"], strict=True)
    ]
    out["Transaction_Notes"] = [r.text for r in redactions]
    out["Notes_PII_Entities"] = [";".join(sorted(r.entities)) for r in redactions]

    entity_rows = Counter(str(entity) for r in redactions for entity in r.entities)
    stats: dict[str, Any] = {
        "structured_columns_masked": list(STRUCTURED_PII_COLUMNS),
        "notes_with_pii_redacted": sum(1 for r in redactions if r.entities),
        "notes_entity_row_counts": dict(sorted(entity_rows.items())),
        "distinct_customer_tokens": int(out["Customer_Token"].replace("", pd.NA).nunique()),
        "distinct_card_tokens": int(out["Card_Token"].replace("", pd.NA).nunique()),
        "hmac_key_fingerprint": tokenizer.fingerprint,
    }
    logger.info(
        "Masked %d structured PII columns; redacted PII in %s notes %s",
        len(STRUCTURED_PII_COLUMNS),
        f"{stats['notes_with_pii_redacted']:,}",
        stats["notes_entity_row_counts"],
    )
    return out[list(OUTPUT_COLUMNS)], stats
