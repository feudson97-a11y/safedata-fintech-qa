"""Compiled detection patterns. Kept in one module so detection and leak re-scans can never diverge."""

from __future__ import annotations

import re
from typing import Final

from safedata.domain.schema import PiiEntity

EMAIL_RE: Final = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
EMAIL_FULL_RE: Final = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
# 13-19 digits (PAN length range), optionally separated by single spaces or dashes.
CARD_RE: Final = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
# NANP phone numbers with optional +1 and common separators.
PHONE_RE: Final = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
# Capitalised word; allows O'Brien / Smith-Jones style surnames.
CAP_WORD_RE: Final = re.compile(r"\b[A-Z][a-z]*(?:['\u2019-][A-Z][a-z]+)*\b")
HONORIFIC_RE: Final = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Mx|Dr)\.?\s+[A-Z][a-z]*(?:['\u2019-][A-Z][a-z]+)*\b")
# Digit runs with light punctuation, used by the exact-value leak scan.
DIGIT_RUN_RE: Final = re.compile(r"\d(?:[\s().+-]{0,2}\d)*")

REGEX_DETECTORS: Final[dict[PiiEntity, re.Pattern[str]]] = {
    PiiEntity.EMAIL: EMAIL_RE,
    PiiEntity.CARD: CARD_RE,
    PiiEntity.PHONE: PHONE_RE,
}

REDACTED_NAME: Final = "[REDACTED_NAME]"
REDACTED_PHONE: Final = "[REDACTED_PHONE]"
REDACTED_CARD: Final = "[REDACTED_CARD]"
REDACTED_EMAIL: Final = "[REDACTED_EMAIL]"
