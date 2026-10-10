"""Pure, dependency-free helpers shared by the privacy and quality layers."""

from __future__ import annotations

import re
from decimal import Decimal

import pandas as pd

from safedata.domain.schema import DATE_FORMAT

AMOUNT_FORMAT_RE = re.compile(r"-?\d+(?:\.\d{1,2})?")
_NON_DIGIT_RE = re.compile(r"\D")


def digits_only(value: str | None) -> str:
    return _NON_DIGIT_RE.sub("", value or "")


def luhn_is_valid(pan: str) -> bool:
    """Mod-10 checksum used by all major card schemes."""
    if not pan.isdigit():
        return False
    total = 0
    for position, char in enumerate(reversed(pan)):
        digit = int(char)
        if position % 2 == 1:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        total += digit
    return total % 10 == 0


def amount_to_cents(value: str | None) -> int | None:
    """Exact decimal string -> integer cents (no float drift); None if malformed."""
    if not AMOUNT_FORMAT_RE.fullmatch(value or ""):
        return None
    return int(Decimal(value) * 100)


def format_cents(cents: int) -> str:
    return f"{Decimal(cents) / 100:.2f}"


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format=DATE_FORMAT, errors="coerce")
