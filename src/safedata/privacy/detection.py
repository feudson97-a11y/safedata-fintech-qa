"""PII detection: name lexicon, free-text span finding and column classification."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Self

import pandas as pd

from safedata.domain.schema import FREE_TEXT_COLUMNS, LINEAGE_COLUMN, STRUCTURED_PII_COLUMNS, PiiEntity
from safedata.privacy.patterns import CAP_WORD_RE, HONORIFIC_RE, REGEX_DETECTORS

logger = logging.getLogger(__name__)

Span = tuple[int, int]


@dataclass(frozen=True, slots=True)
class NameLexicon:
    """First/last-name vocabulary learned from the structured name column."""

    first_names: frozenset[str]
    last_names: frozenset[str]

    @classmethod
    def from_names(cls, names: Iterable[str]) -> Self:
        firsts: set[str] = set()
        lasts: set[str] = set()
        for name in names:
            parts = str(name).split()
            if len(parts) >= 2:
                firsts.add(parts[0])
                lasts.add(parts[-1])
        return cls(frozenset(firsts), frozenset(lasts))


def _merge_spans(spans: Iterable[Span]) -> list[Span]:
    merged: list[Span] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def find_name_spans(text: str, lexicon: NameLexicon, own_name: str = "") -> list[Span]:
    """
    Locate person names with three complementary signals:
      1. honorific + capitalised surname ("Ms. O'Brien"), fail-closed;
      2. adjacent capitalised words matching a known first + last name;
      3. any token of the row's own customer name ("...shipped to Priya").
    """
    if not text:
        return []
    spans: list[Span] = [m.span() for m in HONORIFIC_RE.finditer(text)]

    words = list(CAP_WORD_RE.finditer(text))
    for left, right in pairwise(words):
        is_adjacent = text[left.end() : right.start()].strip() == ""
        if is_adjacent and left.group() in lexicon.first_names and right.group() in lexicon.last_names:
            spans.append((left.start(), right.end()))

    for token in own_name.split():
        if len(token) >= 2:
            spans.extend(m.span() for m in re.finditer(rf"\b{re.escape(token)}\b", text))

    return _merge_spans(spans)


def contains_name(series: pd.Series, lexicon: NameLexicon) -> pd.Series:
    return series.map(lambda value: bool(find_name_spans(str(value), lexicon)))


def entity_hit_matrix(series: pd.Series, lexicon: NameLexicon) -> pd.DataFrame:
    """Boolean frame: one column per PII entity, True where the value contains it."""
    text = series.astype(str)
    hits = {entity: text.str.contains(pattern, regex=True) for entity, pattern in REGEX_DETECTORS.items()}
    hits[PiiEntity.NAME] = contains_name(text, lexicon)
    return pd.DataFrame(hits)


def detect_pii(df: pd.DataFrame, lexicon: NameLexicon) -> dict[str, Any]:
    """
    Scan every source column for every PII entity and classify it.
    Undeclared columns containing PII are reported as drift (fails the run downstream).
    """
    columns: dict[str, Any] = {}
    undeclared: list[str] = []

    for col in (c for c in df.columns if c != LINEAGE_COLUMN):
        matrix = entity_hit_matrix(df[col], lexicon)
        hits = {str(entity): int(n) for entity, n in matrix.sum().items()}
        detected = any(hits.values())
        rows_with_pii = int(matrix.any(axis=1).sum())

        if col in STRUCTURED_PII_COLUMNS:
            classification = f"STRUCTURED_PII:{STRUCTURED_PII_COLUMNS[col]}"
        elif col in FREE_TEXT_COLUMNS:
            classification = "FREE_TEXT_PII" if detected else "FREE_TEXT_CLEAN"
        elif detected:
            classification = "UNDECLARED_PII"
            undeclared.append(col)
        else:
            classification = "NON_PII"

        columns[col] = {
            "classification": classification,
            "entity_hits": hits,
            "rows_with_pii": rows_with_pii,
            "pii_row_rate_pct": round(100 * rows_with_pii / len(df), 3),
        }
        logger.info("Detect  %-20s %-26s rows_with_pii=%s", col, classification, f"{rows_with_pii:,}")

    return {"columns": columns, "undeclared_pii_columns": undeclared}
