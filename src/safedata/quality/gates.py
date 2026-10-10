"""
Dataset-level QA gates. Any failure blocks the release of all row-level data.
Gates report counts and row numbers only, never the offending values.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pandas as pd

from safedata.domain.models import GateName, GateResult
from safedata.domain.primitives import amount_to_cents, digits_only, format_cents, parse_dates
from safedata.domain.schema import LINEAGE_COLUMN, PROTECTED_ANALYTIC_COLUMNS, TOKEN_COLUMNS
from safedata.privacy.detection import NameLexicon, contains_name
from safedata.privacy.patterns import DIGIT_RUN_RE, REGEX_DETECTORS

_ROW_SAMPLE_LIMIT = 20


def _pattern_rescan(published: pd.DataFrame, scan_cols: list[str], lexicon: NameLexicon) -> dict[str, int]:
    hits: dict[str, int] = {}
    for col in scan_cols:
        series = published[col].astype(str)
        for entity, pattern in REGEX_DETECTORS.items():
            if n := int(series.str.contains(pattern, regex=True).sum()):
                hits[f"{col}:{entity}"] = n
        if n := int(contains_name(series, lexicon).sum()):
            hits[f"{col}:NAME"] = n
    return hits


def _row_leaks(row_text: str, email: str, phone: str, card: str, name: str) -> set[str]:
    """Which of this row's own original PII values survive anywhere in its published text."""
    leaked: set[str] = set()
    digit_runs = {digits_only(m.group()) for m in DIGIT_RUN_RE.finditer(row_text)}

    email = email.strip().lower()
    if len(email) > 3 and email in row_text.lower():
        leaked.add("EMAIL")
    phone = digits_only(phone)[-10:]
    if len(phone) == 10 and any(phone in run for run in digit_runs):
        leaked.add("PHONE")
    card = digits_only(card)
    if len(card) >= 13 and any(card in run for run in digit_runs):
        leaked.add("CARD")
    if any(len(part) >= 2 and re.search(rf"\b{re.escape(part)}\b", row_text) for part in name.split()):
        leaked.add("NAME")
    return leaked


def zero_pii_leakage(raw: pd.DataFrame, published: pd.DataFrame, lexicon: NameLexicon) -> GateResult:
    """
    Two independent tests over everything that will be published:
      A. pattern rescan: no email/phone/card/name pattern survives anywhere;
      B. exact-value scan: no row contains its own original PII values.
    Token columns are excluded from A (random hex can mimic digit runs) and
    validated by strict format instead.
    """
    scan_cols = [c for c in published.columns if c not in TOKEN_COLUMNS and c != LINEAGE_COLUMN]
    pattern_hits = _pattern_rescan(published, scan_cols, lexicon)

    token_violations = {
        col: n
        for col, rx in TOKEN_COLUMNS.items()
        if (n := int((~published[col].astype(str).str.fullmatch(rx.pattern)).sum()))
    }

    src = raw.set_index(LINEAGE_COLUMN).reindex(published[LINEAGE_COLUMN])
    row_texts = published[scan_cols].astype(str).agg("\x1f".join, axis=1)
    exact_leaks: Counter[str] = Counter()
    leaking_rows: list[int] = []
    for row_no, row_text, email, phone, card, name in zip(
        published[LINEAGE_COLUMN],
        row_texts,
        src["Email_Address"].astype(str),
        src["Phone_Number"].astype(str),
        src["Credit_Card_Number"].astype(str),
        src["Customer_Name"].astype(str),
        strict=True,
    ):
        if leaked := _row_leaks(row_text, email, phone, card, name):
            exact_leaks.update(leaked)
            leaking_rows.append(int(row_no))

    return GateResult(
        name=GateName.ZERO_PII_LEAKAGE,
        passed=not (pattern_hits or token_violations or exact_leaks),
        details={
            "columns_scanned": scan_cols,
            "rows_scanned": len(published),
            "pattern_rescan_hits": pattern_hits,
            "exact_value_leaks_by_type": dict(exact_leaks),
            "leaking_row_numbers_sample": leaking_rows[:_ROW_SAMPLE_LIMIT],
            "token_format_violations": token_violations,
        },
    )


def _control_total(series: pd.Series) -> tuple[int, int, int]:
    """(total cents, parseable count, unparseable count) using exact Decimal arithmetic."""
    cents = [amount_to_cents(str(v)) for v in series]
    valid = [c for c in cents if c is not None]
    return sum(valid), len(valid), len(cents) - len(valid)


def financial_reconciliation(raw: pd.DataFrame, clean: pd.DataFrame, quarantined: pd.DataFrame) -> GateResult:
    """Integer-cent control totals: source == clean + quarantined, with counts conserved."""
    src_total, src_n, src_bad = _control_total(raw["Transaction_Amount"])
    cln_total, cln_n, cln_bad = _control_total(clean["Transaction_Amount"])
    qrt_total, qrt_n, qrt_bad = _control_total(quarantined["Transaction_Amount"])

    return GateResult(
        name=GateName.FINANCIAL_RECONCILIATION,
        passed=(src_total == cln_total + qrt_total and src_n == cln_n + qrt_n and src_bad == cln_bad + qrt_bad),
        details={
            "source_total": format_cents(src_total),
            "clean_total": format_cents(cln_total),
            "quarantined_total": format_cents(qrt_total),
            "variance": format_cents(src_total - cln_total - qrt_total),
            "source_parseable_amounts": src_n,
            "published_parseable_amounts": cln_n + qrt_n,
            "source_unparseable_amounts": src_bad,
            "published_unparseable_amounts": cln_bad + qrt_bad,
        },
    )


def _analytic_stats(frame: pd.DataFrame) -> dict[str, Any]:
    amounts = pd.to_numeric(frame["Transaction_Amount"], errors="coerce")
    dates = parse_dates(frame["Transaction_Date"])
    return {
        "amount_mean": round(float(amounts.mean()), 6),
        "amount_std": round(float(amounts.std()), 6),
        "date_min": str(dates.min()),
        "date_max": str(dates.max()),
    }


def analytic_field_integrity(raw: pd.DataFrame, published: pd.DataFrame) -> GateResult:
    """Every protected value must be byte-identical to its source row (a sum alone can hide offsetting errors)."""
    cols = [LINEAGE_COLUMN, *PROTECTED_ANALYTIC_COLUMNS]
    merged = raw[cols].merge(published[cols], on=LINEAGE_COLUMN, how="inner", suffixes=("_src", "_out"))
    mismatches = {
        col: int((merged[f"{col}_src"].astype(str) != merged[f"{col}_out"].astype(str)).sum())
        for col in PROTECTED_ANALYTIC_COLUMNS
    }
    before, after = _analytic_stats(raw), _analytic_stats(published)
    return GateResult(
        name=GateName.ANALYTIC_FIELD_INTEGRITY,
        passed=len(merged) == len(raw) and not any(mismatches.values()) and before == after,
        details={
            "rows_compared": len(merged),
            "value_mismatches": mismatches,
            "stats_before": before,
            "stats_after": after,
        },
    )


def row_conservation(raw: pd.DataFrame, clean: pd.DataFrame, quarantined: pd.DataFrame) -> GateResult:
    """Every source row lands in exactly one output: none lost, none duplicated."""
    src = set(raw[LINEAGE_COLUMN])
    cln, qrt = set(clean[LINEAGE_COLUMN]), set(quarantined[LINEAGE_COLUMN])
    return GateResult(
        name=GateName.ROW_CONSERVATION,
        passed=(cln | qrt) == src and not (cln & qrt) and len(clean) + len(quarantined) == len(raw),
        details={
            "source_rows": len(raw),
            "clean_rows": len(clean),
            "quarantined_rows": len(quarantined),
            "missing_rows": len(src - (cln | qrt)),
            "rows_in_both_outputs": len(cln & qrt),
        },
    )


def schema_and_pii_drift(detection: dict[str, Any], prof: dict[str, Any]) -> GateResult:
    undeclared = detection["undeclared_pii_columns"]
    return GateResult(
        name=GateName.SCHEMA_AND_PII_DRIFT,
        passed=not undeclared,
        details={"undeclared_pii_columns": undeclared, "unexpected_columns_dropped": prof["unexpected_columns"]},
    )


def quarantine_rate_within_tolerance(n_quarantined: int, n_total: int, max_rate: float) -> GateResult:
    rate = n_quarantined / n_total
    return GateResult(
        name=GateName.QUARANTINE_RATE_WITHIN_TOLERANCE,
        passed=rate <= max_rate,
        details={"quarantine_rate_pct": round(100 * rate, 3), "threshold_pct": round(100 * max_rate, 3)},
    )
