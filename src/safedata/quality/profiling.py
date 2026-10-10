"""Aggregate-only data profile: contains no raw values, so it is safe to share."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from safedata.domain.primitives import amount_to_cents, parse_dates
from safedata.domain.schema import EXPECTED_COLUMNS, LINEAGE_COLUMN

logger = logging.getLogger(__name__)


def profile(df: pd.DataFrame) -> dict[str, Any]:
    row_count = len(df)
    columns: dict[str, Any] = {}
    for col in EXPECTED_COLUMNS:
        series = df[col].astype(str)
        blanks = int((series.str.strip() == "").sum())
        columns[col] = {
            "blank_count": blanks,
            "completeness_pct": round(100 * (1 - blanks / row_count), 3),
            "distinct_count": int(series.nunique()),
        }

    cents = df["Transaction_Amount"].map(amount_to_cents)
    amounts = cents.dropna().astype(np.int64) / 100
    dates = parse_dates(df["Transaction_Date"])
    has_dates = bool(dates.notna().any())

    result: dict[str, Any] = {
        "row_count": row_count,
        "unexpected_columns": [c for c in df.columns if c not in EXPECTED_COLUMNS and c != LINEAGE_COLUMN],
        "duplicate_full_rows": int(df.duplicated(subset=list(EXPECTED_COLUMNS)).sum()),
        "columns": columns,
        "transaction_amount": {
            "parseable": int(cents.notna().sum()),
            "unparseable": int(cents.isna().sum()),
            "min": float(amounts.min()) if len(amounts) else None,
            "median": float(amounts.median()) if len(amounts) else None,
            "max": float(amounts.max()) if len(amounts) else None,
        },
        "transaction_date": {
            "parseable": int(dates.notna().sum()),
            "unparseable": int(dates.isna().sum()),
            "min": str(dates.min()) if has_dates else None,
            "max": str(dates.max()) if has_dates else None,
        },
    }
    logger.info(
        "Profile: %d amounts unparseable, %d dates unparseable, %d unexpected columns",
        result["transaction_amount"]["unparseable"],
        result["transaction_date"]["unparseable"],
        len(result["unexpected_columns"]),
    )
    return result
