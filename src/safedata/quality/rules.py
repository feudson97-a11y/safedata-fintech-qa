"""Row-level data-quality rules evaluated on raw values. True = rule violated."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from safedata.domain.models import PipelineConfig, ValidationRule
from safedata.domain.primitives import AMOUNT_FORMAT_RE, digits_only, luhn_is_valid, parse_dates
from safedata.domain.schema import ALLOWED_MERCHANT_CATEGORIES
from safedata.privacy.patterns import EMAIL_FULL_RE

logger = logging.getLogger(__name__)

_MAD_TO_SIGMA = 0.6745  # scales the median absolute deviation to a normal-equivalent sigma
_UNCATEGORISED = "__UNCATEGORISED__"


def _amount_outliers(amount: pd.Series, eligible: pd.Series, category: pd.Series, threshold: float) -> pd.Series:
    """
    Modified z-score of log(amount) within each merchant category. Median/MAD are
    robust: extreme values cannot inflate the baseline they are measured against.
    """
    log_amount = np.log(amount.where(eligible))
    median = log_amount.groupby(category).transform("median")
    mad = (log_amount - median).abs().groupby(category).transform("median")
    robust_z = _MAD_TO_SIGMA * (log_amount - median) / mad.replace(0, np.nan)
    return robust_z.abs() > threshold


def evaluate_rules(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    amount_raw = df["Transaction_Amount"].astype(str)
    amount_format_ok = amount_raw.str.fullmatch(AMOUNT_FORMAT_RE.pattern)
    amount = pd.to_numeric(amount_raw.where(amount_format_ok), errors="coerce")
    dates = parse_dates(df["Transaction_Date"])
    category_ok = df["Merchant_Category"].isin(ALLOWED_MERCHANT_CATEGORIES)
    txn_id = df["Transaction_ID"].astype(str).str.strip()
    card_raw = df["Credit_Card_Number"].astype(str)
    card_digits = card_raw.map(digits_only)
    email_ok = df["Email_Address"].astype(str).str.strip().str.fullmatch(EMAIL_FULL_RE.pattern)

    rule = ValidationRule
    flags = pd.DataFrame(
        {
            rule.MISSING_TRANSACTION_ID: txn_id == "",
            rule.DUPLICATE_TRANSACTION_ID: (txn_id != "") & txn_id.duplicated(keep=False),
            rule.INVALID_AMOUNT_FORMAT: ~amount_format_ok,
            rule.NON_POSITIVE_AMOUNT: amount_format_ok & (amount <= 0),
            rule.AMOUNT_OUTLIER: _amount_outliers(
                amount,
                eligible=amount_format_ok & (amount > 0),
                category=df["Merchant_Category"].where(category_ok, _UNCATEGORISED),
                threshold=cfg.outlier_z_threshold,
            ),
            rule.INVALID_DATE: dates.isna(),
            rule.FUTURE_DATE: dates > pd.Timestamp(cfg.run_timestamp),
            rule.DATE_BEFORE_MIN: dates < pd.Timestamp(cfg.min_valid_date),
            rule.INVALID_MERCHANT_CATEGORY: ~category_ok,
            rule.INVALID_EMAIL_FORMAT: ~email_ok,
            rule.INVALID_CARD_NUMBER: ~(
                card_raw.str.fullmatch(r"[\d -]+")
                & card_digits.str.len().between(13, 19)
                & card_digits.map(luhn_is_valid)
            ),
        },
        index=df.index,
    )
    flags = flags.fillna(False).astype(bool)
    flags.columns = [str(c) for c in flags.columns]

    violations = {name: int(n) for name, n in flags.sum().items() if n}
    logger.info("Validation: %s rows failed at least one rule %s", f"{int(flags.any(axis=1).sum()):,}", violations)
    return flags


def describe_failures(flags: pd.DataFrame) -> pd.Series:
    """Semicolon-separated list of violated rule names per row."""
    return flags.apply(lambda row: ";".join(row.index[row.to_numpy()]), axis=1)
