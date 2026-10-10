"""Immutable value objects exchanged between layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from safedata.domain.exceptions import ConfigurationError
from safedata.domain.schema import PiiEntity


class ValidationRule(StrEnum):
    MISSING_TRANSACTION_ID = "MISSING_TRANSACTION_ID"
    DUPLICATE_TRANSACTION_ID = "DUPLICATE_TRANSACTION_ID"
    INVALID_AMOUNT_FORMAT = "INVALID_AMOUNT_FORMAT"
    NON_POSITIVE_AMOUNT = "NON_POSITIVE_AMOUNT"
    AMOUNT_OUTLIER = "AMOUNT_OUTLIER"
    INVALID_DATE = "INVALID_DATE"
    FUTURE_DATE = "FUTURE_DATE"
    DATE_BEFORE_MIN = "DATE_BEFORE_MIN"
    INVALID_MERCHANT_CATEGORY = "INVALID_MERCHANT_CATEGORY"
    INVALID_EMAIL_FORMAT = "INVALID_EMAIL_FORMAT"
    INVALID_CARD_NUMBER = "INVALID_CARD_NUMBER"


class GateName(StrEnum):
    ZERO_PII_LEAKAGE = "ZERO_PII_LEAKAGE"
    FINANCIAL_RECONCILIATION = "FINANCIAL_RECONCILIATION"
    ANALYTIC_FIELD_INTEGRITY = "ANALYTIC_FIELD_INTEGRITY"
    ROW_CONSERVATION = "ROW_CONSERVATION"
    SCHEMA_AND_PII_DRIFT = "SCHEMA_AND_PII_DRIFT"
    QUARANTINE_RATE_WITHIN_TOLERANCE = "QUARANTINE_RATE_WITHIN_TOLERANCE"


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Validated run configuration. Invalid values fail fast at construction."""

    input_path: Path
    output_dir: Path
    outlier_z_threshold: float = 3.5  # Iglewicz & Hoaglin modified z-score
    max_quarantine_rate: float = 0.05
    min_valid_date: str = "2015-01-01"
    run_timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if self.outlier_z_threshold <= 0:
            raise ConfigurationError("outlier_z_threshold must be positive")
        if not 0 < self.max_quarantine_rate <= 1:
            raise ConfigurationError("max_quarantine_rate must be in (0, 1]")

    def to_dict(self) -> dict[str, str]:
        return {name: str(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class Redaction:
    text: str
    entities: frozenset[PiiEntity]


@dataclass(frozen=True, slots=True)
class GateResult:
    name: GateName
    passed: bool
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": str(self.name), "passed": self.passed, "details": self.details}


@dataclass(frozen=True, slots=True)
class PipelineResult:
    passed: bool
    report: dict[str, Any]
