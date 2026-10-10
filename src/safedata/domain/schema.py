"""Data contract for the transaction extract: the single source of truth for column semantics."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Final


class PiiEntity(StrEnum):
    NAME = "NAME"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    CARD = "CARD"


LINEAGE_COLUMN: Final = "Source_Row_Number"
DATE_FORMAT: Final = "%Y-%m-%d %H:%M:%S"

EXPECTED_COLUMNS: Final[tuple[str, ...]] = (
    "Transaction_ID",
    "Customer_Name",
    "Email_Address",
    "Phone_Number",
    "Credit_Card_Number",
    "Transaction_Amount",
    "Transaction_Date",
    "Merchant_Category",
    "Transaction_Notes",
)

# Columns whose entire value is PII, with the entity they hold.
STRUCTURED_PII_COLUMNS: Final[dict[str, PiiEntity]] = {
    "Customer_Name": PiiEntity.NAME,
    "Email_Address": PiiEntity.EMAIL,
    "Phone_Number": PiiEntity.PHONE,
    "Credit_Card_Number": PiiEntity.CARD,
}

# Columns that may hide PII anywhere inside unstructured text.
FREE_TEXT_COLUMNS: Final[tuple[str, ...]] = ("Transaction_Notes",)

# Columns that must leave the pipeline byte-identical to the source.
PROTECTED_ANALYTIC_COLUMNS: Final[tuple[str, ...]] = ("Transaction_Amount", "Transaction_Date")

# Surrogate keys produced by masking; validated by strict format, not PII rescans.
TOKEN_COLUMNS: Final[dict[str, re.Pattern[str]]] = {
    "Customer_Token": re.compile(r"^(?:CUST_[0-9a-f]{16})?$"),
    "Card_Token": re.compile(r"^(?:CARD_[0-9a-f]{16})?$"),
}

ALLOWED_MERCHANT_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "Groceries",
        "Restaurants",
        "Fuel",
        "E-Commerce",
        "Utilities",
        "Healthcare",
        "Entertainment",
        "Electronics",
        "Travel",
        "Peer-to-Peer Transfer",
    }
)

OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    LINEAGE_COLUMN,
    "Transaction_ID",
    "Customer_Token",
    "Email_Address",
    "Phone_Number",
    "Credit_Card_Number",
    "Card_Token",
    "Transaction_Amount",
    "Transaction_Date",
    "Merchant_Category",
    "Transaction_Notes",
    "Notes_PII_Entities",
)
