"""Lossless ingestion of the raw transaction extract."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from safedata.domain.exceptions import InputError, SchemaError
from safedata.domain.schema import EXPECTED_COLUMNS, LINEAGE_COLUMN

logger = logging.getLogger(__name__)

_HASH_CHUNK_BYTES = 1 << 20


def read_transactions(path: Path) -> pd.DataFrame:
    """
    Read every field as text with NA-inference disabled: leading zeros, exact
    decimal strings and malformed values arrive verbatim, so validation sees
    exactly what the source system sent. Adds a row-lineage column.
    """
    if not path.is_file():
        raise InputError(f"Input file not found: {path}")
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise InputError(f"Unable to parse {path}: {exc}") from exc

    if missing := [c for c in EXPECTED_COLUMNS if c not in df.columns]:
        raise SchemaError(f"Missing required columns: {missing}")
    if df.empty:
        raise SchemaError("Input file contains no data rows")

    df = df.reset_index(drop=True)
    df.insert(0, LINEAGE_COLUMN, np.arange(len(df), dtype=np.int64))
    logger.info("Ingested %s rows x %d columns from %s", f"{len(df):,}", len(df.columns) - 1, path)
    return df


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
