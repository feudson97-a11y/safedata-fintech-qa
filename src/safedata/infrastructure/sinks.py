"""Output sinks. The pipeline depends on the ``OutputSink`` protocol, not on a concrete storage."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


class OutputSink(Protocol):
    """Port for publishing artefacts (local disk today; object storage could implement it later)."""

    def write_csv(self, name: str, df: pd.DataFrame) -> str: ...
    def write_json(self, name: str, payload: dict[str, Any]) -> str: ...
    def discard(self, name: str) -> None: ...


class LocalDirectorySink:
    """Atomic writes (temp file + rename) so consumers never observe a partial file."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def _commit(self, name: str, write: Callable[[Path], object]) -> str:
        target = self._root / name
        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            write(tmp)
            tmp.replace(target)
        finally:
            tmp.unlink(missing_ok=True)
        return str(target)

    def write_csv(self, name: str, df: pd.DataFrame) -> str:
        return self._commit(name, lambda tmp: df.to_csv(tmp, index=False))

    def write_json(self, name: str, payload: dict[str, Any]) -> str:
        return self._commit(
            name, lambda tmp: tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        )

    def discard(self, name: str) -> None:
        (self._root / name).unlink(missing_ok=True)
