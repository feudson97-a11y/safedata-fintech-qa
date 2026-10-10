"""Typed exception hierarchy. The CLI maps each family to a distinct exit code."""


class SafeDataError(Exception):
    """Root of every error raised intentionally by this package."""


class ConfigurationError(SafeDataError):
    """Invalid runtime configuration (thresholds, secrets, arguments)."""


class InputError(SafeDataError):
    """The input file is missing, unreadable or unparseable."""


class SchemaError(InputError):
    """The input does not satisfy the expected data contract."""
