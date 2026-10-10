"""SafeData Fintech QA - PII governance and data-quality validation for Fintech datasets."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("safedata-fintech-qa")
except PackageNotFoundError:  # running from a source checkout without installation
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
