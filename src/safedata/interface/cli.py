"""
Command-line interface.

    safedata generate [--rows N] [--seed S] [--defect-rate R] [--output PATH]
    safedata run      [--input PATH] [--output-dir DIR] [--outlier-z Z] [--max-quarantine-rate R]
                      [--engine {classic,langgraph}]

Exit codes: 0 success - 1 QA gate failed (data withheld) - 2 input/config error - 3 unexpected error.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from enum import IntEnum
from pathlib import Path

from safedata import __version__
from safedata.application.pipeline import run_pipeline
from safedata.domain.exceptions import ConfigurationError, InputError
from safedata.domain.models import PipelineConfig
from safedata.privacy.tokenization import Tokenizer
from safedata.synthetic.generator import generate_dataset

logger = logging.getLogger("safedata")


class ExitCode(IntEnum):
    SUCCESS = 0
    QA_GATE_FAILED = 1
    INPUT_OR_CONFIG_ERROR = 2
    UNEXPECTED_ERROR = 3


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="safedata", description="PII masking & data-quality pipeline for Fintech data."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    commands = parser.add_subparsers(dest="command", required=True)

    gen = commands.add_parser("generate", help="Generate a synthetic transaction dataset.")
    gen.add_argument("--rows", type=int, default=10_000, help="Number of transactions (default: 10000).")
    gen.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility (default: 42).")
    gen.add_argument("--defect-rate", type=float, default=0.02, help="Share of rows given an injected defect.")
    gen.add_argument("--output", type=Path, default=Path("data/raw_transactions.csv"), help="Output CSV path.")

    run = commands.add_parser("run", help="Mask, validate and publish a transaction extract.")
    run.add_argument("--input", type=Path, default=Path("data/raw_transactions.csv"), help="Raw transactions CSV.")
    run.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory for outputs.")
    run.add_argument("--outlier-z", type=float, default=3.5, help="Modified z-score threshold for amount outliers.")
    run.add_argument(
        "--max-quarantine-rate", type=float, default=0.05, help="Max share of rows that may be quarantined."
    )
    run.add_argument(
        "--engine",
        choices=["classic", "langgraph"],
        default="classic",
        help='Orchestration engine (default: classic). "langgraph" needs the [agentic] extra.',
    )
    return parser


def _cmd_generate(args: argparse.Namespace) -> ExitCode:
    try:
        df, manifest = generate_dataset(n_rows=args.rows, seed=args.seed, defect_rate=args.defect_rate)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    manifest_path = args.output.with_name("injected_defects.csv")
    manifest.to_csv(manifest_path, index=False)
    logger.info(
        "Wrote %s rows to %s; %d injected defects listed in %s",
        f"{len(df):,}",
        args.output,
        len(manifest),
        manifest_path,
    )
    return ExitCode.SUCCESS


def _cmd_run(args: argparse.Namespace) -> ExitCode:
    cfg = PipelineConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        outlier_z_threshold=args.outlier_z,
        max_quarantine_rate=args.max_quarantine_rate,
    )
    if args.engine == "langgraph":
        from safedata.application.graph import run_pipeline_graph  # lazy: optional dependency

        result = run_pipeline_graph(cfg, Tokenizer.from_environment())
    else:
        result = run_pipeline(cfg, Tokenizer.from_environment())
    return ExitCode.SUCCESS if result.passed else ExitCode.QA_GATE_FAILED


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.log_level)
    try:
        match args.command:
            case "generate":
                return _cmd_generate(args)
            case "run":
                return _cmd_run(args)
            case _:  # unreachable: argparse enforces the sub-command
                return ExitCode.INPUT_OR_CONFIG_ERROR
    except (InputError, ConfigurationError) as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return ExitCode.INPUT_OR_CONFIG_ERROR
    except OSError as exc:
        logger.error("I/O failure: %s", exc)
        return ExitCode.INPUT_OR_CONFIG_ERROR
    except Exception:  # last-resort boundary: log the traceback, never crash silently
        logger.exception("Unexpected failure")
        return ExitCode.UNEXPECTED_ERROR
