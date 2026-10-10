"""
Masking & validation use case:
Ingest -> Profile -> Detect -> Mask -> Validate -> Assure (QA gates) -> Publish (all-or-nothing).
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Final

import pandas as pd

from safedata.domain.models import GateResult, PipelineConfig, PipelineResult
from safedata.domain.schema import OUTPUT_COLUMNS
from safedata.infrastructure.reader import file_sha256, read_transactions
from safedata.infrastructure.sinks import LocalDirectorySink, OutputSink
from safedata.privacy.detection import NameLexicon, detect_pii
from safedata.privacy.masking import mask_dataset
from safedata.privacy.tokenization import Tokenizer
from safedata.quality import gates
from safedata.quality.profiling import profile
from safedata.quality.rules import describe_failures, evaluate_rules

logger = logging.getLogger(__name__)

MASKED_DATASET: Final = "masked_transactions.csv"
FAILED_QA_LOG: Final = "failed_qa_log.csv"
QA_REPORT: Final = "qa_report.json"


def _split_by_validation(masked: pd.DataFrame, flags: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean rows vs. quarantined rows (both already masked) with failure reasons attached."""
    failed = flags.any(axis=1)
    clean = masked.loc[~failed].reset_index(drop=True)
    quarantined = masked.loc[failed].copy()
    quarantined.insert(1, "Failed_Rules", describe_failures(flags.loc[failed]))
    quarantined.insert(2, "Rule_Count", flags.loc[failed].sum(axis=1).astype(int))
    return clean, quarantined.reset_index(drop=True)


def _log_scorecard(gate_results: list[GateResult]) -> None:
    logger.info("-" * 64)
    for gate in gate_results:
        logger.info("QA GATE  %-34s %s", gate.name, "PASS" if gate.passed else "FAIL")
    logger.info("-" * 64)


def run_pipeline(cfg: PipelineConfig, tokenizer: Tokenizer, sink: OutputSink | None = None) -> PipelineResult:
    started = time.perf_counter()
    run_id = uuid.uuid4().hex[:12]
    sink = sink or LocalDirectorySink(cfg.output_dir)
    logger.info("Run %s started", run_id)

    raw = read_transactions(cfg.input_path)
    prof = profile(raw)
    lexicon = NameLexicon.from_names(raw["Customer_Name"])
    detection = detect_pii(raw, lexicon)
    masked, masking_stats = mask_dataset(raw, tokenizer, lexicon)
    flags = evaluate_rules(raw, cfg)
    clean, quarantined = _split_by_validation(masked, flags)

    published = pd.concat([clean, quarantined[list(OUTPUT_COLUMNS)]], ignore_index=True)
    gate_results = [
        gates.zero_pii_leakage(raw, published, lexicon),
        gates.financial_reconciliation(raw, clean, quarantined),
        gates.analytic_field_integrity(raw, published),
        gates.row_conservation(raw, clean, quarantined),
        gates.schema_and_pii_drift(detection, prof),
        gates.quarantine_rate_within_tolerance(len(quarantined), len(raw), cfg.max_quarantine_rate),
    ]
    passed = all(g.passed for g in gate_results)

    outputs: dict[str, str | None]
    if passed:
        outputs = {
            "masked_dataset": sink.write_csv(MASKED_DATASET, clean),
            "failed_qa_log": sink.write_csv(FAILED_QA_LOG, quarantined),
        }
    else:
        # All-or-nothing: no row-level data leaves the pipeline, and stale outputs are removed.
        for name in (MASKED_DATASET, FAILED_QA_LOG):
            sink.discard(name)
        outputs = {"masked_dataset": None, "failed_qa_log": None}

    report: dict[str, Any] = {
        "run": {
            "run_id": run_id,
            "status": "PASSED" if passed else "FAILED",
            "started_at": cfg.run_timestamp.isoformat(timespec="seconds"),
            "duration_seconds": round(time.perf_counter() - started, 3),
            "input_file": str(cfg.input_path),
            "input_sha256": file_sha256(cfg.input_path),
            "config": cfg.to_dict(),
        },
        "profile": prof,
        "pii_detection": detection,
        "masking": masking_stats,
        "validation": {
            "rows_passed": len(clean),
            "rows_quarantined": len(quarantined),
            "rule_violation_counts": {k: int(v) for k, v in flags.sum().items()},
        },
        "qa_gates": [g.to_dict() for g in gate_results],
        "outputs": outputs,
    }
    report["outputs"]["qa_report"] = sink.write_json(QA_REPORT, report)

    _log_scorecard(gate_results)
    recon = gate_results[1].details
    logger.info(
        "Reconciliation: source %s = clean %s + quarantined %s",
        recon["source_total"],
        recon["clean_total"],
        recon["quarantined_total"],
    )
    logger.info("Run %s %s in %.2fs", run_id, report["run"]["status"], report["run"]["duration_seconds"])
    return PipelineResult(passed=passed, report=report)
