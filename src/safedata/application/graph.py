"""
Optional LangGraph engine for the masking & validation use case.

Same stages, same business functions and same all-or-nothing contract as ``pipeline.run_pipeline``;
only the orchestration differs. Profile, detection, masking and validation are independent of each
other, so they run as parallel branches and join at the QA-gate node:

    ingest -> {profile, detect, mask, validate} -> assure -> publish | withhold -> report

LangGraph is an optional dependency (``pip install "safedata-fintech-qa[agentic]"``). Importing this
module without it raises a ``ConfigurationError`` with install instructions; the default engine never
imports it.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Literal, TypedDict

import pandas as pd

from safedata.application.pipeline import (
    FAILED_QA_LOG,
    MASKED_DATASET,
    QA_REPORT,
    _log_scorecard,
    _split_by_validation,
)
from safedata.domain.exceptions import ConfigurationError
from safedata.domain.models import GateResult, PipelineConfig, PipelineResult
from safedata.domain.schema import OUTPUT_COLUMNS
from safedata.infrastructure.reader import file_sha256, read_transactions
from safedata.infrastructure.sinks import LocalDirectorySink, OutputSink
from safedata.privacy.detection import NameLexicon, detect_pii
from safedata.privacy.masking import mask_dataset
from safedata.privacy.tokenization import Tokenizer
from safedata.quality import gates
from safedata.quality.profiling import profile
from safedata.quality.rules import evaluate_rules

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover - exercised only when the extra is missing
    raise ConfigurationError(
        'The LangGraph engine requires the optional extra: pip install "safedata-fintech-qa[agentic]"'
    ) from exc

logger = logging.getLogger(__name__)


class PipelineState(TypedDict, total=False):
    # Inputs
    cfg: PipelineConfig
    tokenizer: Tokenizer
    sink: OutputSink
    run_id: str
    started: float
    # Stage outputs (each written by exactly one node, so parallel branches never conflict)
    raw: pd.DataFrame
    lexicon: NameLexicon
    profile: dict[str, Any]
    detection: dict[str, Any]
    masked: pd.DataFrame
    masking_stats: dict[str, Any]
    flags: pd.DataFrame
    clean: pd.DataFrame
    quarantined: pd.DataFrame
    gate_results: list[GateResult]
    passed: bool
    outputs: dict[str, str | None]
    report: dict[str, Any]


# --- Nodes ---------------------------------------------------------------------------------------


def _ingest(state: PipelineState) -> PipelineState:
    raw = read_transactions(state["cfg"].input_path)
    return {"raw": raw, "lexicon": NameLexicon.from_names(raw["Customer_Name"])}


def _profile(state: PipelineState) -> PipelineState:
    return {"profile": profile(state["raw"])}


def _detect(state: PipelineState) -> PipelineState:
    return {"detection": detect_pii(state["raw"], state["lexicon"])}


def _mask(state: PipelineState) -> PipelineState:
    masked, stats = mask_dataset(state["raw"], state["tokenizer"], state["lexicon"])
    return {"masked": masked, "masking_stats": stats}


def _validate(state: PipelineState) -> PipelineState:
    return {"flags": evaluate_rules(state["raw"], state["cfg"])}


def _assure(state: PipelineState) -> PipelineState:
    raw, cfg = state["raw"], state["cfg"]
    clean, quarantined = _split_by_validation(state["masked"], state["flags"])
    published = pd.concat([clean, quarantined[list(OUTPUT_COLUMNS)]], ignore_index=True)
    results = [
        gates.zero_pii_leakage(raw, published, state["lexicon"]),
        gates.financial_reconciliation(raw, clean, quarantined),
        gates.analytic_field_integrity(raw, published),
        gates.row_conservation(raw, clean, quarantined),
        gates.schema_and_pii_drift(state["detection"], state["profile"]),
        gates.quarantine_rate_within_tolerance(len(quarantined), len(raw), cfg.max_quarantine_rate),
    ]
    return {
        "clean": clean,
        "quarantined": quarantined,
        "gate_results": results,
        "passed": all(g.passed for g in results),
    }


def _route(state: PipelineState) -> Literal["publish", "withhold"]:
    return "publish" if state["passed"] else "withhold"


def _publish(state: PipelineState) -> PipelineState:
    sink = state["sink"]
    return {
        "outputs": {
            "masked_dataset": sink.write_csv(MASKED_DATASET, state["clean"]),
            "failed_qa_log": sink.write_csv(FAILED_QA_LOG, state["quarantined"]),
        }
    }


def _withhold(state: PipelineState) -> PipelineState:
    # All-or-nothing: no row-level data leaves the pipeline, and stale outputs are removed.
    for name in (MASKED_DATASET, FAILED_QA_LOG):
        state["sink"].discard(name)
    return {"outputs": {"masked_dataset": None, "failed_qa_log": None}}


def _report(state: PipelineState) -> PipelineState:
    cfg, gate_results = state["cfg"], state["gate_results"]
    report: dict[str, Any] = {
        "run": {
            "run_id": state["run_id"],
            "status": "PASSED" if state["passed"] else "FAILED",
            "started_at": cfg.run_timestamp.isoformat(timespec="seconds"),
            "duration_seconds": round(time.perf_counter() - state["started"], 3),
            "input_file": str(cfg.input_path),
            "input_sha256": file_sha256(cfg.input_path),
            "config": cfg.to_dict(),
            "engine": "langgraph",
        },
        "profile": state["profile"],
        "pii_detection": state["detection"],
        "masking": state["masking_stats"],
        "validation": {
            "rows_passed": len(state["clean"]),
            "rows_quarantined": len(state["quarantined"]),
            "rule_violation_counts": {k: int(v) for k, v in state["flags"].sum().items()},
        },
        "qa_gates": [g.to_dict() for g in gate_results],
        "outputs": dict(state["outputs"]),
    }
    report["outputs"]["qa_report"] = state["sink"].write_json(QA_REPORT, report)

    _log_scorecard(gate_results)
    recon = gate_results[1].details
    logger.info(
        "Reconciliation: source %s = clean %s + quarantined %s",
        recon["source_total"],
        recon["clean_total"],
        recon["quarantined_total"],
    )
    logger.info("Run %s %s in %.2fs", state["run_id"], report["run"]["status"], report["run"]["duration_seconds"])
    return {"report": report}


# --- Graph ---------------------------------------------------------------------------------------

_PARALLEL_STAGES = ("profile", "detect", "mask", "validate")


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("ingest", _ingest)
    graph.add_node("profile", _profile)
    graph.add_node("detect", _detect)
    graph.add_node("mask", _mask)
    graph.add_node("validate", _validate)
    graph.add_node("assure", _assure)
    graph.add_node("publish", _publish)
    graph.add_node("withhold", _withhold)
    graph.add_node("report", _report)

    graph.add_edge(START, "ingest")
    for stage in _PARALLEL_STAGES:
        graph.add_edge("ingest", stage)
    graph.add_edge(list(_PARALLEL_STAGES), "assure")  # join: waits for all four branches
    graph.add_conditional_edges("assure", _route, {"publish": "publish", "withhold": "withhold"})
    graph.add_edge("publish", "report")
    graph.add_edge("withhold", "report")
    graph.add_edge("report", END)
    return graph.compile()


def run_pipeline_graph(cfg: PipelineConfig, tokenizer: Tokenizer, sink: OutputSink | None = None) -> PipelineResult:
    """Drop-in alternative to ``pipeline.run_pipeline`` with identical inputs, outputs and guarantees."""
    run_id = uuid.uuid4().hex[:12]
    logger.info("Run %s started (engine: langgraph)", run_id)
    final = build_graph().invoke(
        {
            "cfg": cfg,
            "tokenizer": tokenizer,
            "sink": sink or LocalDirectorySink(cfg.output_dir),
            "run_id": run_id,
            "started": time.perf_counter(),
        }
    )
    return PipelineResult(passed=final["passed"], report=final["report"])
