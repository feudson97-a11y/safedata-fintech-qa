"""
Optional LangGraph engine for the masking & validation use case.

Same stages, same business functions and same all-or-nothing contract as ``pipeline.run_pipeline``;
only the orchestration differs. Profile, detection, masking and validation are independent of each
other, so they run as parallel branches and join at the QA-gate node:

    ingest -> {profile, detect, mask, validate} -> assure -> [approve] -> publish | withhold -> report -> [summarize]

Two opt-in extensions, both off by default (the default graph is unchanged):

* **Human approval** (``approver``): when every QA gate passes, the graph pauses with a LangGraph
  ``interrupt`` and shows the reviewer aggregates only. Rejection withholds all row-level output.
  The checkpointer is in-memory and process-local by design: graph state holds raw rows, so it must
  never be persisted.
* **AI summary** (``summarizer``): after the audit report is written, a callable (e.g. Claude) turns
  the aggregates-only report into ``qa_summary.md``. The step is advisory: a failure is logged and
  never changes the run outcome.

LangGraph is an optional dependency (``pip install "safedata-fintech-qa[agentic]"``). Importing this
module without it raises a ``ConfigurationError`` with install instructions; the default engine never
imports it.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Literal, TypedDict

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
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.graph import END, START, StateGraph
    from langgraph.runtime import Runtime
    from langgraph.types import Command, interrupt
except ImportError as exc:  # pragma: no cover - exercised only when the extra is missing
    raise ConfigurationError(
        'The LangGraph engine requires the optional extra: pip install "safedata-fintech-qa[agentic]"'
    ) from exc

logger = logging.getLogger(__name__)

QA_SUMMARY: Final = "qa_summary.md"

Approver = Callable[[dict[str, Any]], bool]
Summarizer = Callable[[dict[str, Any]], str]


@dataclass(frozen=True, slots=True)
class RunContext:
    """Per-run dependencies. Kept out of graph state so they are never checkpointed."""

    cfg: PipelineConfig
    tokenizer: Tokenizer
    sink: OutputSink
    run_id: str
    started: float
    require_approval: bool = False
    summarizer: Summarizer | None = None


class PipelineState(TypedDict, total=False):
    # Each key is written by exactly one node, so parallel branches never conflict.
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
    approval: Literal["approved", "rejected"]
    outputs: dict[str, str | None]
    report: dict[str, Any]


# --- Nodes ---------------------------------------------------------------------------------------


def _ingest(state: PipelineState, runtime: Runtime[RunContext]) -> PipelineState:
    raw = read_transactions(runtime.context.cfg.input_path)
    return {"raw": raw, "lexicon": NameLexicon.from_names(raw["Customer_Name"])}


def _profile(state: PipelineState) -> PipelineState:
    return {"profile": profile(state["raw"])}


def _detect(state: PipelineState) -> PipelineState:
    return {"detection": detect_pii(state["raw"], state["lexicon"])}


def _mask(state: PipelineState, runtime: Runtime[RunContext]) -> PipelineState:
    masked, stats = mask_dataset(state["raw"], runtime.context.tokenizer, state["lexicon"])
    return {"masked": masked, "masking_stats": stats}


def _validate(state: PipelineState, runtime: Runtime[RunContext]) -> PipelineState:
    return {"flags": evaluate_rules(state["raw"], runtime.context.cfg)}


def _assure(state: PipelineState, runtime: Runtime[RunContext]) -> PipelineState:
    raw = state["raw"]
    clean, quarantined = _split_by_validation(state["masked"], state["flags"])
    published = pd.concat([clean, quarantined[list(OUTPUT_COLUMNS)]], ignore_index=True)
    results = [
        gates.zero_pii_leakage(raw, published, state["lexicon"]),
        gates.financial_reconciliation(raw, clean, quarantined),
        gates.analytic_field_integrity(raw, published),
        gates.row_conservation(raw, clean, quarantined),
        gates.schema_and_pii_drift(state["detection"], state["profile"]),
        gates.quarantine_rate_within_tolerance(len(quarantined), len(raw), runtime.context.cfg.max_quarantine_rate),
    ]
    return {
        "clean": clean,
        "quarantined": quarantined,
        "gate_results": results,
        "passed": all(g.passed for g in results),
    }


def _route_after_assure(
    state: PipelineState, runtime: Runtime[RunContext]
) -> Literal["approve", "publish", "withhold"]:
    if not state["passed"]:
        return "withhold"  # failed gates are never offered for approval
    return "approve" if runtime.context.require_approval else "publish"


def approval_request(state: PipelineState) -> dict[str, Any]:
    """What the reviewer sees: aggregates only, never row-level values."""
    recon = state["gate_results"][1].details
    return {
        "rows_total": len(state["raw"]),
        "rows_clean": len(state["clean"]),
        "rows_quarantined": len(state["quarantined"]),
        "qa_gates": {str(g.name): g.passed for g in state["gate_results"]},
        "reconciliation": {k: recon[k] for k in ("source_total", "clean_total", "quarantined_total", "variance")},
    }


def _approve(state: PipelineState) -> PipelineState:
    decision = interrupt(approval_request(state))
    return {"approval": "approved" if decision is True else "rejected"}


def _route_after_approval(state: PipelineState) -> Literal["publish", "withhold"]:
    return "publish" if state["approval"] == "approved" else "withhold"


def _publish(state: PipelineState, runtime: Runtime[RunContext]) -> PipelineState:
    sink = runtime.context.sink
    return {
        "outputs": {
            "masked_dataset": sink.write_csv(MASKED_DATASET, state["clean"]),
            "failed_qa_log": sink.write_csv(FAILED_QA_LOG, state["quarantined"]),
        }
    }


def _withhold(state: PipelineState, runtime: Runtime[RunContext]) -> PipelineState:
    # All-or-nothing: no row-level data leaves the pipeline, and stale outputs are removed.
    for name in (MASKED_DATASET, FAILED_QA_LOG):
        runtime.context.sink.discard(name)
    return {"outputs": {"masked_dataset": None, "failed_qa_log": None}}


def _status(state: PipelineState) -> str:
    if not state["passed"]:
        return "FAILED"
    return "REJECTED" if state.get("approval") == "rejected" else "PASSED"


def _report(state: PipelineState, runtime: Runtime[RunContext]) -> PipelineState:
    ctx = runtime.context
    cfg, gate_results = ctx.cfg, state["gate_results"]
    run: dict[str, Any] = {
        "run_id": ctx.run_id,
        "status": _status(state),
        "started_at": cfg.run_timestamp.isoformat(timespec="seconds"),
        "duration_seconds": round(time.perf_counter() - ctx.started, 3),
        "input_file": str(cfg.input_path),
        "input_sha256": file_sha256(cfg.input_path),
        "config": cfg.to_dict(),
        "engine": "langgraph",
    }
    if ctx.require_approval:
        run["approval"] = state.get("approval", "not_requested")  # not_requested: gates failed first
    report: dict[str, Any] = {
        "run": run,
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
    report["outputs"]["qa_report"] = ctx.sink.write_json(QA_REPORT, report)

    _log_scorecard(gate_results)
    recon = gate_results[1].details
    logger.info(
        "Reconciliation: source %s = clean %s + quarantined %s",
        recon["source_total"],
        recon["clean_total"],
        recon["quarantined_total"],
    )
    logger.info("Run %s %s in %.2fs", ctx.run_id, run["status"], run["duration_seconds"])
    return {"report": report}


def _route_after_report(state: PipelineState, runtime: Runtime[RunContext]) -> Literal["summarize", "__end__"]:
    return "summarize" if runtime.context.summarizer is not None else END


def _summarize(state: PipelineState, runtime: Runtime[RunContext]) -> PipelineState:
    ctx = runtime.context
    if ctx.summarizer is None:  # unreachable: _route_after_report only routes here with a summarizer
        return {}
    try:
        text = ctx.summarizer(state["report"])
        path = ctx.sink.write_text(QA_SUMMARY, text)
    except Exception:  # advisory step: must never change the run outcome (logged with traceback)
        logger.warning("AI summary skipped: summarizer failed", exc_info=True)
        return {}
    logger.info("AI summary written to %s", path)
    return {}


# --- Graph ---------------------------------------------------------------------------------------

# Domain types allowed back out of the checkpoint (explicit allow-list instead of arbitrary deserialization).
_CHECKPOINT_TYPES = [
    ("safedata.privacy.detection", "NameLexicon"),
    ("safedata.domain.models", "GateName"),
    ("safedata.domain.models", "GateResult"),
]


def _approval_checkpointer() -> InMemorySaver:
    """In-memory, process-local: required to pause/resume, never persisted (graph state holds raw rows)."""
    try:
        serde = JsonPlusSerializer(pickle_fallback=True, allowed_msgpack_modules=_CHECKPOINT_TYPES)
    except TypeError:  # older langgraph-checkpoint without the allow-list option
        serde = JsonPlusSerializer(pickle_fallback=True)
    return InMemorySaver(serde=serde)


_PARALLEL_STAGES = ("profile", "detect", "mask", "validate")


def build_graph(*, checkpointer: InMemorySaver | None = None):
    graph = StateGraph(PipelineState, context_schema=RunContext)
    graph.add_node("ingest", _ingest)
    graph.add_node("profile", _profile)
    graph.add_node("detect", _detect)
    graph.add_node("mask", _mask)
    graph.add_node("validate", _validate)
    graph.add_node("assure", _assure)
    graph.add_node("approve", _approve)
    graph.add_node("publish", _publish)
    graph.add_node("withhold", _withhold)
    graph.add_node("report", _report)
    graph.add_node("summarize", _summarize)

    graph.add_edge(START, "ingest")
    for stage in _PARALLEL_STAGES:
        graph.add_edge("ingest", stage)
    graph.add_edge(list(_PARALLEL_STAGES), "assure")  # join: waits for all four branches
    graph.add_conditional_edges("assure", _route_after_assure, ["approve", "publish", "withhold"])
    graph.add_conditional_edges("approve", _route_after_approval, ["publish", "withhold"])
    graph.add_edge("publish", "report")
    graph.add_edge("withhold", "report")
    graph.add_conditional_edges("report", _route_after_report, ["summarize", END])
    graph.add_edge("summarize", END)
    return graph.compile(checkpointer=checkpointer)


def run_pipeline_graph(
    cfg: PipelineConfig,
    tokenizer: Tokenizer,
    sink: OutputSink | None = None,
    *,
    approver: Approver | None = None,
    summarizer: Summarizer | None = None,
) -> PipelineResult:
    """
    Drop-in alternative to ``pipeline.run_pipeline``. With no ``approver`` and no ``summarizer`` it has
    identical inputs, outputs and guarantees. ``approver`` receives the aggregates-only approval request
    and returns True to publish; anything else withholds the data (fail-closed).
    """
    run_id = uuid.uuid4().hex[:12]
    logger.info("Run %s started (engine: langgraph)", run_id)
    context = RunContext(
        cfg=cfg,
        tokenizer=tokenizer,
        sink=sink or LocalDirectorySink(cfg.output_dir),
        run_id=run_id,
        started=time.perf_counter(),
        require_approval=approver is not None,
        summarizer=summarizer,
    )

    if approver is None:
        final = build_graph().invoke({}, context=context)
    else:
        graph = build_graph(checkpointer=_approval_checkpointer())
        config = {"configurable": {"thread_id": run_id}}
        final = graph.invoke({}, config, context=context)
        if pending := final.get("__interrupt__"):
            decision = approver(pending[0].value) is True
            logger.info("Approval for run %s: %s", run_id, "approved" if decision else "rejected")
            final = graph.invoke(Command(resume=decision), config, context=context)

    report = final["report"]
    return PipelineResult(passed=report["run"]["status"] == "PASSED", report=report)
