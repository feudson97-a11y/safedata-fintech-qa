"""Parity tests: the optional LangGraph engine must behave exactly like the classic pipeline."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from safedata.application.pipeline import MASKED_DATASET, run_pipeline
from safedata.domain.models import PipelineConfig, Redaction
from safedata.privacy.tokenization import Tokenizer
from safedata.synthetic.generator import generate_dataset

TEST_KEY = b"integration-test-key-0123456789"
HAS_LANGGRAPH = importlib.util.find_spec("langgraph") is not None
VOLATILE_RUN_FIELDS = ("run_id", "duration_seconds", "engine")


def _comparable(report: dict) -> dict:
    out = {k: v for k, v in report.items() if k not in ("run", "outputs")}
    out["run"] = {k: v for k, v in report["run"].items() if k not in VOLATILE_RUN_FIELDS}
    out["outputs"] = {k: (v is not None) for k, v in report["outputs"].items()}
    return out


@unittest.skipUnless(HAS_LANGGRAPH, 'optional extra not installed: pip install ".[agentic]"')
class TestGraphEngineParity(unittest.TestCase):
    def setUp(self) -> None:
        from safedata.application.graph import run_pipeline_graph

        self.run_graph = run_pipeline_graph
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        raw, _ = generate_dataset(n_rows=1_500, seed=11)
        self.input = self.root / "raw.csv"
        raw.to_csv(self.input, index=False)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _cfgs(self) -> tuple[PipelineConfig, PipelineConfig]:
        classic = PipelineConfig(input_path=self.input, output_dir=self.root / "classic")
        graph = PipelineConfig(
            input_path=self.input, output_dir=self.root / "graph", run_timestamp=classic.run_timestamp
        )
        return classic, graph

    def _assert_parity(self, classic_cfg: PipelineConfig, graph_cfg: PipelineConfig) -> tuple:
        a = run_pipeline(classic_cfg, Tokenizer(TEST_KEY))
        b = self.run_graph(graph_cfg, Tokenizer(TEST_KEY))
        self.assertEqual(a.passed, b.passed)
        ca, cb = _comparable(a.report), _comparable(b.report)
        ca["run"].pop("config"), cb["run"].pop("config")  # output_dir differs by design
        self.assertEqual(ca, cb)
        return a, b

    def test_identical_report_and_outputs_on_passing_run(self) -> None:
        classic_cfg, graph_cfg = self._cfgs()
        _, b = self._assert_parity(classic_cfg, graph_cfg)
        self.assertTrue(b.passed)
        self.assertEqual(b.report["run"]["engine"], "langgraph")
        for name in (MASKED_DATASET, "failed_qa_log.csv"):
            self.assertEqual(
                (classic_cfg.output_dir / name).read_bytes(), (graph_cfg.output_dir / name).read_bytes(), name
            )

    def test_fail_closed_is_preserved(self) -> None:
        def broken(text: str, lexicon: object, own_name: str = "") -> Redaction:
            return Redaction(text, frozenset())

        classic_cfg, graph_cfg = self._cfgs()
        with mock.patch("safedata.privacy.masking.redact_free_text", broken):
            _, b = self._assert_parity(classic_cfg, graph_cfg)
        self.assertFalse(b.passed)
        self.assertFalse((graph_cfg.output_dir / MASKED_DATASET).exists())


@unittest.skipUnless(HAS_LANGGRAPH, 'optional extra not installed: pip install ".[agentic]"')
class TestGraphExtensions(unittest.TestCase):
    """Human approval (LangGraph interrupt) and advisory AI summary."""

    def setUp(self) -> None:
        from safedata.application.graph import QA_SUMMARY, run_pipeline_graph

        self.run_graph, self.summary_name = run_pipeline_graph, QA_SUMMARY
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.raw, _ = generate_dataset(n_rows=800, seed=7)
        self.input = root / "raw.csv"
        self.raw.to_csv(self.input, index=False)
        self.cfg = PipelineConfig(input_path=self.input, output_dir=root / "out")
        self.requests: list[dict] = []

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _approver(self, decision: object):
        def approve(request: dict) -> object:
            self.requests.append(request)
            return decision

        return approve

    def test_approved_run_publishes(self) -> None:
        result = self.run_graph(self.cfg, Tokenizer(TEST_KEY), approver=self._approver(True))
        self.assertTrue(result.passed)
        self.assertEqual(result.report["run"]["approval"], "approved")
        self.assertTrue((self.cfg.output_dir / MASKED_DATASET).exists())
        self.assertEqual(len(self.requests), 1)

    def test_rejected_run_withholds_everything(self) -> None:
        result = self.run_graph(self.cfg, Tokenizer(TEST_KEY), approver=self._approver(False))
        self.assertFalse(result.passed)
        self.assertEqual(result.report["run"]["status"], "REJECTED")
        self.assertFalse((self.cfg.output_dir / MASKED_DATASET).exists())
        self.assertTrue((self.cfg.output_dir / "qa_report.json").exists())

    def test_only_literal_true_approves(self) -> None:
        result = self.run_graph(self.cfg, Tokenizer(TEST_KEY), approver=self._approver("yes"))
        self.assertEqual(result.report["run"]["status"], "REJECTED")

    def test_approval_request_holds_no_raw_values(self) -> None:
        self.run_graph(self.cfg, Tokenizer(TEST_KEY), approver=self._approver(False))
        text = repr(self.requests[0])
        for column in ("Customer_Name", "Email_Address", "Credit_Card_Number"):
            self.assertNotIn(str(self.raw.loc[0, column]), text)

    def test_failed_gates_skip_approval(self) -> None:
        def broken(text: str, lexicon: object, own_name: str = "") -> Redaction:
            return Redaction(text, frozenset())

        with mock.patch("safedata.privacy.masking.redact_free_text", broken):
            result = self.run_graph(self.cfg, Tokenizer(TEST_KEY), approver=self._approver(True))
        self.assertEqual(self.requests, [])
        self.assertEqual(result.report["run"]["status"], "FAILED")
        self.assertEqual(result.report["run"]["approval"], "not_requested")

    def test_summary_is_written_from_report(self) -> None:
        seen: list[dict] = []

        def summarizer(report: dict) -> str:
            seen.append(report)
            return "## Verdict\nPublished."

        result = self.run_graph(self.cfg, Tokenizer(TEST_KEY), summarizer=summarizer)
        self.assertTrue(result.passed)
        self.assertEqual(seen[0]["run"]["run_id"], result.report["run"]["run_id"])
        summary = (self.cfg.output_dir / self.summary_name).read_text(encoding="utf-8")
        self.assertIn("Verdict", summary)

    def test_summary_failure_never_changes_outcome(self) -> None:
        def failing(report: dict) -> str:
            raise RuntimeError("API down")

        with self.assertLogs("safedata.application.graph", level="WARNING"):
            result = self.run_graph(self.cfg, Tokenizer(TEST_KEY), summarizer=failing)
        self.assertTrue(result.passed)
        self.assertFalse((self.cfg.output_dir / self.summary_name).exists())


if __name__ == "__main__":
    unittest.main()
