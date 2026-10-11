"""End-to-end tests: generator -> pipeline, including negative (fail-closed) scenarios."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from safedata.application.pipeline import FAILED_QA_LOG, MASKED_DATASET, QA_REPORT, run_pipeline
from safedata.domain.models import GateName, PipelineConfig, Redaction
from safedata.infrastructure.reader import read_transactions
from safedata.interface.cli import ExitCode, main
from safedata.privacy.tokenization import Tokenizer
from safedata.quality import gates
from safedata.synthetic.generator import generate_dataset

TEST_KEY = b"integration-test-key-0123456789"


def _gate(report: dict, name: GateName) -> dict:
    return next(g for g in report["qa_gates"] if g["name"] == name)


class TestEndToEnd(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.raw, self.manifest = generate_dataset(n_rows=1_500, seed=11)
        self.input = root / "raw.csv"
        self.raw.to_csv(self.input, index=False)
        self.cfg = PipelineConfig(input_path=self.input, output_dir=root / "out")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_all_gates_pass_and_every_injected_defect_is_quarantined(self) -> None:
        result = run_pipeline(self.cfg, Tokenizer(TEST_KEY))
        self.assertTrue(result.passed, [g for g in result.report["qa_gates"] if not g["passed"]])
        failed = pd.read_csv(self.cfg.output_dir / FAILED_QA_LOG)
        missed = set(self.manifest["Source_Row_Number"]) - set(failed["Source_Row_Number"])
        self.assertEqual(missed, set())
        self.assertEqual(_gate(result.report, GateName.FINANCIAL_RECONCILIATION)["details"]["variance"], "0.00")

    def test_leak_gate_fails_closed_on_masking_regression(self) -> None:
        def broken_redaction(text: str, lexicon: object, own_name: str = "") -> Redaction:
            return Redaction(text, frozenset())

        with mock.patch("safedata.privacy.masking.redact_free_text", broken_redaction):
            result = run_pipeline(self.cfg, Tokenizer(TEST_KEY))

        self.assertFalse(result.passed)
        self.assertFalse(_gate(result.report, GateName.ZERO_PII_LEAKAGE)["passed"])
        self.assertFalse((self.cfg.output_dir / MASKED_DATASET).exists())
        # The audit report itself must never contain the leaked values.
        report_text = (self.cfg.output_dir / QA_REPORT).read_text(encoding="utf-8")
        self.assertNotIn(self.raw.loc[0, "Email_Address"], report_text)

    def test_reconciliation_detects_tampering(self) -> None:
        raw = read_transactions(self.input)
        clean = raw.iloc[:-1].copy()
        clean.iloc[0, clean.columns.get_loc("Transaction_Amount")] = "0.01"
        self.assertFalse(gates.financial_reconciliation(raw, clean, raw.iloc[-1:]).passed)

    def test_undeclared_pii_column_is_drift(self) -> None:
        df = pd.read_csv(self.input, dtype=str, keep_default_na=False)
        df["Backup_Contact"] = df["Email_Address"]
        df.to_csv(self.input, index=False)
        result = run_pipeline(self.cfg, Tokenizer(TEST_KEY))
        drift = _gate(result.report, GateName.SCHEMA_AND_PII_DRIFT)
        self.assertFalse(result.passed)
        self.assertIn("Backup_Contact", drift["details"]["undeclared_pii_columns"])


class TestCliExitCodes(unittest.TestCase):
    def test_missing_input_returns_input_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = main(["--log-level", "ERROR", "run", "--input", str(Path(tmp) / "nope.csv"), "--output-dir", tmp])
        self.assertEqual(code, ExitCode.INPUT_OR_CONFIG_ERROR)

    def test_invalid_threshold_returns_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = main(["--log-level", "ERROR", "run", "--max-quarantine-rate", "2", "--output-dir", tmp])
        self.assertEqual(code, ExitCode.INPUT_OR_CONFIG_ERROR)

    def test_graph_only_flags_rejected_on_classic_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for flag in ("--require-approval", "--ai-summary"):
                code = main(["--log-level", "ERROR", "run", flag, "--output-dir", tmp])
                self.assertEqual(code, ExitCode.INPUT_OR_CONFIG_ERROR, flag)


if __name__ == "__main__":
    unittest.main()
