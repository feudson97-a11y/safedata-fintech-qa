"""ClaudeSummarizer adapter, exercised with a fake client (no network)."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from safedata.domain.exceptions import ConfigurationError
from safedata.infrastructure.llm import DEFAULT_MODEL, ClaudeSummarizer


class _FakeMessages:
    def __init__(self, text: str) -> None:
        self.text, self.calls = text, []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.text)])


class TestClaudeSummarizer(unittest.TestCase):
    def test_sends_report_and_returns_markdown(self) -> None:
        messages = _FakeMessages("## Verdict\nPublished.")
        summarizer = ClaudeSummarizer(model="m-test", client=SimpleNamespace(messages=messages))
        out = summarizer({"run": {"status": "PASSED"}})
        self.assertIn("## Verdict", out)
        self.assertIn("m-test", out)
        self.assertEqual(messages.calls[0]["model"], "m-test")
        self.assertIn('"PASSED"', messages.calls[0]["messages"][0]["content"])

    def test_empty_answer_is_an_error(self) -> None:
        summarizer = ClaudeSummarizer(client=SimpleNamespace(messages=_FakeMessages("  ")))
        with self.assertRaises(ValueError):
            summarizer({})

    def test_model_resolution(self) -> None:
        with mock.patch.dict(os.environ, {"SAFEDATA_SUMMARY_MODEL": "env-model"}):
            self.assertEqual(ClaudeSummarizer().model, "env-model")
            self.assertEqual(ClaudeSummarizer(model="cli-model").model, "cli-model")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ClaudeSummarizer().model, DEFAULT_MODEL)

    def test_missing_api_key_fails_fast(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(ConfigurationError):
            ClaudeSummarizer().ensure_ready()


if __name__ == "__main__":
    unittest.main()
