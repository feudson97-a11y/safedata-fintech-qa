import unittest

import numpy as np

from safedata.domain.exceptions import ConfigurationError
from safedata.domain.models import PipelineConfig
from safedata.domain.primitives import amount_to_cents, digits_only, format_cents, luhn_is_valid
from safedata.domain.schema import ALLOWED_MERCHANT_CATEGORIES
from safedata.synthetic.generator import MERCHANT_PROFILES, generate_card_number, luhn_check_digit


class TestPrimitives(unittest.TestCase):
    def test_luhn_known_values(self) -> None:
        self.assertTrue(luhn_is_valid("4111111111111111"))
        self.assertFalse(luhn_is_valid("4111111111111112"))
        self.assertFalse(luhn_is_valid("4111-1111"))
        self.assertEqual(luhn_check_digit("411111111111111"), "1")

    def test_generated_cards_are_luhn_valid(self) -> None:
        rng = np.random.default_rng(0)
        self.assertTrue(all(luhn_is_valid(generate_card_number(rng)) for _ in range(500)))

    def test_exact_decimal_cents(self) -> None:
        self.assertEqual(amount_to_cents("0.10"), 10)
        self.assertEqual(amount_to_cents("1234.5"), 123450)
        self.assertIsNone(amount_to_cents("12.3.4"))
        self.assertIsNone(amount_to_cents(None))
        self.assertEqual(format_cents(30), "0.30")  # where 0.1 + 0.2 in float would drift

    def test_digits_only(self) -> None:
        self.assertEqual(digits_only("+1 (415) 555-0123"), "14155550123")
        self.assertEqual(digits_only(None), "")


class TestContracts(unittest.TestCase):
    def test_generator_and_schema_share_one_category_vocabulary(self) -> None:
        self.assertEqual(set(MERCHANT_PROFILES), set(ALLOWED_MERCHANT_CATEGORIES))

    def test_config_rejects_invalid_thresholds(self) -> None:
        from pathlib import Path

        with self.assertRaises(ConfigurationError):
            PipelineConfig(Path("in.csv"), Path("out"), max_quarantine_rate=1.5)
        with self.assertRaises(ConfigurationError):
            PipelineConfig(Path("in.csv"), Path("out"), outlier_z_threshold=0)


if __name__ == "__main__":
    unittest.main()
