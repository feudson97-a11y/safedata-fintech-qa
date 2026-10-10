import unittest

from safedata.domain.exceptions import ConfigurationError
from safedata.domain.schema import PiiEntity
from safedata.privacy.detection import NameLexicon
from safedata.privacy.masking import mask_card, mask_email, mask_phone, redact_free_text
from safedata.privacy.patterns import REDACTED_CARD, REDACTED_EMAIL
from safedata.privacy.tokenization import Tokenizer

TEST_KEY = b"unit-test-key-0123456789abcdef"


class TestStructuredMasking(unittest.TestCase):
    def test_mask_email(self) -> None:
        self.assertEqual(mask_email("john.doe@email.com"), "j***@email.com")
        self.assertEqual(mask_email("john.doe.email.com"), REDACTED_EMAIL)
        self.assertEqual(mask_email(""), "")

    def test_mask_phone_and_card(self) -> None:
        self.assertEqual(mask_phone("+1 (415) 555-0123"), "***-***-0123")
        self.assertEqual(mask_card("4111 1111 1111 1111"), "****-****-****-1111")
        self.assertEqual(mask_card("1234"), REDACTED_CARD)


class TestTokenizer(unittest.TestCase):
    def test_tokens_are_deterministic_and_keyed(self) -> None:
        a, b = Tokenizer(TEST_KEY), Tokenizer(b"a-completely-different-key-9999")
        self.assertEqual(a.token("4111111111111111", "CARD"), a.token(" 4111111111111111 ", "CARD"))
        self.assertNotEqual(a.token("4111111111111111", "CARD"), b.token("4111111111111111", "CARD"))
        self.assertRegex(a.token("Jane Doe", "CUST"), r"^CUST_[0-9a-f]{16}$")

    def test_short_key_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            Tokenizer(b"too-short")


class TestFreeTextRedaction(unittest.TestCase):
    lexicon = NameLexicon(frozenset({"Priya", "James"}), frozenset({"Patel", "O'Brien"}))

    def test_all_entity_types_redacted(self) -> None:
        note = "Priya Patel (priya.patel@gmail.com) called from (415) 555-0123 about card 4111-1111-1111-1111."
        result = redact_free_text(note, self.lexicon, own_name="Priya Patel")
        self.assertEqual(result.entities, frozenset(PiiEntity))
        for secret in ("Priya", "Patel", "priya.patel@", "555-0123", "4111"):
            self.assertNotIn(secret, result.text)
        self.assertIn("p***@gmail.com", result.text)

    def test_honorific_and_first_name_only(self) -> None:
        self.assertNotIn("O'Brien", redact_free_text("Ms. O'Brien requested a callback.", self.lexicon).text)
        self.assertNotIn("James", redact_free_text("Card shipped to James.", self.lexicon, "James Smith").text)

    def test_clean_text_untouched(self) -> None:
        note = "Card-not-present transaction; 3DS authentication passed."
        self.assertEqual(redact_free_text(note, self.lexicon).text, note)


if __name__ == "__main__":
    unittest.main()
