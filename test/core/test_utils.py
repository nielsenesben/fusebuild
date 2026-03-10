import unittest

from fusebuild.core.utils import escape_whitespace as escape
from fusebuild.core.utils import unescape_whitespace as unescape


class TestWhitespaceEncoding(unittest.TestCase):

    def test_round_trip_simple(self) -> None:
        original = "hello world"
        self.assertEqual(unescape(escape(original)), original)

    def test_round_trip_all_whitespace(self) -> None:
        original = " \t\n\r\v\f"
        self.assertEqual(unescape(escape(original)), original)

    def test_round_trip_mixed_content(self) -> None:
        original = "Line 1\nLine 2\tEnd"
        self.assertEqual(unescape(escape(original)), original)

    def test_empty_string(self) -> None:
        original = ""
        self.assertEqual(unescape(escape(original)), original)

    def test_literal_percent_sequences(self) -> None:
        # Ensure already-percent-like sequences survive round-trip
        original = "100% safe %20 text"
        self.assertEqual(unescape(escape(original)), original)


if __name__ == "__main__":
    unittest.main()
