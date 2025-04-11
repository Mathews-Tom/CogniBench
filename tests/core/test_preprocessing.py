import unittest

from CogniBench.core.preprocessing import (
    convert_math_notation,
    extract_final_answer,
    normalize_text_formats,
)


class TestPreprocessing(unittest.TestCase):
    def test_extract_final_answer(self):
        self.assertEqual(extract_final_answer("Final Answer: 42"), "42")
        self.assertEqual(extract_final_answer("Answer: x = 10"), "x = 10")
        self.assertEqual(extract_final_answer("No answer here"), None)

    def test_normalize_text_formats(self):
        self.assertEqual(
            normalize_text_formats("  Extra   spaces\nand\ttabs  "),
            "Extra spaces and tabs",
        )
        self.assertEqual(normalize_text_formats("Already clean."), "Already clean.")
        self.assertEqual(normalize_text_formats(None), None)

    def test_convert_math_notation(self):
        self.assertEqual(convert_math_notation(r"\(x + y\)"), "$x + y$")
        self.assertEqual(
            convert_math_notation(r"\[x^2 + y^2 = z^2\]"), "$$x^2 + y^2 = z^2$$"
        )
        self.assertEqual(
            convert_math_notation(r"Escaped \\( not math \\)"),
            r"Escaped \\( not math \\)",
        )
        self.assertEqual(convert_math_notation(None), None)


if __name__ == "__main__":
    unittest.main()
