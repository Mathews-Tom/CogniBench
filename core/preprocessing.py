# -*- coding: utf-8 -*-
"""
CogniBench Preprocessing Module.

Provides functions for cleaning and preparing text data before evaluation,
including extracting potential final answers from model responses and
normalizing text formats.

Version: 0.2 (Phase 5 - Code Quality Enhancements)
"""

import json
import logging
import re
import unicodedata
from typing import Optional

# Setup logger for this module
logger = logging.getLogger("backend")


def safe_json_parse(json_string: str) -> Optional[dict]:
    """
    Safely parse a JSON string, handling empty or malformed inputs gracefully.

    Args:
        json_string (str): The JSON string to parse.

    Returns:
        Optional[dict]: Parsed JSON object if successful, None otherwise.
    """
    try:
        if not json_string.strip():
            raise ValueError("Empty JSON string")
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        logger.warning(
            f"JSON parsing failed: {e.msg} at line {e.lineno} column {e.colno}"
        )
        return None
    except ValueError as e:
        logger.warning(f"JSON parsing failed: {str(e)}")
        return None


def extract_structured_response(response_text: str) -> Optional[str]:
    """
    Attempts to parse structured JSON response and extract the final answer.

    Args:
        response_text: The structured JSON response text.

    Returns:
        The extracted final answer if JSON parsing is successful, otherwise None.
    """
    structured_data = safe_json_parse(response_text)
    if structured_data:
        final_answer = structured_data.get("final_answer", None)
        if final_answer:
            logger.info("Successfully extracted final answer from structured JSON.")
            return final_answer.strip()
    return None


def normalize_text_formats(text: Optional[str]) -> Optional[str]:
    """
    Performs basic text normalization on a given string.

    Steps include:
    1.  Unicode Normalization (NFC form).
    2.  Replacing sequences of multiple whitespace characters (spaces, tabs,
        newlines, etc.) with a single space.
    3.  Stripping leading and trailing whitespace from the result.

    Note:
        This function currently does *not* handle specific markup like LaTeX or
        MathML. Such normalization would require dedicated libraries (e.g., SymPy)
        and should be implemented as part of the enhancement plan if needed,
        likely during the answer verification stage in postprocessing rather
        than general text normalization here.

    Args:
        text: The input string to normalize, or None.

    Returns:
        The normalized string, or None if the input `text` was None.
    """
    if text is None:
        return None

    # 1. Unicode normalization (NFC is common, combines characters and accents)
    normalized_text: str = unicodedata.normalize("NFC", text)

    # 2. Replace multiple whitespace characters (space, tab, newline, etc.) with a single space
    whitespace_normalized_text: str = re.sub(r"\s+", " ", normalized_text)

    # 3. Strip leading/trailing whitespace
    stripped_text: str = whitespace_normalized_text.strip()

    return stripped_text


if __name__ == "__main__":
    # Setup basic logging for testing if run directly
    logging.basicConfig(level=logging.INFO)
    # Removed test cases for extract_final_answer as the function is removed.
    logger.info("\n--- Testing: Text Normalization ---")
    print("\n--- Text Normalization ---")
    test_norm_1 = "  Extra   spaces\nand\ttabs  "
    test_norm_2 = "Already clean."
    test_norm_3 = None
    print(f"'{test_norm_1}' -> '{normalize_text_formats(test_norm_1)}'")
    print(f"'{test_norm_2}' -> '{normalize_text_formats(test_norm_2)}'")
    print(f"'{test_norm_3}' -> '{normalize_text_formats(test_norm_3)}'")
    print(f"'{test_norm_3}' -> '{normalize_text_formats(test_norm_3)}'")
