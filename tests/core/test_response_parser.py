# CogniBench/tests/core/test_response_parser.py

import pytest

from CogniBench.core.response_parser import parse_judge_response

# Explicitly define allowed scores and expected criteria for tests
ALLOWED_SCORES = ["Yes", "No", "Partial"]
EXPECTED_CRITERIA_FULL_L1 = [
    "Problem Understanding",
    "Assumptions",
    "Logical Implications",
    "Results Formulae",
    "Rigor and Completeness",
]

# --- Test Data ---

VALID_JSON_FENCED = """
Some text before.
```json
{
    "evaluation": {
        "Problem Understanding": {
            "score": "Yes",
            "justification": "Understood."
        },
        "Assumptions": {
            "Score": "NO",
            "Justification": "Missed key assumption."
        },
        "Logical Implications": {
            "score": "Partial",
            "justification": "One step was shaky."
        },
        "Results Formulae": {
            "score": "yes",
            "justification": "Correct final answer."
        },
        "Rigor and Completeness": {
            "score": "No",
            "justification": "Lacked detail."
        }
    },
    "extra_info": "Some comment"
}
```
Some text after.
"""

VALID_JSON_PLAIN = """
{
    "evaluation": {
        "problem understanding": {
            "score": "Yes",
            "justification": "OK"
        },
        "assumptions": {
            "score": "No",
            "justification": "Bad assumption"
        },
        "logical implications": {
            "score": "Yes",
            "justification": "Fine"
        },
        "Results Formulae": {
            "score": "Yes",
            "justification": "Match"
        },
        "Rigor and Completeness": {
            "score": "Partial",
            "justification": "Needed more steps"
        }
    }
}
"""

INVALID_JSON_STRING = "This is not JSON { evaluation: ... }"
MISSING_EVAL_KEY = """{"other_key": "value"}"""
EVAL_NOT_DICT = """{"evaluation": ["list", "not", "dict"]}"""
MISSING_CRITERION = """
{
    "evaluation": {
        "Problem Understanding": {"score": "Yes", "justification": "OK"},
        "Assumptions": {"score": "No", "justification": "Bad"},
        "Logical Implications": {"score": "Yes", "justification": "Fine"},
        "Results Formulae": {"score": "Yes", "justification": "Match"}
        // Missing Rigor and Completeness
    }
}"""
CRITERION_NOT_DICT = """
{
    "evaluation": {
        "Problem Understanding": "Yes",
        "Assumptions": {"score": "No", "justification": "Bad"}
    }
}"""
MISSING_SCORE_KEY = """
{
    "evaluation": {
        "Problem Understanding": {"justification": "OK"}
    }
}"""
MISSING_JUSTIFICATION_KEY = """
{
    "evaluation": {
        "Problem Understanding": {"score": "Yes"}
    }
}"""
INVALID_SCORE_VALUE = """
{
    "evaluation": {
        "Problem Understanding": {"score": "Maybe", "justification": "OK"}
    }
}"""
NO_JSON_BLOCK = "The LLM only returned this text."
EMPTY_STRING = ""
WHITESPACE_STRING = "   \n\t   "

# --- Test Cases ---


def test_parse_valid_json_fenced():
    """Test parsing valid JSON enclosed in markdown fences."""
    result = parse_judge_response(VALID_JSON_FENCED)
    assert "error" not in result
    assert "evaluation" in result
    assert len(result["evaluation"]) == 5  # All expected criteria found
    # Check one criterion's details (case-insensitivity of keys handled internally)
    assert result["evaluation"]["Assumptions"]["Score"] == "NO"
    assert (
        result["evaluation"]["Assumptions"]["Justification"] == "Missed key assumption."
    )
    assert result["extra_info"] == "Some comment"  # Extra top-level keys preserved


def test_parse_valid_json_plain():
    """Test parsing valid plain JSON with normalized keys."""
    result = parse_judge_response(VALID_JSON_PLAIN)
    assert "error" not in result
    assert "evaluation" in result
    assert len(result["evaluation"]) == 5
    # Check original keys are preserved in output
    assert "problem understanding" in result["evaluation"]
    assert result["evaluation"]["problem understanding"]["score"] == "Yes"
    assert result["evaluation"]["Rigor and Completeness"]["score"] == "Partial"


def test_parse_invalid_json_string():
    """Test handling of completely invalid JSON."""
    result = parse_judge_response(INVALID_JSON_STRING)
    assert "error" in result
    # It finds the braces, so it tries to parse, leading to JSONDecodeError
    assert "Invalid JSON format" in result["error"]


def test_parse_missing_eval_key():
    """Test error when the top-level 'evaluation' key is missing."""
    result = parse_judge_response(MISSING_EVAL_KEY)
    assert "error" in result
    assert "Missing required top-level key: 'evaluation'" in result["error"]


def test_parse_eval_not_dict():
    """Test error when 'evaluation' value is not a dictionary."""
    result = parse_judge_response(EVAL_NOT_DICT)
    assert "error" in result
    assert "'evaluation' value is not a JSON object" in result["error"]


def test_parse_missing_criterion():
    """Test error when an expected criterion is missing."""
    # Need to handle potential JSON parsing error due to comment
    clean_missing_criterion = """
    {
        "evaluation": {
            "Problem Understanding": {"score": "Yes", "justification": "OK"},
            "Assumptions": {"score": "No", "justification": "Bad"},
            "Logical Implications": {"score": "Yes", "justification": "Fine"},
            "Results Formulae": {"score": "Yes", "justification": "Match"}
        }
    }"""
    result = parse_judge_response(clean_missing_criterion)
    assert "error" in result
    assert "Missing expected criteria" in result["error"]
    assert "Rigor and Completeness" in result["error"]


def test_parse_criterion_not_dict():
    """Test error when a criterion's value is not a dictionary."""
    # Ensure the input contains all expected criteria for this test
    input_json = """
    {
        "evaluation": {
            "Problem Understanding": "Yes",
            "Assumptions": {"score": "No", "justification": "Bad"}
        }
    }"""
    result = parse_judge_response(
        input_json, expected_criteria=["Problem Understanding", "Assumptions"]
    )
    assert "error" in result
    assert (
        "Value for criterion 'Problem Understanding' is not a JSON object"
        in result["error"]
    )


def test_parse_missing_score_key():
    """Test error when 'score' key is missing within a criterion."""
    result = parse_judge_response(
        MISSING_SCORE_KEY, expected_criteria=["Problem Understanding"]
    )
    assert "error" in result
    assert (
        "Missing 'score' key for criterion 'Problem Understanding'" in result["error"]
    )


def test_parse_missing_justification_key():
    """Test error when 'justification' key is missing within a criterion."""
    result = parse_judge_response(
        MISSING_JUSTIFICATION_KEY, expected_criteria=["Problem Understanding"]
    )
    assert "error" in result
    assert (
        "Missing 'justification' key for criterion 'Problem Understanding'"
        in result["error"]
    )


def test_parse_invalid_score_value():
    """Test error when 'score' has an disallowed value."""
    result = parse_judge_response(
        INVALID_SCORE_VALUE, expected_criteria=["Problem Understanding"]
    )
    assert "error" in result
    assert "Invalid score value 'Maybe'" in result["error"]
    assert str(ALLOWED_SCORES) in result["error"]


def test_parse_no_json_block():
    """Test handling when no JSON block is found."""
    result = parse_judge_response(NO_JSON_BLOCK)
    assert "error" in result
    assert "Could not find JSON block" in result["error"]


def test_parse_empty_string():
    """Test handling of empty input string."""
    result = parse_judge_response(EMPTY_STRING)
    assert "error" in result
    assert "Could not find JSON block" in result["error"]


def test_parse_whitespace_string():
    """Test handling of input string with only whitespace."""
    result = parse_judge_response(WHITESPACE_STRING)
    assert "error" in result
    assert "Could not find JSON block" in result["error"]


def test_parse_custom_expected_criteria():
    """Test using a subset of expected criteria."""
    custom_criteria = ["Problem Understanding", "Results Formulae"]
    result = parse_judge_response(VALID_JSON_FENCED, expected_criteria=custom_criteria)
    assert "error" not in result
    assert "evaluation" in result
    # The output evaluation dict should only contain the keys that were expected and validated
    assert len(result["evaluation"]) == 2
    assert "Problem Understanding" in result["evaluation"]
    assert "Results Formulae" in result["evaluation"]


def test_parse_custom_allowed_scores():
    """Test using custom allowed scores."""
    custom_scores = ["Pass", "Fail"]
    json_with_custom_score = """
    {
        "evaluation": {
            "Problem Understanding": {"score": "Pass", "justification": "It passed."}
        }
    }"""
    result = parse_judge_response(
        json_with_custom_score,
        expected_criteria=["Problem Understanding"],
        allowed_scores=custom_scores,
    )
    assert "error" not in result
    assert result["evaluation"]["Problem Understanding"]["score"] == "Pass"

    # Test failure with default scores
    result_fail = parse_judge_response(
        json_with_custom_score, expected_criteria=["Problem Understanding"]
    )
    assert "error" in result_fail
    assert "Invalid score value 'Pass'" in result_fail["error"]
    # Test failure with default scores
    result_fail = parse_judge_response(
        json_with_custom_score, expected_criteria=["Problem Understanding"]
    )
    assert "error" in result_fail
    assert "Invalid score value 'Pass'" in result_fail["error"]
