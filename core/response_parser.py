# -*- coding: utf-8 -*-
"""
CogniBench LLM Judge Response Parser Module.

This module provides functionality to parse and validate the raw text response
from a judge LLM, which is expected to contain evaluation results, typically
in a JSON format. It handles extraction of JSON from surrounding text (including
markdown fences) and validates the structure against expected criteria and
allowed score values.

Version: 0.3 (Phase 5 - Code Quality Enhancements)
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Setup logger for this module
logger = logging.getLogger("backend")

# --- Module Constants ---
# Note: The default constants EXPECTED_CRITERIA_FULL_L1 and ALLOWED_SCORES
# have been removed as per Enhancement Plan C1. These values should now
# be passed explicitly to the parse_judge_response function, typically
# loaded from the configuration.

# --- Type Aliases ---
# Represents the structure of a single criterion's evaluation data (score, justification, etc.)
CriterionEvaluation = Dict[str, Any]  # Allow for extra keys beyond score/justification

# Represents the main 'evaluation' object containing multiple criteria evaluations
EvaluationDict = Dict[str, CriterionEvaluation]

# Represents the final return type of the parser: either the successfully parsed
# data (containing the 'evaluation' dict) or an error dictionary.
ParsedResponse = Dict[
    str, Union[EvaluationDict, str, Any]
]  # Allow original structure return


def _normalize_key(key: str) -> str:
    """
    Normalizes a dictionary key string for case-insensitive and format-insensitive comparison.

    Normalization involves:
    1. Replacing spaces and underscores with an empty string.
    2. Converting the result to lowercase.

    Args:
        key: The original key string.

    Returns:
        The normalized key string.
    """
    if not isinstance(key, str):  # Basic type check
        return ""
    # Remove spaces and underscores, then lowercase
    return re.sub(r"[\s_]+", "", key).lower()


def _find_json_block(text: str) -> Optional[str]:
    """
    Extracts a JSON object string from a text block, handling common LLM outputs.

    Prioritizes JSON within markdown code fences (```json ... ``` or ``` ... ```).
    Falls back to finding the outermost curly braces `{...}`.
    Finally, checks if the entire stripped text is a JSON object.

    Args:
        text: The raw text potentially containing a JSON block.

    Returns:
        The extracted JSON string if found, otherwise None.
    """
    if not isinstance(text, str) or not text.strip():
        logger.debug("Cannot find JSON block in empty or non-string input.")
        return None

    # Pattern explanation:
    # ```      - Matches the opening fence
    # (?:json)? - Optionally matches 'json' (non-capturing group)
    # \s*      - Matches optional whitespace after 'json' or ```
    # (\{.*?\}) - Captures the JSON block (non-greedy match for content within {})
    # \s*      - Matches optional whitespace before the closing fence
    # ```      - Matches the closing fence
    # re.DOTALL allows '.' to match newlines. re.IGNORECASE handles 'json' casing.
    fence_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    fence_match = re.search(fence_pattern, text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        logger.debug("Found JSON block using markdown fence regex.")
        return fence_match.group(1).strip()  # Return the captured JSON part

    # Fallback 1: Find the first '{' and the last '}'
    # This is less reliable but catches cases without fences if JSON is the main part.
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        potential_json = text[first_brace : last_brace + 1].strip()
        # Basic check: Does it still look like a JSON object after stripping?
        if potential_json.startswith("{") and potential_json.endswith("}"):
            # Attempt a quick validation to avoid returning non-JSON brace pairs
            try:
                json.loads(potential_json)  # Try parsing
                logger.debug(
                    "Found potential JSON block using first/last brace fallback and basic validation."
                )
                return potential_json
            except json.JSONDecodeError:
                logger.debug("First/last brace content was not valid JSON.")
                # Continue to next fallback if this wasn't valid JSON

    # Fallback 2: Check if the entire stripped text is a JSON object
    stripped_text = text.strip()
    if stripped_text.startswith("{") and stripped_text.endswith("}"):
        try:
            json.loads(stripped_text)  # Validate it's actual JSON
            logger.debug(
                "Assuming the entire stripped text is the JSON block after validation."
            )
            return stripped_text
        except json.JSONDecodeError:
            logger.debug("Entire stripped text was not valid JSON.")

    logger.warning(
        "Could not reliably find and validate a JSON block in the provided text."
    )
    return None


def _parse_json_string(
    json_string: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Attempts to parse a JSON string, with a fallback to json5 for tolerant parsing.

    Args:
        json_string: The string suspected to contain JSON data.

    Returns:
        A tuple:
        - Parsed data as a dictionary if successful, otherwise None.
        - An error message string if parsing failed, otherwise None.
    """
    try:
        parsed_data = json.loads(json_string)
        logger.debug("Successfully parsed JSON block using standard json library.")
        return parsed_data, None
    except json.JSONDecodeError as e_std:
        logger.warning(
            "Standard JSON parsing failed (%s), attempting tolerant parsing with json5.",
            e_std,
        )
        try:
            import json5  # Import only when needed

            parsed_data = json5.loads(json_string)
            logger.info("Successfully parsed JSON block using json5 library.")
            return parsed_data, None
        except ImportError:
            error_msg = "Standard JSON parsing failed, and json5 library is not installed. Cannot perform tolerant parsing."
            logger.error(error_msg)
            return None, error_msg
        except Exception as e_json5:
            error_msg = (
                f"Invalid JSON format even with tolerant json5 parsing: {e_json5}"
            )
            logger.error(error_msg)
            return None, error_msg
    except Exception as e_other:
        error_msg = f"Unexpected error during JSON parsing: {e_other}"
        logger.error(error_msg, exc_info=True)
        return None, error_msg


def _validate_initial_structure(parsed_data: Any) -> Optional[str]:
    """
    Performs basic structural validation on the parsed JSON data.

    Checks if the data is a dictionary and contains the top-level 'evaluation' key,
    whose value must also be a dictionary.

    Args:
        parsed_data: The data parsed from the JSON string.

    Returns:
        An error message string if validation fails, otherwise None.
    """
    if not isinstance(parsed_data, dict):
        return "Parsed JSON is not a dictionary (object)."

    # Check for the mandatory 'evaluation' top-level key (case-sensitive)
    # Consider case-insensitive if needed: `evaluation_key = next((k for k in parsed_data if k.lower() == 'evaluation'), None)`
    if "evaluation" not in parsed_data:
        return "Parsed JSON is missing the required top-level key: 'evaluation'."

    evaluation_content = parsed_data["evaluation"]
    if not isinstance(evaluation_content, dict):
        return "The value associated with the 'evaluation' key is not a dictionary (object)."

    return None  # Structure is valid


def _validate_evaluation_criteria(
    evaluation_content: Dict[str, Any],
    expected_criteria: List[str],
    allowed_scores: List[str],
) -> Tuple[EvaluationDict, List[str]]:
    """
    Validates the content of the 'evaluation' dictionary against expected criteria and scores.

    Checks for presence of all expected criteria, validates the structure within each
    (presence of 'score' and 'justification'), and validates the score value.
    Uses case-insensitive matching for criteria names and score values.

    Args:
        evaluation_content: The dictionary found under the 'evaluation' key.
        expected_criteria: List of expected criterion names (case-insensitive match).
        allowed_scores: List of valid score values (case-insensitive match).

    Returns:
        A tuple containing:
        - validated_evaluation_output (EvaluationDict): A dictionary containing only the
          validated criteria, preserving original key casing from the input.
        - all_errors (List[str]): A list of all validation error messages found.
    """
    validated_evaluation_output: EvaluationDict = {}
    missing_criteria_errors: List[str] = []
    structural_errors_list: List[str] = []

    # Prepare for case-insensitive lookups
    normalized_input_map: Dict[str, Tuple[str, Any]] = {
        _normalize_key(k): (k, v)
        for k, v in evaluation_content.items()
        if isinstance(k, str)
    }
    if len(normalized_input_map) != len(evaluation_content):
        logger.warning(
            "Non-string keys found in evaluation content were ignored during validation."
        )

    normalized_expected_criteria_set: Set[str] = {
        _normalize_key(c) for c in expected_criteria
    }
    normalized_allowed_scores_set: Set[str] = {str(s).lower() for s in allowed_scores}

    # --- Iterate through EXPECTED criteria ---
    for norm_expected_criterion in normalized_expected_criteria_set:
        # Find original expected name for error messages
        original_expected_criterion_name = next(
            (
                c
                for c in expected_criteria
                if _normalize_key(c) == norm_expected_criterion
            ),
            norm_expected_criterion,  # Fallback
        )
        current_criterion_errors: List[str] = []

        # 1. Check Presence (using normalized map)
        if norm_expected_criterion not in normalized_input_map:
            missing_criteria_errors.append(f"'{original_expected_criterion_name}'")
            continue  # Skip further checks for this missing criterion

        original_key, criterion_value = normalized_input_map[norm_expected_criterion]

        # 2. Check Structure (must be a dictionary)
        if not isinstance(criterion_value, dict):
            current_criterion_errors.append(
                f"Value for criterion '{original_key}' is not a dictionary (found type: {type(criterion_value).__name__})."
            )
            structural_errors_list.extend(current_criterion_errors)
            continue  # Skip further checks for this malformed criterion

        # 3. Check for 'score' and 'justification' keys (case-insensitive)
        criterion_data_dict: Dict[str, Any] = criterion_value
        score_value: Optional[Any] = None
        justification_value: Optional[Any] = None  # Keep justification as Any for now
        score_original_key: Optional[str] = None
        justification_original_key: Optional[str] = None

        # Find keys case-insensitively
        for key, value in criterion_data_dict.items():
            if isinstance(key, str):
                norm_inner_key = key.lower()
                if norm_inner_key == "score":
                    score_value = value
                    score_original_key = key
                elif norm_inner_key == "justification":
                    justification_value = value
                    justification_original_key = key

        # Report missing mandatory keys
        if score_original_key is None:
            current_criterion_errors.append(
                f"Missing 'score' key within criterion '{original_key}'."
            )
        if justification_original_key is None:
            current_criterion_errors.append(
                f"Missing 'justification' key within criterion '{original_key}'."
            )

        # 4. Validate Score Value (if score key was found)
        if score_original_key is not None:
            if not isinstance(score_value, str):
                current_criterion_errors.append(
                    f"'score' value for criterion '{original_key}' is not a string (found type: {type(score_value).__name__})."
                )
            elif str(score_value).lower() not in normalized_allowed_scores_set:
                current_criterion_errors.append(
                    f"Invalid score value '{score_value}' for criterion '{original_key}'. Allowed (case-insensitive): {allowed_scores}"
                )

        # --- Store results or errors ---
        if not current_criterion_errors:
            # If valid, store the criterion data, preserving original keys
            validated_criterion_data: CriterionEvaluation = {}
            if score_original_key:
                validated_criterion_data[score_original_key] = score_value
            if justification_original_key:
                validated_criterion_data[justification_original_key] = (
                    justification_value
                )

            # Include any other keys the LLM might have provided
            for key, value in criterion_data_dict.items():
                if key != score_original_key and key != justification_original_key:
                    validated_criterion_data[key] = value

            validated_evaluation_output[original_key] = validated_criterion_data
        else:
            structural_errors_list.extend(current_criterion_errors)

    # --- Combine Errors ---
    all_errors = []
    if missing_criteria_errors:
        all_errors.append(
            f"Missing expected criteria: {', '.join(sorted(missing_criteria_errors))}"
        )
    if structural_errors_list:
        all_errors.extend(structural_errors_list)

    return validated_evaluation_output, all_errors


def parse_judge_response(
    raw_response_content: str,
    expected_criteria: List[str],
    allowed_scores: List[str],
) -> ParsedResponse:
    """
    Parses and validates the raw text response from a Judge LLM.

    Extracts JSON, parses it (with json5 fallback), validates the basic structure
    (must be dict with 'evaluation' key), and validates the content of the
    'evaluation' dictionary against expected criteria and allowed scores.

    Args:
        raw_response_content: Raw string output from the judge LLM.
        expected_criteria: List of criterion names expected within 'evaluation' (case-insensitive match).
        allowed_scores: List of valid score values for each criterion (case-insensitive match).

    Returns:
        A dictionary representing the parsing outcome:
        - On success: The original parsed dictionary, with the 'evaluation' key's value
          replaced by a validated version containing only the expected criteria
          that passed validation (preserving original key casing). Example:
          {'evaluation': {'Problem Understanding': {'score': 'Yes', ...}, ...}, 'overall_comment': '...'}
        - On failure: A dictionary {'error': message} describing the failure reason(s).
    """
    logger.debug("Attempting to parse judge response content.")

    # 1. Extract JSON block
    json_string = _find_json_block(raw_response_content)
    if not json_string:
        logger.warning("Parser: Could not find JSON block in the raw response.")
        return {"error": "Could not find JSON block in the response."}

    # 2. Parse JSON string (with json5 fallback)
    parsed_data, error_msg = _parse_json_string(json_string)
    if error_msg:
        return {"error": error_msg}  # Return parsing error

    # 3. Validate Initial Structure
    error_msg = _validate_initial_structure(parsed_data)
    if error_msg:
        logger.warning("Parser: Initial structure validation failed: %s", error_msg)
        return {"error": error_msg}

    # We now know parsed_data is a dict and has parsed_data['evaluation'] as a dict
    evaluation_content = parsed_data["evaluation"]

    # 4. Validate Evaluation Criteria Content
    logger.debug("Validating content within the 'evaluation' object...")
    validated_evaluation, validation_errors = _validate_evaluation_criteria(
        evaluation_content, expected_criteria, allowed_scores
    )

    if validation_errors:
        combined_error_msg = "Validation Failed: " + "; ".join(validation_errors) + "."
        logger.warning("Parser validation failed: %s", combined_error_msg)
        return {"error": combined_error_msg}

    # 5. Success: Replace original evaluation with validated content
    #    This ensures only expected and valid criteria are passed downstream.
    parsed_data["evaluation"] = validated_evaluation
    logger.info(
        "Parser validation successful."
    )  # Changed level to INFO for successful parse
    return parsed_data  # Return the full original structure but with validated 'evaluation'


# --- Example Usage & Basic Tests ---
if __name__ == "__main__":
    # Setup basic logging for testing if run directly
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Running response_parser module tests...")

    # --- Test Cases ---
    valid_json_string_fenced = """
Some introductory text from the LLM.
```json
{
    "evaluation": {
        "Problem Understanding": {
            "score": "Yes",
            "justification": "Model correctly identified the integral.",
            "confidence": 0.9
        },
        "Results Formulae": {
            "Score": "no",
            "Justification": "Final answer was incorrect."
        },
        "Assumptions": {
            "score": "Partial",
            "justification": "Some assumptions were missed."
        }
    },
    "overall_comment": "Good start, but calculation error."
}
```
Some concluding text.
"""
    valid_json_string_plain = """
    {
        "evaluation": {
            "problem understanding": {
                "score": "Yes",
                "justification": "Understood."
            },
            "Results Formulae": {
                "score": "No",
                "justification": "Wrong."
            }
        }
    }
    """
    invalid_json_string = (
        "Here is the evaluation: { evaluation: { Problem Understanding: ... }"
    )
    missing_crit_json = """{"evaluation": {"Problem Understanding": {"score": "Yes", "justification": "ok"}}}"""
    missing_score_key_json = (
        """{"evaluation": {"Problem Understanding": {"justification": "ok"}}}"""
    )
    invalid_score_value_json = """{"evaluation": {"Problem Understanding": {"score": "Maybe", "justification": "ok"}}}"""
    no_json_block = "The LLM failed to provide a JSON output."

    # Define expected criteria for this test run (subset)
    # Define test data (mimicking loading from config)
    test_expected = ["Problem Understanding", "Results Formulae", "Assumptions"]
    test_allowed_scores = ["Yes", "No", "Partial"]  # Use the common values for testing

    logger.info("\n--- Testing Valid JSON (Fenced) ---")
    parsed = parse_judge_response(
        raw_response_content=valid_json_string_fenced,
        expected_criteria=test_expected,
        allowed_scores=test_allowed_scores,
    )
    logger.info("Input:\n%s", valid_json_string_fenced)
    logger.info("Result:\n%s", json.dumps(parsed, indent=2))
    assert "error" not in parsed
    assert _normalize_key("Problem Understanding") in {
        _normalize_key(k) for k in parsed.get("evaluation", {})
    }
    assert parsed["evaluation"]["Problem Understanding"]["score"] == "Yes"
    assert (
        parsed["evaluation"]["Results Formulae"]["Score"] == "no"
    )  # Note: original key preserved
    assert parsed["evaluation"]["Assumptions"]["score"] == "Partial"

    logger.info("\n--- Testing Valid JSON (Plain, Missing Expected Criterion) ---")
    parsed = parse_judge_response(
        raw_response_content=valid_json_string_plain,
        expected_criteria=test_expected,
        allowed_scores=test_allowed_scores,
    )
    logger.info("Input:\n%s", valid_json_string_plain)
    logger.info("Result:\n%s", json.dumps(parsed, indent=2))
    # We expect an error because "Assumptions" is missing
    assert "error" in parsed
    assert "Missing expected criteria" in parsed["error"]
    assert "Assumptions" in parsed["error"]

    logger.info("\n--- Testing Invalid JSON (No JSON Block Found) ---")
    parsed = parse_judge_response(
        raw_response_content=invalid_json_string,
        expected_criteria=test_expected,
        allowed_scores=test_allowed_scores,
    )
    logger.info("Input:\n%s", invalid_json_string)
    logger.info("Result:\n%s", json.dumps(parsed, indent=2))
    assert "error" in parsed
    assert "Could not find JSON block" in parsed["error"]  # Updated expectation

    logger.info("\n--- Testing Missing Expected Criterion JSON ---")
    parsed = parse_judge_response(
        raw_response_content=missing_crit_json,
        expected_criteria=test_expected,
        allowed_scores=test_allowed_scores,
    )
    logger.info("Input:\n%s", missing_crit_json)
    logger.info("Result:\n%s", json.dumps(parsed, indent=2))
    assert "error" in parsed
    # Check if both missing criteria are mentioned in the combined error message
    assert "Missing expected criteria" in parsed["error"]
    assert "'Results Formulae'" in parsed["error"]
    assert "'Assumptions'" in parsed["error"]

    logger.info("\n--- Testing Missing 'score' Key JSON ---")
    # Test with a single expected criterion
    single_test_expected = ["Problem Understanding"]
    parsed = parse_judge_response(
        raw_response_content=missing_score_key_json,
        expected_criteria=single_test_expected,
        allowed_scores=test_allowed_scores,
    )
    logger.info("Input:\n%s", missing_score_key_json)
    logger.info("Result:\n%s", json.dumps(parsed, indent=2))
    assert "error" in parsed
    assert "Missing 'score' key" in parsed["error"]  # Check specific error message part

    logger.info("\n--- Testing Invalid 'score' Value JSON ---")
    parsed = parse_judge_response(
        raw_response_content=invalid_score_value_json,
        expected_criteria=single_test_expected,  # Use the single criterion list
        allowed_scores=test_allowed_scores,
    )
    logger.info("Input:\n%s", invalid_score_value_json)
    logger.info("Result:\n%s", json.dumps(parsed, indent=2))
    assert "error" in parsed
    assert (
        "Invalid score value 'Maybe'" in parsed["error"]
    )  # Check specific error message part

    logger.info("\n--- Testing Input with No JSON Block ---")
    parsed = parse_judge_response(
        raw_response_content=no_json_block,
        expected_criteria=test_expected,
        allowed_scores=test_allowed_scores,
    )
    logger.info("Input:\n%s", no_json_block)
    logger.info("Result:\n%s", json.dumps(parsed, indent=2))
    assert "error" in parsed
    assert "Could not find JSON block" in parsed["error"]

    logger.info("\n--- Testing Empty String Input ---")
    parsed = parse_judge_response(
        raw_response_content="",
        expected_criteria=test_expected,
        allowed_scores=test_allowed_scores,
    )
    logger.info("Input: ''")
    logger.info("Result:\n%s", json.dumps(parsed, indent=2))
    assert "error" in parsed
    assert "Could not find JSON block" in parsed["error"]

    logger.info("\n--- Testing Only Whitespace String Input ---")
    parsed = parse_judge_response(
        raw_response_content="   \n\t   ",
        expected_criteria=test_expected,
        allowed_scores=test_allowed_scores,
    )
    logger.info("Input: '   \\n\\t   '")
    logger.info("Result:\n%s", json.dumps(parsed, indent=2))
    assert "error" in parsed
    assert "Could not find JSON block" in parsed["error"]

    logger.info("\n--- All Response Parser Tests Passed (Implicitly) ---")
