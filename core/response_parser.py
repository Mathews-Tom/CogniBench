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
logger = logging.getLogger(__name__)

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
    Attempts to extract a JSON object string from a larger text block.

    Handles JSON enclosed in markdown code fences (e.g., ```json ... ``` or ``` ... ```)
    as often produced by LLMs. Also includes fallbacks for finding the outermost
    curly braces or checking if the entire stripped text is a JSON object.

    Args:
        text: The raw text potentially containing a JSON block.

    Returns:
        The extracted JSON string if found, otherwise None.
    """
    # Regex to find JSON possibly enclosed in ```json ... ``` or ``` ... ```
    if not isinstance(text, str) or not text.strip():
        logger.debug("Cannot find JSON block in empty or non-string input.")
        return None

    # 1. Try finding JSON within markdown code fences (common LLM output format)
    #    Handles ```json { ... } ``` or ``` { ... } ```
    #    Uses non-greedy matching for the content {.*?}
    fence_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE
    )
    if fence_match:
        logger.debug("Found JSON block using markdown fence regex.")
        return fence_match.group(1).strip()

    # 2. Fallback: Find the first '{' and the last '}' in the text.
    #    This is less robust but can work if the JSON is the main content.
    json_start_index = text.find("{")
    json_end_index = text.rfind("}")
    if json_start_index != -1 and json_end_index > json_start_index:
        potential_json = text[json_start_index : json_end_index + 1]
        # Basic sanity check: does it look like JSON? (Starts/ends with braces)
        if potential_json.strip().startswith("{") and potential_json.strip().endswith(
            "}"
        ):
            logger.debug(
                "Found potential JSON block using first '{' and last '}' fallback."
            )
            return potential_json.strip()

    # 3. Fallback: Check if the entire stripped text is a JSON object.
    stripped_text = text.strip()
    if stripped_text.startswith("{") and stripped_text.endswith("}"):
        logger.debug("Assuming the entire stripped text is the JSON block.")
        return stripped_text

    logger.warning("Could not reliably find a JSON block in the provided text.")
    return None


def parse_judge_response(
    raw_response_content: str,
    expected_criteria: List[str],
    allowed_scores: List[str],
) -> ParsedResponse:
    """Parses and validates the raw text response from a Judge LLM.

    This function attempts to extract a JSON block from the raw_response_content,
    parse it, and then validate its structure and content based on the provided
    expected_criteria and allowed_scores.

    Steps:
        1. Extracts a potential JSON string using _find_json_block.
        2. Attempts to parse the extracted string into a Python dictionary using json.loads.
        3. Validates the presence of a top-level 'evaluation' key containing a dictionary.
        4. Normalizes keys for comparison (both from the input and expected criteria).
        5. Checks if all expected_criteria are present within the 'evaluation' dictionary.
        6. For each present and expected criterion, validates:
            - It contains 'score' and 'justification' keys (case-insensitive check).
            - The 'score' value is one of the allowed_scores (case-insensitive check).
        7. Reports *all* validation errors encountered (missing criteria,
            structural errors like missing keys or wrong types, invalid score values)
            combined into a single error message.

    Args:
        raw_response_content: The raw string output received from the judge LLM.
        expected_criteria: A list of criterion names (strings) that are expected
            to be present as keys within the 'evaluation' dictionary. Case is
            preserved for error messages, but matching is case-insensitive.
        allowed_scores: A list of score values (strings) that are considered valid
            for the 'score' field within each criterion. Matching is case-insensitive.

    Returns:
        A dictionary representing the parsing outcome:
        - On success: Returns the original parsed dictionary, potentially with
            additional keys if the LLM provided them, but guarantees the presence
            and basic validation of the 'evaluation' structure according to the
            expected criteria. Example:
            {'evaluation': {'Problem Understanding': {'score': 'Yes', ...}, ...}, 'overall_comment': '...'}
        - On failure: Returns a dictionary containing a single 'error' key with a
            string describing the reason(s) for failure. Example:
            {'error': "Validation Failed: Missing expected criteria: ['Assumptions']; Invalid score value 'Maybe' for criterion 'Problem Understanding'. Allowed: ['Yes', 'No', 'Partial']"}
    """
    logger.debug("Attempting to parse judge response content.")
    json_string: Optional[str] = _find_json_block(raw_response_content)

    if not json_string:
        logger.warning("Parser: Could not find JSON block in the raw response.")
        return {"error": "Could not find JSON block in the response."}

    try:
        # Attempt to parse the extracted JSON string
        parsed_data: Any = json.loads(json_string)
        logger.debug("Successfully parsed JSON block.")
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON format in extracted block: {e}"
        logger.warning(error_msg, exc_info=False)  # Log basic error
        # Optionally log the problematic JSON string for debugging (can be verbose)
        # logger.debug("Problematic JSON string:\n---\n%s\n---", json_string)
        return {"error": error_msg}
    except Exception as e:  # Catch other unexpected errors during parsing
        error_msg = f"Unexpected error during JSON parsing: {e}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}

    # --- Initial Structure Validation ---
    logger.debug("Validating basic JSON structure...")
    if not isinstance(parsed_data, dict):
        msg = "Parsed JSON is not a dictionary (object)."
        logger.warning(msg)
        return {"error": msg}

    # Check for the mandatory 'evaluation' top-level key (case-sensitive for now)
    if "evaluation" not in parsed_data:
        # TODO: Consider allowing case-insensitive check for 'evaluation' key?
        msg = "Parsed JSON is missing the required top-level key: 'evaluation'."
        logger.warning(msg)
        return {"error": msg}

    evaluation_content: Any = parsed_data["evaluation"]
    if not isinstance(evaluation_content, dict):
        msg = "The value associated with the 'evaluation' key is not a dictionary (object)."
        logger.warning(msg)
        return {"error": msg}

    # --- Detailed Validation of 'evaluation' Content ---
    logger.debug("Validating content within the 'evaluation' object...")

    # Create a map from normalized input keys to (original_key, original_value)
    # This handles case/format variations in the LLM's output keys.
    normalized_input_map: Dict[str, Tuple[str, Any]] = {}
    for k, v in evaluation_content.items():
        if isinstance(k, str):  # Ensure keys are strings before normalizing
            normalized_input_map[_normalize_key(k)] = (k, v)
        else:
            logger.warning(
                "Non-string key found in evaluation content: %s. Skipping.", k
            )

    # Prepare normalized sets/lists for efficient lookup
    normalized_expected_criteria_set: Set[str] = {
        _normalize_key(c) for c in expected_criteria
    }
    normalized_allowed_scores_list: List[str] = [str(s).lower() for s in allowed_scores]

    # This will store the validated criteria, preserving original keys
    validated_evaluation_output: EvaluationDict = {}

    # Lists to collect validation errors
    missing_criteria_errors: List[str] = []
    structural_errors_list: List[str] = []

    # Iterate through the *expected* criteria to ensure they are present
    for norm_expected_criterion in normalized_expected_criteria_set:
        # Find the original casing of the expected criterion for error messages
        original_expected_criterion_name: str = next(
            (
                c
                for c in expected_criteria
                if _normalize_key(c) == norm_expected_criterion
            ),
            norm_expected_criterion,  # Fallback to normalized if not found (shouldn't happen)
        )
        current_criterion_errors: List[str] = []  # Errors specific to this criterion

        # --- 1. Check Presence ---
        if norm_expected_criterion not in normalized_input_map:
            missing_criteria_errors.append(f"'{original_expected_criterion_name}'")
            continue  # Skip further checks for this missing criterion

        # --- 2. Check Structure if Present ---
        original_key, criterion_value = normalized_input_map[norm_expected_criterion]

        if not isinstance(criterion_value, dict):
            current_criterion_errors.append(
                f"Value for criterion '{original_key}' is not a dictionary (object)."
            )
            # Add errors for this criterion to the main list and skip further checks for it
            structural_errors_list.extend(current_criterion_errors)
            continue

        # --- 3. Check for 'score' and 'justification' keys (case-insensitive) ---
        criterion_data_dict: Dict[str, Any] = criterion_value  # Now known to be a dict
        score_value: Optional[Any] = None
        justification_value: Optional[Any] = None
        score_original_key: Optional[str] = None
        justification_original_key: Optional[str] = None

        # Find score/justification keys case-insensitively
        for key, value in criterion_data_dict.items():
            if not isinstance(key, str):
                continue  # Skip non-string keys
            norm_inner_key = key.lower()
            if norm_inner_key == "score":
                score_value = value
                score_original_key = key
            elif norm_inner_key == "justification":
                justification_value = value
                justification_original_key = key

        # Check if mandatory keys were found
        if score_original_key is None:
            current_criterion_errors.append(
                f"Missing 'score' key within criterion '{original_key}'."
            )
        if justification_original_key is None:
            current_criterion_errors.append(
                f"Missing 'justification' key within criterion '{original_key}'."
            )

        # --- 4. Validate Score Value (if score key was found) ---
        if score_original_key is not None:
            if not isinstance(score_value, str):
                current_criterion_errors.append(
                    f"'score' value for criterion '{original_key}' is not a string (found type: {type(score_value).__name__})."
                )
            # Check against allowed scores (case-insensitive)
            elif score_value.lower() not in normalized_allowed_scores_list:
                current_criterion_errors.append(
                    f"Invalid score value '{score_value}' for criterion '{original_key}'. Allowed (case-insensitive): {allowed_scores}"
                )
        # Note: Justification value type is not strictly validated here, treated as string.

        # --- Store results or errors for this criterion ---
        if not current_criterion_errors:
            # If valid, store the criterion data (preserving original keys)
            # Ensure score/justification keys exist before accessing
            validated_criterion_data: CriterionEvaluation = {}
            if score_original_key:
                validated_criterion_data[score_original_key] = score_value
            if justification_original_key:
                validated_criterion_data[justification_original_key] = (
                    justification_value
                )

            # Include any other keys the LLM might have provided under this criterion
            for key, value in criterion_data_dict.items():
                if key != score_original_key and key != justification_original_key:
                    validated_criterion_data[key] = value

            validated_evaluation_output[original_key] = validated_criterion_data
        else:
            # If errors found for this criterion, add them to the main structural error list
            structural_errors_list.extend(current_criterion_errors)

    # --- Final Error Reporting ---
    # Combine all collected errors
    all_errors = []
    if missing_criteria_errors:
        all_errors.append(
            f"Missing expected criteria: {', '.join(missing_criteria_errors)}"
        )
    if structural_errors_list:
        # Combine all structural/value errors found for individual criteria
        all_errors.extend(structural_errors_list)

    if all_errors:
        # Format a single error message containing all issues
        combined_error_msg = "Validation Failed: " + "; ".join(all_errors) + "."
        logger.warning("Parser validation failed: %s", combined_error_msg)
        return {"error": combined_error_msg}

    # --- Success ---
    # If no errors were found, replace the original evaluation content with the
    # validated structure (which preserves original keys but ensures required ones are present).
    parsed_data["evaluation"] = validated_evaluation_output
    logger.debug("Parser validation successful.")
    return parsed_data  # Return the full original parsed structure with validated 'evaluation'


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
