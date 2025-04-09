# CogniBench - LLM Judge Response Parser
# Version: 0.2 (Phase 4 - Robustness Enhancements)

import json
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

# Define the expected keys for the full L1 rubric evaluation (Phase 2)
# Keep original casing for reference and potential use elsewhere
EXPECTED_CRITERIA_FULL_L1 = [
    "Problem Understanding",
    "Assumptions",
    "Logical Implications",
    "Results/Formulae",
    "Rigor and Completeness",
]

# Define allowed score values (case-insensitive comparison will be used)
ALLOWED_SCORES = ["Yes", "No", "Partial"]  # Added "Partial" as a possibility

# Type alias for the parsed evaluation structure
EvaluationDict = Dict[str, Dict[str, str]]
ParsedResponse = Dict[str, Union[EvaluationDict, str]]  # Can be evaluation or error


def _normalize_key(key: str) -> str:
    """Normalizes a key for comparison (lowercase, replace space/underscore with '')."""
    return re.sub(r"[\s_]+", "", key).lower()


def _find_json_block(text: str) -> Optional[str]:
    """Attempts to extract a JSON block, handling optional markdown fences."""
    # Regex to find JSON possibly enclosed in ```json ... ``` or ``` ... ```
    match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1)

    # Fallback: find first '{' and last '}'
    json_start = text.find("{")
    json_end = text.rfind("}")
    if json_start != -1 and json_end != -1 and json_end > json_start:
        return text[json_start : json_end + 1]

    # If it looks like JSON already (starts with { ends with })
    stripped_text = text.strip()
    if stripped_text.startswith("{") and stripped_text.endswith("}"):
        return stripped_text

    return None  # Could not reliably find a JSON block


def parse_judge_response(
    raw_response_content: str,
    expected_criteria: List[str] = EXPECTED_CRITERIA_FULL_L1,
    allowed_scores: List[str] = ALLOWED_SCORES,
) -> ParsedResponse:
    """
    Parses the raw string response from the Judge LLM, expecting JSON format.
    Handles potential markdown fences, normalizes keys, validates structure
    and score values based on expected criteria.

    Args:
        raw_response_content: The raw string output from the LLM.
        expected_criteria: A list of top-level keys expected under the 'evaluation' object
                           (original casing). Normalization is applied for matching.
        allowed_scores: A list of allowed values for the 'score' field (case-insensitive).

    Returns:
        A dictionary containing the parsed and validated JSON data if successful,
        or a dictionary with an 'error' key if parsing or validation fails.
        Example success: {"evaluation": {"Problem Understanding": {"score": "Yes", ...}, ...}}
        Example error: {"error": "Invalid JSON format: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"}
        Example error: {"error": "Missing expected criterion in 'evaluation': 'Problem Understanding'"}
        Example error: {"error": "Invalid score value 'Maybe' for criterion 'Problem Understanding'. Allowed: ['Yes', 'No', 'Partial']"}
    """
    json_string = _find_json_block(raw_response_content)

    if not json_string:
        return {"error": "Could not find JSON block in the response."}

    try:
        parsed_data: Any = json.loads(json_string)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON format: {e}"
        # print(f"JSON Parsing Error: {e}")
        # print(f"Attempted to parse JSON string:\n---\n{json_string}\n---")
        # print(f"Original raw content was:\n---\n{raw_response_content}\n---")
        return {"error": error_msg}
    except Exception as e:  # Catch other potential issues during loading
        # print(f"An unexpected error occurred during JSON loading: {e}")
        return {"error": f"Unexpected JSON loading error: {e}"}

    # --- Validation ---
    if not isinstance(parsed_data, dict):
        return {"error": "Parsed data is not a JSON object (dictionary)."}

    if "evaluation" not in parsed_data:
        # Try normalizing top-level keys as well? Maybe too lenient.
        # For now, require 'evaluation' literally.
        return {"error": "Missing required top-level key: 'evaluation'."}

    evaluation_content = parsed_data["evaluation"]
    if not isinstance(evaluation_content, dict):
        return {"error": "'evaluation' value is not a JSON object (dictionary)."}

    # Normalize ALL keys from the input evaluation content first
    # Store mapping from normalized key back to original key and original value
    normalized_input_map: Dict[str, Tuple[str, Any]] = {
        _normalize_key(k): (k, v) for k, v in evaluation_content.items()
    }

    # Normalize expected criteria keys for comparison
    normalized_expected_criteria = {_normalize_key(c) for c in expected_criteria}
    normalized_allowed_scores = [s.lower() for s in allowed_scores]

    validated_evaluation: EvaluationDict = {}

    # --- Refined Validation Logic ---
    missing_criteria = []
    structural_errors = []
    current_criterion_structural_errors = []  # Track errors for the current criterion

    for norm_expected_criterion in normalized_expected_criteria:
        original_expected_criterion_name = next(
            (
                c
                for c in expected_criteria
                if _normalize_key(c) == norm_expected_criterion
            ),
            norm_expected_criterion,
        )  # Get original name for error messages
        current_criterion_structural_errors.clear()  # Reset for each criterion

        # 1. Check if the expected criterion exists in the input
        if norm_expected_criterion not in normalized_input_map:
            missing_criteria.append(original_expected_criterion_name)
            continue  # Move to the next expected criterion

        # 2. If it exists, check its structure
        original_key, original_value = normalized_input_map[norm_expected_criterion]

        if not isinstance(original_value, dict):
            current_criterion_structural_errors.append(
                f"Value for criterion '{original_key}' is not a JSON object."
            )
            structural_errors.extend(
                current_criterion_structural_errors
            )  # Add to main error list
            continue  # Cannot check score/justification if not a dict

        # Check for score and justification within the criterion dict
        criterion_data = original_value
        score = None
        justification = None
        score_key_found = None
        justification_key_found = None

        for key, value in criterion_data.items():
            norm_key = key.lower()
            if norm_key == "score":
                score = value
                score_key_found = key
            elif norm_key == "justification":
                justification = value
                justification_key_found = key

        if score_key_found is None:
            current_criterion_structural_errors.append(
                f"Missing 'score' key for criterion '{original_key}'."
            )
        if justification_key_found is None:
            current_criterion_structural_errors.append(
                f"Missing 'justification' key for criterion '{original_key}'."
            )

        # Validate score value only if score key was found
        if score_key_found is not None:
            if (
                not isinstance(score, str)
                or score.lower() not in normalized_allowed_scores
            ):
                current_criterion_structural_errors.append(
                    f"Invalid score value '{score}' for criterion '{original_key}'. Allowed: {allowed_scores}"
                )

        # If no structural errors *for this specific criterion*, add it to validated output
        if not current_criterion_structural_errors:
            validated_evaluation[original_key] = {
                score_key_found: score,
                justification_key_found: justification,
            }
            # Include any other keys the LLM might have added under the criterion
            for key, value in criterion_data.items():
                if key != score_key_found and key != justification_key_found:
                    validated_evaluation[original_key][key] = value
        else:
            # Add errors for this criterion to the main list if any were found
            structural_errors.extend(current_criterion_structural_errors)

    # --- Return based on validation results ---
    if missing_criteria:
        # Prioritize missing criteria errors
        return {
            "error": f"Missing expected criteria in 'evaluation': {', '.join(missing_criteria)}."
        }
    elif structural_errors:
        # If no missing criteria, report the first structural error found
        return {"error": structural_errors[0]}

    # If all checks pass, return the original parsed data structure but potentially
    # with the validated subset if we want strictness, or the full original if lenient.
    # Let's return the original structure containing the validated evaluation part.
    parsed_data["evaluation"] = validated_evaluation
    return parsed_data  # Return the full original structure


# --- Example Usage (Updated) ---
if __name__ == "__main__":
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
            "Results/Formulae": {
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
            "Results/Formulae": {
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
    test_expected = ["Problem Understanding", "Results/Formulae", "Assumptions"]

    print("--- Testing Valid JSON (Fenced) ---")
    parsed = parse_judge_response(
        valid_json_string_fenced, expected_criteria=test_expected
    )
    print(json.dumps(parsed, indent=2))
    assert "error" not in parsed
    assert _normalize_key("Problem Understanding") in {
        _normalize_key(k) for k in parsed.get("evaluation", {})
    }
    assert parsed["evaluation"]["Problem Understanding"]["score"] == "Yes"
    assert (
        parsed["evaluation"]["Results/Formulae"]["Score"] == "no"
    )  # Note: original key preserved
    assert parsed["evaluation"]["Assumptions"]["score"] == "Partial"

    print("\n--- Testing Valid JSON (Plain, Normalized Keys) ---")
    parsed = parse_judge_response(
        valid_json_string_plain, expected_criteria=test_expected
    )
    print(json.dumps(parsed, indent=2))
    # We expect an error because "Assumptions" is missing
    assert "error" in parsed
    assert "Missing expected criteria" in parsed["error"]
    assert "Assumptions" in parsed["error"]

    print("\n--- Testing Invalid JSON ---")
    parsed = parse_judge_response(invalid_json_string, expected_criteria=test_expected)
    print(json.dumps(parsed, indent=2))
    assert "error" in parsed
    assert "Could not find JSON block" in parsed["error"]  # Updated expectation

    print("\n--- Testing Missing Criterion JSON ---")
    parsed = parse_judge_response(missing_crit_json, expected_criteria=test_expected)
    print(json.dumps(parsed, indent=2))
    assert "error" in parsed
    assert "Missing expected criteria" in parsed["error"]
    assert "Results/Formulae" in parsed["error"]
    assert "Assumptions" in parsed["error"]

    print("\n--- Testing Missing Score Key JSON ---")
    parsed = parse_judge_response(
        missing_score_key_json, expected_criteria=["Problem Understanding"]
    )
    print(json.dumps(parsed, indent=2))
    assert "error" in parsed
    assert "Missing 'score' key" in parsed["error"]

    print("\n--- Testing Invalid Score Value JSON ---")
    parsed = parse_judge_response(
        invalid_score_value_json, expected_criteria=["Problem Understanding"]
    )
    print(json.dumps(parsed, indent=2))
    assert "error" in parsed
    assert "Invalid score value 'Maybe'" in parsed["error"]

    print("\n--- Testing No JSON Block ---")
    parsed = parse_judge_response(no_json_block, expected_criteria=test_expected)
    print(json.dumps(parsed, indent=2))
    assert "error" in parsed
    assert "Could not find JSON block" in parsed["error"]

    print("\n--- Testing Empty Input ---")
    parsed = parse_judge_response("", expected_criteria=test_expected)
    print(json.dumps(parsed, indent=2))
    assert "error" in parsed
    assert "Could not find JSON block" in parsed["error"]

    print("\n--- Testing Only Whitespace Input ---")
    parsed = parse_judge_response("   \n\t   ", expected_criteria=test_expected)
    print(json.dumps(parsed, indent=2))
    assert "error" in parsed
    assert "Could not find JSON block" in parsed["error"]

    print("\n--- All Tests Passed (Implicitly) ---")
