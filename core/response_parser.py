# CogniBench - LLM Judge Response Parser
# Version: 0.1 (Phase 1 - Initial Criteria)

import json
from typing import Any, Dict, List

# Define the expected keys for the full L1 rubric evaluation (Phase 2)
EXPECTED_CRITERIA_FULL_L1 = [
    "Problem Understanding",
    "Assumptions",
    "Logical Implications",
    "Results/Formulae",
    "Rigor and Completeness",
]


def parse_judge_response(
    raw_response_content: str, expected_criteria: List[str] = EXPECTED_CRITERIA_FULL_L1
) -> Dict[str, Any]:
    """
    Parses the raw string response from the Judge LLM, expecting JSON format.
    Validates the basic structure based on the expected criteria for the current phase.

    Args:
        raw_response_content: The raw string output from the LLM.
        expected_criteria: A list of top-level keys expected under the 'evaluation' object.

    Returns:
        A dictionary containing the parsed JSON data if successful and valid,
        or a dictionary with an 'error' key if parsing or validation fails.
        Example success: {"evaluation": {"Problem Understanding": {...}, ...}}
        Example error: {"error": "Invalid JSON format."}
        Example error: {"error": "Missing expected criterion: 'Problem Understanding'"}
    """
    try:
        # Attempt to find JSON block if the LLM included extra text (common issue)
        json_start = raw_response_content.find("{")
        json_end = raw_response_content.rfind("}")
        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_string = raw_response_content[json_start : json_end + 1]
        else:
            # Assume the whole string is JSON if no clear block found
            json_string = raw_response_content.strip()

        parsed_data = json.loads(json_string)

        # Basic structural validation
        if not isinstance(parsed_data, dict):
            return {"error": "Parsed data is not a JSON object (dictionary)."}

        if "evaluation" not in parsed_data:
            return {"error": "Missing required top-level key: 'evaluation'."}

        evaluation_content = parsed_data["evaluation"]
        if not isinstance(evaluation_content, dict):
            return {"error": "'evaluation' value is not a JSON object (dictionary)."}

        # Check for expected criteria keys and their structure
        for criterion in expected_criteria:
            if criterion not in evaluation_content:
                return {
                    "error": f"Missing expected criterion in 'evaluation': '{criterion}'."
                }
            if not isinstance(evaluation_content[criterion], dict):
                return {
                    "error": f"Value for criterion '{criterion}' is not a JSON object."
                }
            if "score" not in evaluation_content[criterion]:
                return {"error": f"Missing 'score' key for criterion '{criterion}'."}
            if "justification" not in evaluation_content[criterion]:
                return {
                    "error": f"Missing 'justification' key for criterion '{criterion}'."
                }

        # If all checks pass, return the parsed data
        return parsed_data

    except json.JSONDecodeError as e:
        print(f"JSON Parsing Error: {e}")
        print(f"Raw content was:\n---\n{raw_response_content}\n---")
        return {"error": f"Invalid JSON format: {e}"}
    except Exception as e:
        print(f"An unexpected error occurred during parsing: {e}")
        return {"error": f"Unexpected parsing error: {e}"}


if __name__ == "__main__":
    valid_json_string = """
    ```json
    {
        "evaluation": {
            "Problem Understanding": {
            "score": "Yes",
            "justification": "Model correctly identified the integral."
            },
            "Results/Formulae": {
            "score": "No",
            "justification": "Final answer was incorrect."
            }
        }
    }
    ```
    """
    invalid_json_string = "{ evaluation: { Problem Understanding: ... }"
    missing_key_json = """{"evaluation": {"Problem Understanding": {"score": "Yes"}}}"""

    print("--- Testing Valid JSON ---")
    parsed = parse_judge_response(valid_json_string)
    print(parsed)

    print("\n--- Testing Invalid JSON ---")
    parsed = parse_judge_response(invalid_json_string)
    print(parsed)

    print("\n--- Testing Missing Key JSON ---")
    parsed = parse_judge_response(missing_key_json)
    print(parsed)
