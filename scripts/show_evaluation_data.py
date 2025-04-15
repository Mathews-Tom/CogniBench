"""
CogniBench Evaluation Data Display Script (Potentially Outdated).

Loads data from various (potentially legacy) JSON files (prompts, model responses,
ideal responses, evaluations) and displays the details for a specific evaluation ID.

NOTE: This script assumes a data structure (separate JSON files, evaluations.json)
that might be outdated compared to the current ingestion and output formats
(single ingested file, evaluations.jsonl). It may require updates to function correctly.

Version: 0.1.1 (Phase 6 - Cleanup Pass)
"""

import argparse
import json
from pathlib import Path  # Import Path
from typing import Any, Dict, List, Optional

# --- Configuration ---
# Define paths using pathlib relative to the script's location
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent  # Go up one level to CogniBench
DATA_DIR = BASE_DIR / "data"
# Note: These file paths might point to legacy/non-existent files
PROMPTS_FILE = DATA_DIR / "prompts.json"
MODEL_RESPONSES_FILE = DATA_DIR / "model_responses.json"
IDEAL_RESPONSES_FILE = DATA_DIR / "ideal_responses.json"
EVALUATIONS_FILE = (
    DATA_DIR / "evaluations.json"
)  # Assumes .json, likely needs update to .jsonl


# --- Helper Functions ---
def load_json_data(file_path: Path) -> Optional[List[Dict[str, Any]]]:
    """Loads data from a JSON file containing a list of objects."""
    if not file_path.exists():
        print(f"Error: Data file not found at {file_path}")
        return None
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                print(f"Error: Expected a JSON list in {file_path}")
                return None
            return data
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


def find_item_by_id(
    data_list: Optional[List[Dict[str, Any]]], id_key: str, target_id: str
) -> Optional[Dict[str, Any]]:
    """Finds an item in a list of dictionaries by its ID."""
    if not data_list:
        return None
    for item in data_list:
        if item.get(id_key) == target_id:
            return item
    return None


def print_section(title: str, content: Any) -> None:
    """Helper to print sections clearly."""
    print("\n" + "=" * 10 + f" {title} " + "=" * 10)
    if isinstance(content, dict):
        print(json.dumps(content, indent=2))
    elif isinstance(content, str):
        print(content)
    else:
        print(str(content))
    print("=" * (22 + len(title)))


# --- Main Logic ---
def main() -> None:
    """Parses evaluation ID argument and displays associated data."""
    parser = argparse.ArgumentParser(
        description="Display data for a specific CogniBench evaluation."
    )
    parser.add_argument(
        "evaluation_id", help="The ID of the evaluation to display (e.g., 'eval_...')"
    )
    args = parser.parse_args()
    target_eval_id = args.evaluation_id

    print(f"Attempting to display data for Evaluation ID: {target_eval_id}")

    # Load all data
    evaluations = load_json_data(EVALUATIONS_FILE)
    model_responses = load_json_data(MODEL_RESPONSES_FILE)
    ideal_responses = load_json_data(IDEAL_RESPONSES_FILE)
    prompts = load_json_data(PROMPTS_FILE)

    if not all([evaluations, model_responses, ideal_responses, prompts]):
        print("Error loading one or more data files. Cannot proceed.")
        return

    # Find the target evaluation
    evaluation_data = find_item_by_id(evaluations, "evaluation_id", target_eval_id)
    if not evaluation_data:
        print(
            f"Error: Evaluation ID '{target_eval_id}' not found in {EVALUATIONS_FILE}"
        )
        return

    # Find associated records
    response_id = evaluation_data.get("response_id")
    ideal_response_id = evaluation_data.get("ideal_response_id")

    model_response_data = find_item_by_id(model_responses, "response_id", response_id)
    ideal_response_data = find_item_by_id(
        ideal_responses, "ideal_response_id", ideal_response_id
    )

    if not model_response_data:
        print(f"Warning: Could not find model response data for ID '{response_id}'")
        # Decide how to handle - maybe still show evaluation?
    if not ideal_response_data:
        print(
            f"Warning: Could not find ideal response data for ID '{ideal_response_id}'"
        )

    # Find the prompt (use ID from model response, fallback to ideal response)
    prompt_id = None
    if model_response_data:
        prompt_id = model_response_data.get("prompt_id")
    elif ideal_response_data:
        prompt_id = ideal_response_data.get("prompt_id")

    prompt_data = None
    if prompt_id:
        prompt_data = find_item_by_id(prompts, "prompt_id", prompt_id)
    else:
        print("Warning: Could not determine Prompt ID from response records.")

    if not prompt_data and prompt_id:
        print(f"Warning: Could not find prompt data for ID '{prompt_id}'")

    # Display the data
    if prompt_data:
        print_section("Prompt", prompt_data.get("content", "N/A"))
        print_section("Prompt Metadata", prompt_data.get("metadata", {}))
    else:
        print_section("Prompt", f"Prompt Data Not Found (ID: {prompt_id})")

    if model_response_data:
        print_section(
            "Model Response Text", model_response_data.get("response_text", "N/A")
        )
        print_section(
            "Model Info", {"model_name": model_response_data.get("model_name", "N/A")}
        )
    else:
        print_section(
            "Model Response", f"Model Response Data Not Found (ID: {response_id})"
        )

    if ideal_response_data:
        print_section(
            "Ideal Response Text", ideal_response_data.get("response_text", "N/A")
        )
        print_section(
            "Correct Answer", ideal_response_data.get("correct_answer", "N/A")
        )
    else:
        print_section(
            "Ideal Response", f"Ideal Response Data Not Found (ID: {ideal_response_id})"
        )

    print_section(f"Evaluation Details (ID: {target_eval_id})", evaluation_data)


if __name__ == "__main__":
    main()
