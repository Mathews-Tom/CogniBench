# CogniBench/scripts/review_flagged_evals.py
# Version: 0.1 (Phase 4 - Basic Review Script)

import json
import os
from typing import Any, Dict, List, Optional

# Define paths relative to the script's location
SCRIPT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Go up one level to CogniBench
DATA_DIR = os.path.join(BASE_DIR, "data")
EVALUATIONS_FILE_PATH = os.path.join(DATA_DIR, "evaluations.json")


def load_json_data(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """Loads list data from a JSON file."""
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                print(f"Error: Data in {file_path} is not a JSON list.")
                return None
            return data
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading or parsing data from {file_path}: {e}")
        return None


def display_flagged_evaluations(evaluations: List[Dict[str, Any]]):
    """Filters and displays evaluations flagged for human review."""
    flagged_evals = [
        eval_item
        for eval_item in evaluations
        if eval_item.get("needs_human_review") is True
        and eval_item.get("human_review_status")
        != "Completed"  # Optional: Skip already completed ones
    ]

    if not flagged_evals:
        print("No evaluations currently flagged for human review.")
        return

    print(f"\n--- Evaluations Flagged for Human Review ({len(flagged_evals)}) ---")

    for i, eval_item in enumerate(flagged_evals, 1):
        print(f"\n{i}. Evaluation ID: {eval_item.get('evaluation_id', 'N/A')}")
        print(f"   Response ID:   {eval_item.get('response_id', 'N/A')}")
        print(f"   Ideal Resp ID: {eval_item.get('ideal_response_id', 'N/A')}")
        print(f"   Created At:    {eval_item.get('created_at', 'N/A')}")
        print(f"   Review Status: {eval_item.get('human_review_status', 'N/A')}")
        print(f"   Review Reasons:")
        reasons = eval_item.get("review_reasons", [])
        if reasons:
            for reason in reasons:
                print(f"     - {reason}")
        else:
            print("     - (No specific reasons listed, flagged generically)")

        # Optional: Display parsed scores for context
        parsed_scores = eval_item.get("parsed_rubric_scores")
        if parsed_scores:
            print("   Parsed Scores:")
            for criterion, details in parsed_scores.items():
                score = details.get("score", "N/A")
                justification = details.get("justification", "N/A")
                print(
                    f"     - {criterion}: {score} ('{justification[:50]}...')"
                )  # Truncate justification

        print("-" * 20)

    print("--- End of Flagged Evaluations ---")
    # TODO: Add functionality to mark an evaluation as reviewed or provide feedback directly?


if __name__ == "__main__":
    print(f"Loading evaluations from: {EVALUATIONS_FILE_PATH}")
    all_evaluations = load_json_data(EVALUATIONS_FILE_PATH)

    if all_evaluations is not None:
        display_flagged_evaluations(all_evaluations)
    else:
        print("Could not load evaluations to display.")
