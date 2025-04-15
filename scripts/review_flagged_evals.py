"""
CogniBench Flagged Evaluations Review Script.

Loads evaluation results from the standard JSON output file and displays
evaluations that have been flagged for human review based on the
'needs_human_review' flag.

Version: 0.1.1 (Phase 6 - Cleanup Pass)
"""

import json
from pathlib import Path  # Import Path
from typing import Any, Dict, List, Optional

# Define paths relative to the script's location using pathlib
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent  # Go up one level to CogniBench
DATA_DIR = BASE_DIR / "data"
EVALUATIONS_FILE_PATH = (
    DATA_DIR / "evaluations.json"
)  # Default path, might need update if using JSONL now


def load_json_data(file_path: Path) -> Optional[List[Dict[str, Any]]]:
    """Loads list data from a JSON file."""
    if not file_path.exists():
        print(f"Error: Data file not found at {file_path}")
        return None
    try:
        with file_path.open("r", encoding="utf-8") as f:
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
        print("   Review Reasons:")
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
    print(
        f"Loading evaluations from: {EVALUATIONS_FILE_PATH}"
    )  # Note: This script assumes evaluations.json, might need update for .jsonl
    all_evaluations = load_json_data(EVALUATIONS_FILE_PATH)

    if all_evaluations is not None:
        display_flagged_evaluations(all_evaluations)
    else:
        print("Could not load evaluations to display.")
