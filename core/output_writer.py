# CogniBench - Output Writer
# Version: 0.1 (Phase 1)

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

# Define the path to the evaluations file relative to this script's location
# Adjust if your project structure requires a different way to locate the data file
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
EVALUATIONS_FILE_PATH = os.path.join(DATA_DIR, 'evaluations.json')

def save_evaluation_result(
    evaluation_id: str,
    response_id: str,
    ideal_response_id: str,
    judge_llm_model: str,
    judge_prompt_template_version: str,
    raw_judge_output: Dict[str, Any],
    parsed_rubric_scores: Dict[str, Any],
    aggregated_score: Optional[str] = None, # Optional for Phase 1
    final_answer_verified: Optional[bool] = None # Optional for Phase 1
) -> Dict[str, Any]:
    """
    Appends a new evaluation result to the evaluations JSON file.

    Args:
        evaluation_id: A unique identifier for this evaluation.
        response_id: The ID of the model response being evaluated.
        ideal_response_id: The ID of the ideal response used for comparison.
        judge_llm_model: The name/identifier of the Judge LLM used.
        judge_prompt_template_version: Version identifier for the prompt template used.
        raw_judge_output: The raw output received from the Judge LLM.
        parsed_rubric_scores: The parsed rubric scores (e.g., from response_parser).
        aggregated_score: Overall score (e.g., Pass/Fail), populated later.
        final_answer_verified: Boolean indicating final answer match, populated later.


    Returns:
        A dictionary indicating success or failure.
        Example success: {"status": "success", "evaluation_id": evaluation_id}
        Example error: {"status": "error", "message": "Error details..."}
    """
    try:
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)

        # Read existing evaluations
        evaluations = []
        if os.path.exists(EVALUATIONS_FILE_PATH):
            with open(EVALUATIONS_FILE_PATH, 'r', encoding='utf-8') as f:
                try:
                    evaluations = json.load(f)
                    if not isinstance(evaluations, list):
                        print(f"Warning: {EVALUATIONS_FILE_PATH} does not contain a valid JSON list. Overwriting.")
                        evaluations = []
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {EVALUATIONS_FILE_PATH}. Overwriting.")
                    evaluations = []

        # Construct the new evaluation record
        new_evaluation = {
            "evaluation_id": evaluation_id,
            "response_id": response_id,
            "ideal_response_id": ideal_response_id,
            "judge_llm_model": judge_llm_model,
            "judge_prompt_template_version": judge_prompt_template_version,
            "raw_judge_output": raw_judge_output,
            "parsed_rubric_scores": parsed_rubric_scores,
            "aggregated_score": aggregated_score,
            "final_answer_verified": final_answer_verified,
            "human_review_status": "Pending",
            "human_reviewer_id": None,
            "human_review_timestamp": None,
            "human_corrected_scores": None,
            "human_review_comments": None,
            "created_at": datetime.utcnow().isoformat() + "Z" # ISO 8601 format
        }

        # Append the new record
        evaluations.append(new_evaluation)

        # Write the updated list back to the file
        with open(EVALUATIONS_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(evaluations, f, indent=2, ensure_ascii=False)

        print(f"Successfully saved evaluation {evaluation_id} to {EVALUATIONS_FILE_PATH}")
        return {"status": "success", "evaluation_id": evaluation_id}

    except IOError as e:
        print(f"File I/O Error saving evaluation: {e}")
        return {"status": "error", "message": f"File I/O Error: {e}"}
    except Exception as e:
        print(f"An unexpected error occurred saving evaluation: {e}")
        return {"status": "error", "message": f"Unexpected Error: {e}"}

# Example usage (for testing):
# if __name__ == "__main__":
#     # Example data (replace with actual data from previous steps)
#     test_eval_id = f"eval_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
#     test_parsed_data = {
#         "evaluation": {
#             "Problem Understanding": {"score": "Yes", "justification": "Test justification PU."},
#             "Results/Formulae": {"score": "No", "justification": "Test justification RF."}
#         }
#     }
#     test_raw_output = {"raw_content": json.dumps(test_parsed_data)} # Simulate raw output containing the JSON
#
#     result = save_evaluation_result(
#         evaluation_id=test_eval_id,
#         response_id="resp_001_modelA",
#         ideal_response_id="ideal_resp_001",
#         judge_llm_model="gpt-4o-test",
#         judge_prompt_template_version="v0.1-test",
#         raw_judge_output=test_raw_output,
#         parsed_rubric_scores=test_parsed_data.get("evaluation", {}) # Extract the inner evaluation dict
#     )
#     print(result)