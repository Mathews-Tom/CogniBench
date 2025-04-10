# CogniBench - Output Writer
# Version: 0.2 (Phase 4 - Include Review Flags)

import json
import logging  # Import logging
from datetime import datetime

# import os # Replaced with pathlib
from pathlib import Path
from typing import Any, Dict, List, Optional

# Define the path relative to the project root (CogniBench/) using pathlib
PROJECT_ROOT = Path(__file__).parent.parent  # Assumes this file is in core/
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_EVALUATIONS_FILE_PATH = DATA_DIR / "evaluations.jsonl"  # Default path

# Get logger for this module
logger = logging.getLogger(__name__)


def save_evaluation_result(
    # Required parameters first
    evaluation_id: str,
    judge_llm_model: str,
    judge_prompt_template_path: str,
    raw_judge_output: Dict[str, Any],
    parsed_rubric_scores: Dict[str, Any],
    # Optional parameters follow
    response_id: Optional[str] = None,
    ideal_response_id: Optional[str] = None,
    aggregated_score: Optional[str] = None,
    final_answer_verified: Optional[bool] = None,
    needs_human_review: bool = False,
    review_reasons: Optional[List[str]] = None,
    task_id: Optional[str] = None,
    model_id: Optional[str] = None,
    output_jsonl_path: Optional[Path] = None,  # New parameter for output path
) -> Dict[str, Any]:
    """
    Appends a new evaluation result to the evaluations JSON file.

    Args:
        evaluation_id: A unique identifier for this evaluation.
        response_id: Optional ID of the model response being evaluated (legacy).
        ideal_response_id: Optional ID of the ideal response used for comparison (legacy).
        judge_llm_model: The name/identifier of the Judge LLM used.
        judge_prompt_template_path: Path to the prompt template file used.
        raw_judge_output: The raw output received from the Judge LLM.
        parsed_rubric_scores: The parsed rubric scores (e.g., from response_parser).
        aggregated_score: Overall score (e.g., "Pass", "Fail", "Partial").
        final_answer_verified: Boolean indicating final answer match, or None if skipped.
        needs_human_review: Boolean flag indicating if the evaluation requires human review.
        review_reasons: List of strings explaining why human review is needed.
        task_id: Optional identifier for the original task.
        model_id: Optional identifier for the model that generated the response.
        output_jsonl_path: Optional Path object for the target output JSONL file.
                            If None, defaults to 'data/evaluations.jsonl'.


    Returns:
        A dictionary indicating success or failure.
        Example success: {"status": "success", "evaluation_id": evaluation_id}
        Example error: {"status": "error", "message": "Error details..."}
    """
    try:
        # Determine the output path
        target_path = (
            output_jsonl_path if output_jsonl_path else DEFAULT_EVALUATIONS_FILE_PATH
        )

        # Ensure the directory for the target path exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Removed reading the entire file - we will append directly.
        # Construct the new evaluation record
        new_evaluation = {
            "evaluation_id": evaluation_id,
            "task_id": task_id,  # Added
            "model_id": model_id,  # Added
            "response_id": response_id,  # Kept for potential cross-referencing if needed
            "ideal_response_id": ideal_response_id,
            "judge_llm_model": judge_llm_model,
            "judge_prompt_template_path": str(
                judge_prompt_template_path
            ),  # Store path as string
            "raw_judge_output": raw_judge_output,
            "parsed_rubric_scores": parsed_rubric_scores,
            "aggregated_score": aggregated_score,
            "final_answer_verified": final_answer_verified,
            "needs_human_review": needs_human_review,  # Store the flag
            "review_reasons": review_reasons or [],  # Store the reasons (ensure list)
            "human_review_status": "Needs Review"
            if needs_human_review
            else "Not Required",  # Set status based on flag
            "human_reviewer_id": None,
            "human_review_timestamp": None,
            "human_corrected_scores": None,
            "human_review_comments": None,
            "created_at": datetime.utcnow().isoformat() + "Z",  # ISO 8601 format
        }

        # Convert the new record to a JSON string
        json_line = json.dumps(new_evaluation, ensure_ascii=False)

        # Append the JSON string as a new line to the target file
        with target_path.open("a", encoding="utf-8") as f:
            f.write(json_line + "\n")

        logger.debug(  # Changed to debug
            "Successfully saved evaluation %s to %s", evaluation_id, target_path
        )
        return {"status": "success", "evaluation_id": evaluation_id}

    except IOError as e:
        logger.error(
            "File I/O Error saving evaluation to %s", target_path, exc_info=True
        )
        return {"status": "error", "message": f"File I/O Error: {e}"}
    except Exception as e:
        logger.exception(
            "An unexpected error occurred saving evaluation %s", evaluation_id
        )  # Use exception to log traceback
        return {"status": "error", "message": f"Unexpected Error: {e}"}


if __name__ == "__main__":
    # Example data (replace with actual data from previous steps)
    test_eval_id = f"eval_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    test_parsed_data = {
        "evaluation": {
            "Problem Understanding": {
                "score": "Yes",
                "justification": "Test justification PU.",
            },
            "Results Formulae": {
                "score": "No",
                "justification": "Test justification RF.",
            },
        }
    }
    test_raw_output = {
        "raw_content": json.dumps(test_parsed_data)
    }  # Simulate raw output containing the JSON

    # Update example call if needed, though this __main__ block might be removed later
    result = save_evaluation_result(
        evaluation_id=test_eval_id,
        task_id="example_task_001",  # Example
        model_id="example_model_X",  # Example
        response_id="resp_001_modelA",  # Example
        ideal_response_id="ideal_resp_001",  # Example
        judge_llm_model="gpt-4o-test",
        judge_prompt_template_path="prompts/example_template.txt",  # Example path
        raw_judge_output=test_raw_output,
        parsed_rubric_scores=test_parsed_data.get("evaluation", {}),
        needs_human_review=True,  # Example
        review_reasons=["Example reason"],  # Example
        output_jsonl_path=Path("data/example_output.jsonl"),  # Example path for testing
    )
    # Example usage might need logging setup if run directly
    # from log_setup import setup_logging
    # setup_logging()
    logger.info("Example save result: %s", result)
    # Removed duplicate log line
