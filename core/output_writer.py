# -*- coding: utf-8 -*-
"""
CogniBench Output Writer Module.

Handles saving evaluation results to a JSONL file.
Each line in the output file represents a single evaluation record.

Version: 0.4
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Define the path relative to the project root (CogniBench/) using pathlib
# Assumes this file (output_writer.py) is in CogniBench/core/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_EVALUATIONS_FILE_PATH = DATA_DIR / "evaluations.jsonl"  # Default path

# Get logger for this module
logger = logging.getLogger("backend")


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
    verification_message: Optional[str] = None,  # Added verification_message
    needs_human_review: bool = False,
    review_reasons: Optional[List[str]] = None,
    task_id: Optional[str] = None,
    model_id: Optional[str] = None,
    output_jsonl_path: Optional[Path] = None,
    parsing_error: Optional[str] = None,
    structured_model_response: Optional[Dict[str, Any]] = None,  # Now an object
    structured_ideal_response: Optional[str] = None,  # Still a string here
    # New metrics
    structuring_api_calls: Optional[int] = None,
    judging_api_calls: Optional[int] = None,
    total_time_seconds: Optional[float] = None,
) -> Dict[str, Union[str, None]]:  # Return type indicates status and maybe message/id
    """Appends a structured evaluation result record to a JSONL file.

    Each call appends a single line representing one evaluation to the specified
    output file. If the file or directory doesn't exist, it attempts to create them.

    Args:
        evaluation_id: A unique identifier for this specific evaluation run.
        judge_llm_model: The name/identifier of the Judge LLM used.
        judge_prompt_template_path: Path to the prompt template file used.
        raw_judge_output: The raw output received from the Judge LLM (typically
            a dictionary containing the raw text).
        parsed_rubric_scores: The parsed rubric scores dictionary (criterion ->
            {'score': ..., 'justification': ...}) if parsing was successful,
            otherwise likely an empty dictionary.
        response_id: Optional legacy ID of the model response being evaluated.
        ideal_response_id: Optional legacy ID of the ideal response used.
        aggregated_score: Overall aggregated score ("Pass", "Fail", "Partial", or None).
        final_answer_verified: Boolean indicating final answer match, or None if skipped.
        verification_message: String describing the outcome of the verification step.
        needs_human_review: Boolean flag indicating if the evaluation requires human review.
        review_reasons: List of strings explaining why human review is needed, or None.
        task_id: Optional identifier for the original task/prompt.
        model_id: Optional identifier for the model that generated the response.
        output_jsonl_path: Optional `pathlib.Path` object for the target output
            JSONL file. If None, defaults to `data/evaluations.jsonl` relative
            to the project root.
        parsing_error: An optional string containing the error message if judge
            response parsing failed. Defaults to None.
        structured_model_response: Optional dictionary containing the structuring model name,
            prompt, and response string.
        structured_ideal_response: Optional string containing the structured ideal response.
        structuring_api_calls: Optional integer count of API calls to the structuring model.
        judging_api_calls: Optional integer count of API calls to the judging model.
        total_time_seconds: Optional float representing the total time taken for the evaluation.

    Returns:
        A dictionary indicating the outcome of the save operation:
        - On success: `{"status": "success", "evaluation_id": evaluation_id}`
        - On failure: `{"status": "error", "message": "Detailed error message"}`
    """
    target_path: Path = DEFAULT_EVALUATIONS_FILE_PATH  # Initialize with default
    try:
        # Determine the output path, using default if None provided
        target_path = (
            output_jsonl_path
            if output_jsonl_path is not None
            else DEFAULT_EVALUATIONS_FILE_PATH
        )
        logger.debug("Determined output path: %s", target_path)

        # Ensure the directory for the target path exists
        target_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured output directory exists: %s", target_path.parent)

        # Construct the new evaluation record dictionary
        # Ensure all values are JSON serializable
        new_evaluation: Dict[str, Any] = {
            "evaluation_id": evaluation_id,
            "task_id": task_id,
            "model_id": model_id,
            "response_id": response_id,  # Legacy field
            "ideal_response_id": ideal_response_id,  # Legacy field
            "judge_llm_model": judge_llm_model,
            "judge_prompt_template_path": str(
                judge_prompt_template_path
            ),  # Ensure string
            "raw_judge_output": raw_judge_output,  # Assumed dict/serializable
            "parsed_rubric_scores": parsed_rubric_scores,  # Assumed dict/serializable
            "aggregated_score": aggregated_score,
            "final_answer_verified": final_answer_verified,
            "verification_message": verification_message,  # Added field
            "needs_human_review": needs_human_review,
            "review_reasons": review_reasons or [],  # Ensure list, even if None
            "structured_model_response": structured_model_response,  # Store the object
            "structured_ideal_response": structured_ideal_response,  # Store the string
            "parsing_error": parsing_error,  # Include parsing error if present
            # --- Human Review Fields (Initialized) ---
            "human_review_status": "Needs Review"
            if needs_human_review
            else "Not Required",
            "human_reviewer_id": None,
            "human_review_timestamp": None,
            "human_corrected_scores": None,
            "human_review_comments": None,
            # --- Timestamp ---
            "created_at": datetime.utcnow().isoformat() + "Z",  # ISO 8601 format UTC
            # --- New Metrics ---
            "structuring_api_calls": structuring_api_calls,
            "judging_api_calls": judging_api_calls,
            "total_time_seconds": total_time_seconds,
        }

        # Convert the new record to a JSON string representation
        json_line: str = json.dumps(new_evaluation, ensure_ascii=False)

        # Append the JSON string as a new line to the target file
        with target_path.open("a", encoding="utf-8") as f:
            f.write(json_line + "\n")

        logger.debug(
            "Successfully saved evaluation %s to %s", evaluation_id, target_path
        )
        return {
            "status": "success",
            "evaluation_id": evaluation_id,
        }  # Return success status and ID

    except IOError as e:
        # Log the specific file path in the error message
        error_msg = f"File I/O Error saving evaluation to {target_path}: {e}"
        logger.error(error_msg, exc_info=True)
        return {"status": "error", "message": error_msg}
    except Exception as e:
        # Catch any other unexpected errors during saving
        error_msg = f"Unexpected Error saving evaluation {evaluation_id}: {e}"
        logger.exception(error_msg)  # Use exception to log full traceback
        return {"status": "error", "message": error_msg}


if __name__ == "__main__":
    # Example usage for testing if run directly
    # Setup basic logging for direct script execution test
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Running output_writer example...")

    # Example data (replace with actual data from previous steps in a real scenario)
    test_eval_id = (
        f"eval_test_{datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M')}"
    )
    test_parsed_scores = {
        "Criterion A": {"score": "Yes", "justification": "Looks good."},
        "Criterion B": {"score": "Partial", "justification": "Minor issue."},
    }
    test_raw_output = {"raw_content": "LLM output text..."}
    # Use a temporary directory for example output to avoid cluttering data/
    temp_output_dir = Path("./temp_test_output")
    temp_output_dir.mkdir(exist_ok=True)
    test_output_path = temp_output_dir / "test_output_writer_example.jsonl"

    # Call the function with example data including a parsing error
    result_with_error = save_evaluation_result(
        evaluation_id=test_eval_id + "_err",
        task_id="test_task_01",
        model_id="test_model_A",
        judge_llm_model="test_judge_model",
        judge_prompt_template_path="prompts/test_template.txt",
        raw_judge_output=test_raw_output,
        parsed_rubric_scores={},  # Empty when parsing fails
        aggregated_score=None,
        final_answer_verified=True,  # Verification might still run
        verification_message="Verification skipped: Extracted answer was None.",  # Example message
        needs_human_review=True,
        review_reasons=["LLM response parsing failed: Example parsing error message."],
        parsing_error="Example parsing error message.",  # Test with parsing error
        output_jsonl_path=test_output_path,
    )
    logger.info(
        "Example save_evaluation_result outcome (with error): %s", result_with_error
    )

    # Call the function with example data for success case
    result_success = save_evaluation_result(
        evaluation_id=test_eval_id + "_ok",
        task_id="test_task_02",
        model_id="test_model_B",
        judge_llm_model="test_judge_model",
        judge_prompt_template_path="prompts/test_template.txt",
        raw_judge_output=test_raw_output,
        parsed_rubric_scores=test_parsed_scores,
        aggregated_score="Partial",
        final_answer_verified=False,
        verification_message="Verification (String): Mismatch (Extracted: 'a', Correct: 'b').",  # Example message
        needs_human_review=True,
        review_reasons=["Partial score", "Answer mismatch"],
        parsing_error=None,  # No parsing error
        output_jsonl_path=test_output_path,
    )
    logger.info("Example save_evaluation_result outcome (success): %s", result_success)

    # Verify the file content (optional manual check or add assertions)
    if test_output_path.exists():
        logger.info("Check the content of %s", test_output_path)
        # Clean up the test file afterwards? Optional.
        # try:
        #     test_output_path.unlink()
        #     temp_output_dir.rmdir() # Remove dir only if empty
        #     logger.info("Cleaned up test output file and directory.")
        # except OSError as e:
        #     logger.warning("Could not clean up test output: %s", e)
    else:
        logger.error("Test output file was not created.")
