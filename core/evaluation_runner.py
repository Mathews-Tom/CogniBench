"""
CogniBench Evaluation Runner Module.

Provides core functions to orchestrate the evaluation process for single tasks
or batches of tasks defined in input files. It integrates various components
like configuration loading, workflow execution, and output handling.
"""

import json
import logging
import re
import sys
import threading  # Added import
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from .config import AppConfig
from .workflow import run_evaluation_workflow

logger = logging.getLogger(__name__)


# --- Helper Functions ---
# (Assuming _keys_to_snake_case and _to_snake_case are defined below or imported)


def _to_snake_case(name: str) -> str:
    """Converts CamelCase, PascalCase, or space-separated string to snake_case."""
    if not isinstance(name, str):
        return name  # Return original if not a string
    if " " in name:  # Handle space-separated
        return name.lower().replace(" ", "_")
    # Handle CamelCase/PascalCase
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _keys_to_snake_case(data: Any) -> Any:
    """Recursively converts dictionary keys to snake_case."""
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            new_key = _to_snake_case(key)
            new_dict[new_key] = _keys_to_snake_case(value)
        return new_dict
    elif isinstance(data, list):
        return [_keys_to_snake_case(item) for item in data]
    else:
        return data


# --- Core Evaluation Functions ---


def run_single_task_evaluation_core(
    task_index: int,  # Added task index
    task_data: Dict[str, Any],
    config: AppConfig,
    use_structured: bool,
    output_jsonl_path: Optional[Path] = None,
    structured_ideal_cache: Optional[Dict[str, Any]] = None,
    stop_event: Optional[threading.Event] = None,  # Added stop_event
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Runs the evaluation workflow for a single task dictionary.
    Processes one task, evaluating multiple model responses against the ideal.
    """
    logger.debug(f"Received task_data: {json.dumps(task_data, indent=2)}")
    # --- Extract data from task_data dictionary (handling original RLHF structure) ---
    # Prioritize "taskId", then "id", then "task_id"
    task_id = task_data.get(
        "taskId", task_data.get("id", task_data.get("task_id", f"unknown_{task_index}"))
    )

    messages = task_data.get("messages", [])
    prompt_text = None
    ideal_response_text = None
    model_responses_raw = []
    correct_answer = task_data.get(
        "final_answer"
    )  # Keep trying to get this if available

    # Initialize variables for extracted data
    prompt_text = None
    ideal_response_text = None
    model_responses_raw = []
    # Keep existing correct_answer logic if it's top-level
    correct_answer = task_data.get("correct_answer")

    # --- Primary Extraction (RLHF Format) ---
    messages = task_data.get("messages", [])
    # 1. Extract Prompt from messages[1]['text']
    # Check if messages is a list, has at least 2 elements, and the second element is a dict
    if (
        isinstance(messages, list)
        and len(messages) > 1
        and isinstance(messages[1], dict)
    ):
        prompt_text = messages[1].get("text")  # Use 'text' key as per RLHF structure
        if prompt_text:
            logger.debug(f"Extracted prompt from messages[1]['text']")

    # 2. Extract Ideal Response and Model Responses from messages[2]
    # Check if messages is a list, has at least 3 elements, and the third element is a dict
    if (
        isinstance(messages, list)
        and len(messages) > 2
        and isinstance(messages[2], dict)
    ):
        assistant_message = messages[2]

        # 2a. Extract Ideal Response from messages[2]['signal']['ideal_response']
        signal_data = assistant_message.get("signal", {})
        if isinstance(signal_data, dict):
            ideal_response_text = signal_data.get("ideal_response")
            if ideal_response_text:
                logger.debug(
                    f"Extracted ideal_response from messages[2]['signal']['ideal_response']"
                )

        # 2b. Extract Model Responses from messages[2]['response_options'][*]['text']
        response_options = assistant_message.get("response_options", [])
        if isinstance(response_options, list):
            # Use 'text' key for response as per RLHF structure
            model_responses_raw = [
                {
                    "model_id": opt.get("model_id"),
                    "response_text": opt.get("text"),
                }  # Use 'text' key
                for opt in response_options
                if opt.get("model_id") and opt.get("text")  # Check 'text' key
            ]
            if model_responses_raw:
                logger.debug(
                    f"Extracted {len(model_responses_raw)} model responses from messages[2]['response_options']"
                )

    # --- Fallback Extraction (Ingested/Direct Format) ---
    # Apply fallback only if primary RLHF extraction failed for each field

    if prompt_text is None:
        logger.debug("RLHF prompt extraction failed, falling back to 'prompt' key.")
        prompt_text = task_data.get("prompt")

    if ideal_response_text is None:
        logger.debug(
            "RLHF ideal response extraction failed, falling back to 'ideal_response' key."
        )
        ideal_response_text = task_data.get("ideal_response")

    if not model_responses_raw:
        logger.debug(
            "RLHF model responses extraction failed, falling back to 'model_responses' key."
        )
        fallback_responses = task_data.get("model_responses", [])
        # Ensure fallback model responses also use 'response_text' key for consistency downstream
        # This handles cases where the ingested data might use 'response' instead of 'response_text'
        if isinstance(fallback_responses, list):
            model_responses_raw = [
                # Try 'response' then 'text' for broader compatibility in fallback
                {
                    "model_id": opt.get("model_id"),
                    "response_text": opt.get("text"),
                }  # Use 'text' key
                for opt in fallback_responses
                if opt.get("model_id") and opt.get("text")  # Check 'text' key
            ]
            if model_responses_raw:
                logger.debug(
                    f"Extracted {len(model_responses_raw)} model responses using fallback 'model_responses' key."
                )

    structured_ideal_response = None  # Still not expected in this format

    # Transform model_responses_raw into the expected format
    # (Handles both RLHF structure's 'response' key and ingested 'response_text')
    model_responses = []
    if isinstance(model_responses_raw, list):
        for resp_option in model_responses_raw:
            model_id = resp_option.get("model_id")
            # Check both 'response' (from RLHF) and 'response_text' (from potential ingestion)
            text = resp_option.get(
                "response_text"
            )  # Use the correct key 'response_text'
            if model_id and text is not None:
                model_responses.append({"model_id": model_id, "response_text": text})
            else:
                logger.warning(
                    "Task [%s]: Skipping invalid model response option: %s",
                    task_id,
                    resp_option,
                )
    else:
        logger.warning("Task [%s]: Could not find valid model responses list.", task_id)

    task_results = []
    task_success = True

    # Check if essential data was found after trying both structures
    if not prompt_text or not ideal_response_text:
        logger.warning(
            "Task [%s]: Skipping task due to missing prompt (%s) or ideal response (%s) after checking structures.",
            task_id,
            "found" if prompt_text else "missing",
            "found" if ideal_response_text else "missing",
        )
        return [], False

    if structured_ideal_cache is None:
        structured_ideal_cache = {}

    for model_response in model_responses:
        # --- Check for stop signal ---
        if stop_event and stop_event.is_set():
            logger.warning(
                "Stop event detected during model response loop for task [%s]. Aborting task.",
                task_id,
            )
            task_success = False  # Mark task as failed due to cancellation
            break  # Exit the model_response loop

        response_text = model_response.get("response_text")
        structured_model_response = model_response.get("structured_model_response")
        model_id = model_response.get("model_id", "unknown_model")

        response_input: Optional[Union[str, Dict]] = None
        ideal_input: Optional[Union[str, Dict]] = None
        structured_eval = False

        if (
            use_structured
            and structured_model_response is not None
            and structured_ideal_response is not None
        ):
            if isinstance(structured_model_response, dict) and isinstance(
                structured_ideal_response, dict
            ):
                response_input = structured_model_response
                ideal_input = structured_ideal_response
                structured_eval = True
            else:
                logger.warning(
                    "Task [%s] Model [%s]: Structured evaluation requested but "
                    "structured_model_response (type: %s) or structured_ideal_response (type: %s) "
                    "is not a dictionary. Falling back to text.",
                    task_id,
                    model_id,
                    type(structured_model_response).__name__,
                    type(structured_ideal_response).__name__,
                )
                response_input = response_text
                ideal_input = ideal_response_text
                structured_eval = False
        else:
            response_input = response_text
            ideal_input = ideal_response_text
            structured_eval = False

        if (
            response_input is None
        ):  # Check only response_input, ideal_input validated earlier
            logger.warning(
                "Task [%s] Model [%s]: Skipping response due to missing required input "
                "(response: %s, ideal: %s) after structure selection (structured_eval=%s).",
                task_id,
                model_id,
                "present" if response_input is not None else "missing",
                "present" if ideal_input is not None else "missing",
                structured_eval,
            )
            continue

        # --- Call the core workflow ---
        result = run_evaluation_workflow(
            prompt=prompt_text,
            response=response_input,
            ideal_response=ideal_input,
            correct_answer=correct_answer,
            config=config,
            task_id=task_id,
            model_id=model_id,
            structured=structured_eval,
            output_jsonl_path=output_jsonl_path,
            structured_ideal_cache=structured_ideal_cache,
            # Note: stop_event is not directly passed to workflow currently
            # Workflow needs modification if cancellation during API calls is needed
        )
        logger.debug(
            "Task [%s] Model [%s]: Received result from workflow: Type=%s, Value=%s",
            task_id,
            model_id,
            type(result),
            result,
        )

        task_results.append(result)

        if result.get("status") != "success":
            task_success = False
            logger.error(
                "Task [%s] Model [%s]: Workflow error: %s",
                task_id,
                model_id,
                result.get("message", "Unknown error"),
            )

    return task_results, task_success


def run_batch_evaluation_core(
    config: AppConfig,
    output_dir: Path,
    use_structured: bool,
    stop_event: Optional[threading.Event] = None,
) -> Optional[List[str]]:
    """
    Orchestrates the full batch evaluation process for multiple input files.
    """
    logger.info("Starting batch evaluation core process...")
    logger.info(f"Received output_dir argument: {output_dir}")  # Added diagnostic log
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Using structured evaluation: {use_structured}")
    output_dir_path = Path(output_dir)  # Ensure output_dir is a Path object
    logger.info(f"Converted output_dir_path: {output_dir_path}")  # Added diagnostic log

    # --- Get Input File Paths from Config ---
    input_options = getattr(config, "input_options", None)
    file_paths_str = getattr(input_options, "file_paths", []) if input_options else []

    if not file_paths_str:
        logger.error(
            "No input file paths found in configuration (config.input_options.file_paths)."
        )
        return None

    input_file_paths = [Path(p) for p in file_paths_str]
    logger.info(f"Processing {len(input_file_paths)} input file(s): {file_paths_str}")

    all_final_results_paths = []
    overall_batch_success = True

    # --- Loop Through Each Input File ---
    for input_path in input_file_paths:
        logger.info(f"--- Processing file: {input_path.name} ---")
        file_processing_success = True  # Track success for this specific file

        if stop_event and stop_event.is_set():
            logger.warning(
                "Stop event detected before processing file %s. Aborting batch.",
                input_path.name,
            )
            overall_batch_success = False
            break  # Exit the file loop

        # --- 1. Define Output Paths for this file ---
        try:
            # Get the stem from the input file path
            original_stem = input_path.stem
            cleaned_stem = original_stem
            # Clean common suffixes like '_ingested' or '_tasks'
            common_suffixes = [
                "_ingested",
                "_tasks",
                ".json",
            ]  # .json might be redundant with .stem but safe to keep
            for suffix in common_suffixes:
                if cleaned_stem.endswith(suffix):
                    cleaned_stem = cleaned_stem[: -len(suffix)]
            logger.info(f"Using cleaned input stem for output files: {cleaned_stem}")

            output_dir_path.mkdir(
                parents=True, exist_ok=True
            )  # Ensure output directory exists

            # Construct output paths using the cleaned input stem
            # The timestamp is implicitly handled by the output_dir_path itself
            eval_jsonl_path = output_dir_path / f"{cleaned_stem}_evaluations.jsonl"
            logger.info(f"Constructed eval_jsonl_path: {eval_jsonl_path}")
            eval_json_path = (
                output_dir_path / f"{cleaned_stem}_evaluations_formatted.json"
            )
            final_results_path = output_dir_path / f"{cleaned_stem}_final_results.json"

            logger.info(f"  Detailed results (JSONL): {eval_jsonl_path}")
            logger.info(f"  Formatted results (JSON): {eval_json_path}")
            logger.info(f"  Final combined results (JSON): {final_results_path}")

        except Exception as e:
            logger.error(
                f"Error defining output paths for {input_path.name}: {e}", exc_info=True
            )
            overall_batch_success = False
            file_processing_success = False
            continue  # Skip to the next file

        # --- 2. Load Ingested Data for this file ---
        try:
            if not input_path.is_file():
                logger.error(f"Input data file not found: {input_path}")
                overall_batch_success = False
                file_processing_success = False
                continue
            with input_path.open("r", encoding="utf-8") as f:
                loaded_data = json.load(f)

            # Extract the actual task list (assuming it's under the 'rlhf' key)
            if isinstance(loaded_data, dict) and "rlhf" in loaded_data:
                evaluation_tasks: List[Dict[str, Any]] = loaded_data.get("rlhf", [])
                if not isinstance(evaluation_tasks, list):
                    logger.error(
                        f"Data under 'rlhf' key in {input_path.name} is not a list."
                    )
                    raise TypeError("Expected a list under 'rlhf' key.")
            elif isinstance(loaded_data, list):
                # Assume it's already the list of tasks if not a dict with 'rlhf'
                evaluation_tasks = loaded_data
            else:
                logger.error(
                    f"Unexpected JSON structure in {input_path.name}. Expected a list or a dict with 'rlhf' key."
                )
                raise TypeError("Unexpected JSON structure.")

            logger.info(
                f"  Loaded {len(evaluation_tasks)} tasks from {input_path.name}"
            )
        except (IOError, json.JSONDecodeError, TypeError) as e:  # Added TypeError
            logger.error(
                f"Failed to load or parse task data from {input_path.name}: {e}",
                exc_info=True,
            )
            overall_batch_success = False
            file_processing_success = False
            continue
        except Exception as e:
            logger.error(
                f"Unexpected error loading ingested data {input_path.name}: {e}",
                exc_info=True,
            )
            overall_batch_success = False
            file_processing_success = False
            continue

        # --- 3. Run Evaluations Task by Task for this file ---
        try:  # Wrap evaluation loop in try/except for this file
            structured_ideal_cache = {}
            file_overall_success = True  # Track success of tasks within this file
            total_tasks = len(evaluation_tasks)
            all_evaluation_results_list = []

            # Clear the JSONL file if it exists before starting
            logger.info(
                f"Checking existence for eval_jsonl_path: {eval_jsonl_path}"
            )  # Added diagnostic log
            if eval_jsonl_path.exists():
                logger.warning(
                    f"Output JSONL file {eval_jsonl_path} already exists. Overwriting."
                )
                try:
                    eval_jsonl_path.unlink()
                except OSError as e:
                    logger.error(
                        f"Failed to remove existing JSONL file {eval_jsonl_path}: {e}"
                    )
                    raise  # Re-raise to be caught by the outer try/except for this file

            logger.info(f"  Starting evaluation loop for {total_tasks} tasks...")
            for i, task_data in enumerate(
                tqdm(
                    evaluation_tasks,
                    desc=f"Evaluating Tasks ({input_path.name})",
                    file=sys.stdout,
                    leave=False,
                )
            ):
                if stop_event and stop_event.is_set():
                    logger.warning(
                        "Stop event detected during task evaluation loop. Aborting."
                    )
                    file_overall_success = False
                    break  # Exit the task loop

                task_id = task_data.get("task_id", f"unknown_task_{i}")

                task_results, task_success = run_single_task_evaluation_core(
                    task_index=i,  # Pass the index
                    task_data=task_data,
                    config=config,
                    use_structured=use_structured,
                    output_jsonl_path=eval_jsonl_path,
                    structured_ideal_cache=structured_ideal_cache,
                    stop_event=stop_event,
                )

                all_evaluation_results_list.extend(task_results)

                if not task_success:
                    file_overall_success = False
                    # Error already logged

            logger.info(f"  Evaluation loop finished for {input_path.name}.")
            if not file_overall_success:
                logger.warning(
                    f"  Evaluation for {input_path.name} completed with one or more task errors or was cancelled."
                )
                overall_batch_success = (
                    False  # Mark entire batch as failed if any file has task errors
                )

            # Check if JSONL was created. Only raise error if tasks were expected to succeed.
            if not eval_jsonl_path.is_file():
                if (
                    file_overall_success
                ):  # If tasks succeeded but file is missing, it's an error
                    logger.error(
                        f"Evaluation loop finished successfully for one or more tasks, "
                        f"but output JSONL file {eval_jsonl_path} was not created."
                    )
                    # Raise the original error to be caught by the file-level handler
                    raise IOError(
                        f"JSONL file not created despite task success: {eval_jsonl_path}"
                    )
                else:  # If tasks failed/cancelled and file is missing, it might be expected. Log warning.
                    logger.warning(
                        f"Evaluation loop finished with task failures or cancellation, "
                        f"and output JSONL file {eval_jsonl_path} was not created (potentially expected)."
                    )
            # Continue processing (like formatting) even if file is empty or wasn't created due to task failures

            # --- 4. Format JSONL Output ---
            logger.info(f"  Formatting evaluation results from {eval_jsonl_path}...")
            formatted_evaluations_list = []
            # This section needs its own try/except as it reads the potentially incomplete JSONL
            # Only attempt to read/format if the JSONL file actually exists
            if eval_jsonl_path.exists():
                try:
                    with eval_jsonl_path.open("r", encoding="utf-8") as infile:
                        for line_num, line in enumerate(infile):
                            try:
                                if line.strip():
                                    raw_evaluation_data = json.loads(line)
                                    evaluation_data = _keys_to_snake_case(
                                        raw_evaluation_data
                                    )
                                    evaluation_data.pop("raw_judge_output", None)
                                    formatted_evaluations_list.append(evaluation_data)
                            except json.JSONDecodeError as e:
                                logger.warning(
                                    f"Skipping invalid JSON line {line_num + 1} in {eval_jsonl_path}: {line.strip()} - Error: {e}"
                                )
                    # Write the formatted JSON file
                    with eval_json_path.open("w", encoding="utf-8") as outfile:
                        json.dump(
                            formatted_evaluations_list,
                            outfile,
                            indent=2,
                            ensure_ascii=False,
                        )
                    logger.info(
                        f"  Successfully created formatted JSON at {eval_json_path}"
                    )
                except (IOError, Exception) as e:
                    logger.error(
                        f"Error during JSONL formatting/writing for {eval_json_path}: {e}",
                        exc_info=True,
                    )
                    raise  # Re-raise to be caught by the outer try/except for this file
            else:
                logger.warning(
                    f"  Skipping formatting: Input JSONL file {eval_jsonl_path} does not exist (likely no successful evaluations)."
                )
                # Ensure the formatted JSON is still written, even if empty, to avoid downstream errors
                try:
                    with eval_json_path.open("w", encoding="utf-8") as outfile:
                        json.dump(
                            formatted_evaluations_list,  # This will be empty
                            outfile,
                            indent=2,
                            ensure_ascii=False,
                        )
                    logger.info(
                        f"  Created empty formatted JSON at {eval_json_path} as input JSONL was missing."
                    )
                except IOError as e:
                    logger.error(
                        f"Error writing empty formatted JSON file {eval_json_path}: {e}",
                        exc_info=True,
                    )
                    raise  # Re-raise

            # --- 5. Combine Ingested Data and Formatted Evaluations ---
            logger.info("Starting final result aggregation phase.")  # Added log
            logger.info(
                "  Combining ingested data with formatted evaluation results..."
            )
            # This section also needs its own try/except

            # Load ingested data again and extract the task list correctly
            with input_path.open("r", encoding="utf-8") as f:
                loaded_ingested_data = json.load(f)

            if (
                isinstance(loaded_ingested_data, dict)
                and "rlhf" in loaded_ingested_data  # Check for 'rlhf'
            ):
                ingested_tasks_list = loaded_ingested_data.get(
                    "rlhf", []
                )  # Get from 'rlhf'
                if not isinstance(ingested_tasks_list, list):
                    logger.error(
                        f"Data under 'rlhf' key in {input_path.name} is not a list (for combination step)."  # Log 'rlhf'
                    )
                    raise TypeError(
                        "Expected a list under 'rlhf' key."
                    )  # Error for 'rlhf'
            elif isinstance(loaded_ingested_data, list):
                ingested_tasks_list = loaded_ingested_data
            else:
                logger.error(
                    f"Unexpected JSON structure in {input_path.name} (for combination step)."
                )
                raise TypeError("Unexpected JSON structure.")

            # Now build the map from the extracted list
            ingested_data_map = {
                # Use same logic as run_single_task_evaluation_core for consistency
                # Prioritize "taskId", then "id", then "task_id" for consistency
                task.get(
                    "taskId", task.get("id", task.get("task_id", f"unknown_{i}"))
                ): task
                for i, task in enumerate(ingested_tasks_list)
            }  # Use .get() and taskId

            grouped_results_map = {}
            total_structuring_calls = 0
            total_judging_calls = 0
            total_eval_time = 0.0
            total_evaluations_processed = 0
            model_times = {}
            model_counts = {}

            # Added logs for JSONL path and record count
            logger.info(f"Aggregating results read from: {eval_jsonl_path}")
            logger.info(
                f"Total evaluation records loaded for aggregation: {len(formatted_evaluations_list)}"
            )
            # End added logs

            logger.info(
                f"  Starting final combination. Number of formatted evaluations: {len(formatted_evaluations_list)}"
            )
            if len(formatted_evaluations_list) < 5:  # Log content only if list is small
                logger.debug(
                    f"  Formatted evaluations content: {formatted_evaluations_list}"
                )

            for evaluation in formatted_evaluations_list:
                # (Logic for accumulating metrics and combining data as before)
                total_evaluations_processed += 1
                task_id = evaluation.get("task_id")
                model_id = evaluation.get("model_id")
                logger.debug(
                    f"  Processing evaluation - Extracted task_id: '{task_id}' (type: {type(task_id)}), model_id: '{model_id}' (type: {type(model_id)})"
                )  # Add this line

                if not task_id or not model_id:
                    continue  # Skip if essential IDs missing

                struct_calls = evaluation.get("structuring_api_calls", 0) or 0
                judge_calls = evaluation.get("judging_api_calls", 0) or 0
                eval_time = evaluation.get("total_time_seconds", 0.0) or 0.0
                total_structuring_calls += struct_calls
                total_judging_calls += judge_calls
                total_eval_time += eval_time
                if model_id:
                    model_times[model_id] = model_times.get(model_id, 0.0) + eval_time
                    model_counts[model_id] = model_counts.get(model_id, 0) + 1

                original_task_data = ingested_data_map.get(task_id)
                if not original_task_data:
                    continue  # Skip if original task missing

                # Try extracting from RLHF structure first, then fallback to ingested format
                original_prompt = None
                original_ideal_response = None

                messages = original_task_data.get("messages", [])
                if (
                    isinstance(messages, list)
                    and len(messages) > 1
                    and isinstance(messages[1], dict)
                ):
                    original_prompt = messages[1].get("text")

                # Fallback for prompt extraction remains the same
                if original_prompt is None:
                    original_prompt = original_task_data.get("prompt")

                # Correctly extract ideal response from RLHF structure
                messages = original_task_data.get("messages", [])
                if (
                    isinstance(messages, list)
                    and len(messages) > 2
                    and isinstance(messages[2], dict)
                ):
                    signal_data = messages[2].get("signal", {})
                    if isinstance(signal_data, dict):
                        original_ideal_response = signal_data.get("ideal_response")
                        if original_ideal_response:
                            logger.debug(
                                f"  Task [{task_id}]: Extracted ideal_response from messages[2]['signal']"
                            )

                # Fallback to direct extraction if RLHF extraction failed for ideal_response
                if original_ideal_response is None:
                    original_ideal_response = original_task_data.get("ideal_response")
                    if original_ideal_response:
                        logger.debug(
                            f"  Task [{task_id}]: Extracted ideal_response using fallback key"
                        )

                # Final logging for ideal response
                logger.debug(
                    f"  Task [{task_id}]: Final extracted original_ideal_response (type: {type(original_ideal_response)}): {str(original_ideal_response)[:200]}..."
                )
                original_final_answer_gt = None
                try:
                    # Try RLHF path first
                    original_final_answer_gt = original_task_data["messages"][2][
                        "signal"
                    ]["raw_preference_evaluation_form"][0]["human_input_value"]
                except (KeyError, IndexError, TypeError):
                    # Fallback to direct key if RLHF path fails or doesn't exist
                    logger.debug(
                        f"Could not extract final_answer from RLHF path for task {task_id}, falling back to direct key."
                    )
                    original_final_answer_gt = original_task_data.get("final_answer")
                original_metadata = original_task_data.get(
                    "metadata", {}
                )  # Already extracted by ingestion

                # Get model responses directly from the ingested data structure
                original_model_responses_dict = {}
                model_responses_list = original_task_data.get("model_responses", [])
                if isinstance(model_responses_list, list):
                    for resp_option in model_responses_list:
                        m_id = resp_option.get("model_id")
                        text = resp_option.get(
                            "response_text"
                        )  # Key is response_text in ingested data
                        if m_id and text is not None:
                            original_model_responses_dict[m_id] = text
                else:
                    logger.warning(
                        "Task [%s]: 'model_responses' in ingested data is not a list.",
                        task_id,
                    )

                # Get structured responses from the current evaluation result
                structured_ideal_response_obj = evaluation.get(
                    "structured_ideal_response"
                )
                structured_model_response_obj = evaluation.get(
                    "structured_model_response"
                )

                # Attempt to convert task_id to int for grouping key, fallback to string
                try:
                    processed_task_id: Union[int, str] = int(task_id)
                except (ValueError, TypeError):
                    processed_task_id = task_id  # Keep as string if conversion fails

                if processed_task_id not in grouped_results_map:
                    grouped_results_map[processed_task_id] = {
                        "task_id": processed_task_id,  # Store the potentially converted ID
                        "prompt": original_prompt,  # Use extracted prompt
                        "ideal_response": original_ideal_response,  # Use extracted ideal
                        "final_answer": original_final_answer_gt,  # Use extracted ground truth
                        "metadata": original_metadata,  # Use extracted metadata
                        "structured_ideal_response": None,  # Placeholder, populated below
                        "evaluations": [],
                    }

                # Assign Structured Ideal Response (if not already set for the task)
                # Check if it came from the evaluation results AND is not already set
                if (
                    structured_ideal_response_obj is not None
                    and grouped_results_map[processed_task_id].get(
                        "structured_ideal_response"
                    )  # Use processed_task_id
                    is None
                ):
                    if isinstance(structured_ideal_response_obj, dict):
                        grouped_results_map[processed_task_id][
                            "structured_ideal_response"
                        ] = (  # Use processed_task_id
                            structured_ideal_response_obj
                        )
                    else:
                        logger.warning(
                            f"Task [{task_id}]: structured_ideal_response from evaluation was not a dict, type: {type(structured_ideal_response_obj)}. Skipping assignment."
                        )  # Close parenthesis
                    # Remove erroneous lines below

                # Get the specific model response text from the extracted dictionary
                # Extract raw text from the structured object
                raw_model_response_text = None
                if isinstance(structured_model_response_obj, dict):
                    resp_content = structured_model_response_obj.get("response")
                    if isinstance(resp_content, str):
                        raw_model_response_text = resp_content
                    elif isinstance(
                        resp_content, dict
                    ):  # Handle case where it might be parsed JSON
                        raw_model_response_text = json.dumps(
                            resp_content
                        )  # Re-serialize if needed

                model_response_text = (
                    raw_model_response_text  # Use the text from the evaluated response
                )

                # Find Human Eval data (if present in original data under 'human_evaluations')
                human_evaluation_data = {}
                # Check if 'human_evaluations' key exists before iterating
                if "human_evaluations" in original_task_data and isinstance(
                    original_task_data["human_evaluations"], list
                ):
                    for human_eval in original_task_data.get("human_evaluations", []):
                        if human_eval.get("model_id") == model_id:
                            human_evaluation_data = {
                                k: v for k, v in human_eval.items() if k != "model_id"
                            }
                            break

                excluded_judge_keys = {
                    "task_id",
                    "model_id",
                    "evaluation_id",
                    "response_id",
                    "ideal_response_id",
                    "raw_judge_output",
                    "structured_ideal_response",
                    "structured_model_response",
                    "structuring_prompt",
                    "system_prompt",
                    "judging_prompt",
                    "structuring_api_calls",
                    "judging_api_calls",
                    "total_time_seconds",
                }
                judge_evaluation_data = {
                    k: v for k, v in evaluation.items() if k not in excluded_judge_keys
                }

                grouped_results_map[processed_task_id][
                    "evaluations"
                ].append(  # Use processed_task_id
                    {
                        "model_id": model_id,
                        "model_response": model_response_text,
                        "structured_model_response": structured_model_response_obj
                        if isinstance(structured_model_response_obj, dict)
                        else None,
                        "human_evaluation": human_evaluation_data,
                        "judge_evaluation": judge_evaluation_data,
                    }
                )
            # --- End of loop processing formatted evaluations ---

            final_results_list = list(grouped_results_map.values())

            # --- Calculate Final Summary Statistics ---
            num_tasks = len(grouped_results_map)
            logger.info(
                f"Identified {num_tasks} unique task IDs after grouping."
            )  # Added log
            avg_time_per_task = total_eval_time / num_tasks if num_tasks > 0 else 0.0
            avg_time_per_evaluation = (
                total_eval_time / total_evaluations_processed
                if total_evaluations_processed > 0
                else 0.0
            )
            avg_time_per_model = {
                model_id: (model_times.get(model_id, 0.0) / count if count > 0 else 0.0)
                for model_id, count in model_counts.items()
            }

            summary_stats = {
                "batch_id": cleaned_stem,
                "total_tasks_processed": num_tasks,
                "total_evaluations_processed": total_evaluations_processed,
                "total_structuring_api_calls": total_structuring_calls,
                "total_judging_api_calls": total_judging_calls,
                "total_evaluation_time_seconds": round(total_eval_time, 2),
                "average_time_per_task_seconds": round(avg_time_per_task, 2),
                "average_time_per_evaluation_seconds": round(
                    avg_time_per_evaluation, 2
                ),
                "average_time_per_model_seconds": {
                    model: round(avg, 2) for model, avg in avg_time_per_model.items()
                },
                "models_evaluated": sorted(list(model_counts.keys())),
            }

            final_output = {"summary": summary_stats, "results": final_results_list}

            # --- Save Final Combined Results ---
            logger.info(
                f"Writing final aggregated results to: {final_results_path}"
            )  # Added log
            with final_results_path.open("w", encoding="utf-8") as outfile:
                json.dump(final_output, outfile, indent=2, ensure_ascii=False)
            logger.info(
                f"Successfully wrote final results file: {final_results_path}"
            )  # Added log (modified existing one slightly for clarity)
            # logger.info( # Original log commented out for clarity, replaced above
            #     f"  Successfully combined results and wrote final output with summary to {final_results_path}"
            # )
            all_final_results_paths.append(
                str(final_results_path)
            )  # Append path only on full success for this file

        except (IOError, TypeError, json.JSONDecodeError, Exception) as e:
            logger.error(
                f"Error processing or saving results for {input_path.name}: {e}",
                exc_info=True,
            )
            overall_batch_success = False
            file_processing_success = False
            # No continue here, already outside the main processing block for the file

        if not file_processing_success:
            logger.warning(
                f"File {input_path.name} processing failed or was cancelled."
            )
            # overall_batch_success is already False

    # --- End of loop processing input files ---

    # --- Final Return ---
    if overall_batch_success:
        logger.info(
            f"Batch evaluation completed successfully for {len(all_final_results_paths)} file(s)."
        )
        return all_final_results_paths
    else:
        logger.error("Batch evaluation completed with errors or was cancelled.")
        return None  # Indicate incomplete batch
