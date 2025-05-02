"""
CogniBench Evaluation Runner Module.

Provides core functions to orchestrate the evaluation process for single tasks
or batches of tasks defined in input files. It integrates various components
like configuration loading, workflow execution, and output handling.
"""

import asyncio  # Added for async operations
import datetime  # Added for summary timestamp
import json
import logging
import os  # Added for directory creation
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# External libraries
from tqdm import tqdm

# Use ABSOLUTE imports for all core modules
from core.batch_processor import (
    create_batch_job,
    format_requests_to_jsonl,
    upload_batch_file,
)
from core.config import AppConfig
from core.llm_clients.openai_client import OpenAIClient

# Removed incorrect import of calculate_summary_statistics, combine_and_save_results,
# and format_evaluation_results from output_writer as they are not defined there.
# Functionality might need review if these were intended to be used.
from core.postprocessing import (
    perform_postprocessing,  # Changed to correct function name
)
from core.workflow import run_evaluation_workflow

logger = logging.getLogger(__name__)


# --- Helper Functions ---


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


async def run_single_task_evaluation_core(  # Changed to async def
    task_index: int,
    task_data: Dict[str, Any],
    config: AppConfig,
    use_structured: bool,  # Keep for now, though workflow might ignore if structuring mandatory
    output_jsonl_path: Optional[Path] = None,
    structured_ideal_cache: Optional[Dict[str, Any]] = None,
    stop_event: Optional[threading.Event] = None,
    aggregate_structuring: bool = False,  # Added for aggregation
) -> Union[
    Dict[str, Any], Tuple[List[Dict[str, Any]], bool]
]:  # Return aggregated dict or original tuple
    """
    Runs the evaluation workflow for a single task dictionary.
    Processes one task, evaluating multiple model responses against the ideal.
    If aggregate_structuring is True, returns a dictionary with aggregated requests.
    """
    logger.debug(f"Received task_data: {json.dumps(task_data, indent=2)}")
    # --- Extract data from task_data dictionary (handling original RLHF structure) ---
    task_id = task_data.get(
        "taskId", task_data.get("id", task_data.get("task_id", f"unknown_{task_index}"))
    )

    # Initialize variables for extracted data
    prompt_text = None
    ideal_response_text = None
    model_responses_raw = []
    correct_answer = task_data.get("correct_answer")  # Prefer direct key first

    # --- Primary Extraction (RLHF Format) ---
    messages = task_data.get("messages", [])
    if (
        isinstance(messages, list)
        and len(messages) > 1
        and isinstance(messages[1], dict)
    ):
        prompt_text = messages[1].get("text")
        if prompt_text:
            logger.debug(f"Extracted prompt from messages[1]['text']")

    if (
        isinstance(messages, list)
        and len(messages) > 2
        and isinstance(messages[2], dict)
    ):
        assistant_message = messages[2]
        signal_data = assistant_message.get("signal", {})
        if isinstance(signal_data, dict):
            ideal_response_text = signal_data.get("ideal_response")
            if ideal_response_text:
                logger.debug(
                    f"Extracted ideal_response from messages[2]['signal']['ideal_response']"
                )
            # Attempt to extract final_answer from signal if not found directly
            if correct_answer is None:
                raw_pref_eval = signal_data.get("raw_preference_evaluation_form", [])
                if isinstance(raw_pref_eval, list):
                    for item in raw_pref_eval:
                        if (
                            isinstance(item, dict)
                            and item.get("question") == "Final Answer"
                        ):
                            correct_answer = item.get("human_input_value")
                            if correct_answer:
                                logger.debug(
                                    "Extracted final_answer from raw_preference_evaluation_form"
                                )
                                break

        response_options = assistant_message.get("response_options", [])
        if isinstance(response_options, list):
            model_responses_raw = [
                {
                    "model_id": opt.get("model_id"),
                    "response_text": opt.get("text"),
                }
                for opt in response_options
                if opt.get("model_id") and opt.get("text")
            ]
            if model_responses_raw:
                logger.debug(
                    f"Extracted {len(model_responses_raw)} model responses from messages[2]['response_options']"
                )

    # --- Fallback Extraction (Ingested/Direct Format) ---
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
        if isinstance(fallback_responses, list):
            # Ensure fallback structure matches expected keys
            model_responses_raw = [
                {
                    "model_id": opt.get("model_id"),
                    "response_text": opt.get(
                        "response_text"
                    ),  # Check key name consistency
                }
                for opt in fallback_responses
                if opt.get("model_id")
                and opt.get("response_text")  # Check key name consistency
            ]
            if model_responses_raw:
                logger.debug(
                    f"Extracted {len(model_responses_raw)} model responses using fallback 'model_responses' key."
                )

    # Final check for correct_answer if still missing
    if correct_answer is None:
        correct_answer = task_data.get("final_answer")
        if correct_answer:
            logger.debug("Extracted final_answer from top-level 'final_answer' key.")

    model_responses = []
    if isinstance(model_responses_raw, list):
        for resp_option in model_responses_raw:
            model_id = resp_option.get("model_id")
            text = resp_option.get("response_text")
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

    if not prompt_text or not ideal_response_text:
        logger.warning(
            "Task [%s]: Skipping task due to missing prompt (%s) or ideal response (%s) after checking structures.",
            task_id,
            "found" if prompt_text else "missing",
            "found" if ideal_response_text else "missing",
        )
        return [], False  # Return empty list and False for failure

    if correct_answer is None:
        logger.warning(
            "Task [%s]: Missing ground truth 'correct_answer'. Final answer verification will be skipped.",
            task_id,
        )

    if structured_ideal_cache is None:
        structured_ideal_cache = {}

    for model_response in model_responses:
        if stop_event and stop_event.is_set():
            logger.warning(
                "Stop event detected during model response loop for task [%s]. Aborting task.",
                task_id,
            )
            task_success = False
            break

        response_text = model_response.get("response_text")
        model_id = model_response.get("model_id", "unknown_model")

        if response_text is None:
            logger.warning(
                "Task [%s] Model [%s]: Skipping response due to missing 'response_text'.",
                task_id,
                model_id,
            )
            continue

        # --- Call the core workflow (now async) ---
        # Workflow now always handles structuring internally
        result = await run_evaluation_workflow(  # Added await
            prompt=prompt_text,
            response=response_text,  # Pass raw text, workflow handles structuring
            ideal_response=ideal_response_text,  # Pass raw text, workflow handles structuring/caching
            correct_answer=correct_answer,  # Pass ground truth
            config=config.model_dump(),  # Pass config as dict
            task_id=str(task_id),  # Ensure task_id is string
            model_id=model_id,
            structured=False,  # Let workflow handle structuring internally
            output_jsonl_path=output_jsonl_path,
            structured_ideal_cache=structured_ideal_cache,
            aggregate_structuring=aggregate_structuring,  # Pass down aggregation flag
        )
        logger.debug(
            "Task [%s] Model [%s]: Received result from workflow: Type=%s, Value=%s",
            task_id,
            model_id,
            type(result),
            result,
        )

        # --- Handle Aggregated Result ---
        if (
            aggregate_structuring
            and isinstance(result, dict)
            and result.get("status") == "aggregated"
        ):
            # If aggregating, workflow returns the request dict directly
            # We need to add original data needed for mapping later
            model_req = result.get("model_request")
            ideal_req = result.get("ideal_request")
            aggregated_result_package = {"status": "aggregated"}
            original_data_for_mapping = {
                "task_id": str(task_id),
                "model_id": model_id,
                "prompt": prompt_text,
                "ideal_response": ideal_response_text,
                "correct_answer": correct_answer,
                # Add any other original data needed later (e.g., metadata)
                "metadata": task_data.get("metadata", {}),
            }
            if model_req:
                aggregated_result_package["model_request"] = model_req
                aggregated_result_package["model_original_data"] = (
                    original_data_for_mapping
                )
            if ideal_req:
                # Ideal request custom_id doesn't include model_id
                ideal_original_data = original_data_for_mapping.copy()
                del ideal_original_data["model_id"]
                aggregated_result_package["ideal_request"] = ideal_req
                aggregated_result_package["ideal_original_data"] = ideal_original_data

            return aggregated_result_package  # Return package containing requests and original data

        # --- Handle Normal Evaluation Result ---
        elif isinstance(result, dict):  # Ensure it's a dict before appending
            task_results.append(result)
            if result.get("status") != "success":
                task_success = False
                logger.error(
                    "Task [%s] Model [%s]: Workflow error: %s",
                    task_id,
                    model_id,
                    result.get("message", "Unknown error"),
                )
        else:
            # This case should ideally not happen if not aggregating
            logger.error(
                "Task [%s] Model [%s]: Workflow returned unexpected type: %s",
                task_id,
                model_id,
                type(result),
            )
            task_success = False

    # If aggregating, this return is only reached if loop finishes without returning aggregated package
    # (e.g., no model responses or stop event before first aggregation)
    if aggregate_structuring:
        return {"status": "aggregated", "model_request": None, "ideal_request": None}

    return task_results, task_success


async def run_batch_evaluation_core(  # Changed to async def
    config: AppConfig,
    output_dir: Path,
    # use_structured: bool, # Removed, structuring is now internal to workflow
    stop_event: Optional[threading.Event] = None,
) -> Optional[
    List[str]
]:  # Returns list of final result file paths on success (remains None for aggregate mode)
    """
    Orchestrates the full batch evaluation process for multiple input files.
    If aggregation is enabled, it collects structuring requests and submits them as an OpenAI Batch job.
    If aggregation is disabled, it runs the synchronous evaluation workflow for each task."""
    logger.info("Starting async batch evaluation core process...")
    logger.info(f"Output directory: {output_dir}")
    # logger.info(f"Using structured evaluation: {use_structured}") # Removed log
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)  # Ensure output dir exists early

    # Instantiate OpenAI Client (needed for batch operations)
    try:
        # Pass config object directly if client expects it
        openai_client = OpenAIClient(config=config)
        logger.info("OpenAIClient instantiated successfully.")
    except Exception as e:
        logger.error(f"Failed to instantiate OpenAIClient: {e}", exc_info=True)
        return None

    # --- Determine if we are aggregating based on config ---
    aggregate_mode = False  # Default to False
    if config.batch_settings and config.batch_settings.enabled:
        aggregate_mode = True
        logger.info("Batch Settings: Aggregation ENABLED.")
    else:
        logger.info(
            "Batch Settings: Aggregation DISABLED. Running synchronous evaluation."
        )

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
    structuring_requests: List[
        Dict[str, Any]
    ] = []  # Initialize list for aggregated requests
    intermediate_data_map: Dict[
        str, Dict[str, Any]
    ] = {}  # Map custom_id to original task_data needed for mapping

    # --- Loop Through Each Input File ---
    for input_path in input_file_paths:
        logger.info(f"--- Processing file: {input_path.name} ---")
        file_processing_success = True

        if stop_event and stop_event.is_set():
            logger.warning(
                "Stop event detected before processing file %s. Aborting batch.",
                input_path.name,
            )
            overall_batch_success = False
            break

        # --- 1. Define Output Paths ---
        try:
            original_stem = input_path.stem
            cleaned_stem = original_stem
            common_suffixes = ["_ingested", "_tasks", ".json"]
            for suffix in common_suffixes:
                if cleaned_stem.endswith(suffix):
                    cleaned_stem = cleaned_stem[: -len(suffix)]
            logger.info(f"Using cleaned input stem for output files: {cleaned_stem}")

            eval_jsonl_path = output_dir_path / f"{cleaned_stem}_evaluations.jsonl"
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
            continue

        # --- 2. Load Ingested Data ---
        try:
            if not input_path.is_file():
                logger.error(f"Input data file not found: {input_path}")
                raise FileNotFoundError(f"Input file not found: {input_path}")

            with input_path.open("r", encoding="utf-8") as f:
                loaded_data = json.load(f)

            if isinstance(loaded_data, dict) and "rlhf" in loaded_data:
                evaluation_tasks: List[Dict[str, Any]] = loaded_data.get("rlhf", [])
                if not isinstance(evaluation_tasks, list):
                    raise TypeError(
                        f"Data under 'rlhf' key in {input_path.name} is not a list."
                    )
            elif isinstance(loaded_data, list):
                evaluation_tasks = loaded_data
            else:
                raise TypeError(
                    f"Unexpected JSON structure in {input_path.name}. Expected list or dict with 'rlhf'."
                )

            logger.info(
                f"  Loaded {len(evaluation_tasks)} tasks from {input_path.name}"
            )
        except (IOError, json.JSONDecodeError, TypeError, FileNotFoundError) as e:
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

        # --- 3. Run Evaluations/Aggregation Task by Task ---
        try:
            structured_ideal_cache = {}
            file_overall_success = True
            total_tasks = len(evaluation_tasks)
            all_evaluation_results_list: List[
                Dict[str, Any]
            ] = []  # Used only if not aggregating

            # Clear the JSONL file only if NOT aggregating
            if not aggregate_mode and eval_jsonl_path.exists():
                logger.warning(
                    f"Output JSONL file {eval_jsonl_path} already exists. Overwriting."
                )
                try:
                    eval_jsonl_path.unlink()
                except OSError as e:
                    logger.error(
                        f"Failed to remove existing JSONL file {eval_jsonl_path}: {e}"
                    )
                    raise

            logger.info(
                f"  Starting evaluation/aggregation loop for {total_tasks} tasks..."
            )
            for i, task_data in enumerate(
                tqdm(
                    evaluation_tasks,
                    desc=f"Processing Tasks ({input_path.name})",
                    file=sys.stdout,
                    leave=False,
                )
            ):
                if stop_event and stop_event.is_set():
                    logger.warning(
                        "Stop event detected during task processing loop. Aborting."
                    )
                    file_overall_success = False
                    break

                task_id = task_data.get("task_id", f"unknown_task_{i}")

                # Call the now async single task runner
                single_task_result = await run_single_task_evaluation_core(  # Added await
                    task_index=i,
                    task_data=task_data,
                    config=config,
                    use_structured=False,  # Pass False, workflow handles structuring
                    output_jsonl_path=eval_jsonl_path
                    if not aggregate_mode
                    else None,  # Only pass if needed
                    structured_ideal_cache=structured_ideal_cache,
                    stop_event=stop_event,
                    aggregate_structuring=aggregate_mode,  # Pass mode flag
                )

                # --- Handle Aggregated Requests ---
                if aggregate_mode:
                    if (
                        isinstance(single_task_result, dict)
                        and single_task_result.get("status") == "aggregated"
                    ):
                        model_req = single_task_result.get("model_request")
                        ideal_req = single_task_result.get("ideal_request")
                        # Store original data keyed by custom_id for later mapping
                        if model_req and "custom_id" in model_req:
                            structuring_requests.append(model_req)
                            intermediate_data_map[model_req["custom_id"]] = (
                                single_task_result.get("model_original_data", {})
                            )
                        if ideal_req and "custom_id" in ideal_req:
                            structuring_requests.append(ideal_req)
                            intermediate_data_map[ideal_req["custom_id"]] = (
                                single_task_result.get("ideal_original_data", {})
                            )
                    else:
                        logger.error(
                            f"Task [{task_id}]: Expected aggregated result but got: {single_task_result}"
                        )
                        file_overall_success = False
                    # Continue to next task when aggregating, don't process results now
                    continue

                # --- Handle Normal Evaluation Results (if not aggregating) ---
                elif isinstance(single_task_result, tuple):  # Added elif for clarity
                    task_results, task_success = single_task_result
                    all_evaluation_results_list.extend(task_results)
                    if not task_success:
                        file_overall_success = False
                        # Error already logged in run_single_task_evaluation_core
                else:
                    # This shouldn't happen if aggregate_structuring=False
                    logger.error(
                        f"Task [{task_id}]: Expected tuple result but got: {single_task_result}"
                    )
                    file_overall_success = False

            # --- 4. Post-processing for this file (Only if NOT aggregating) ---
            if not aggregate_mode:
                if file_overall_success:
                    logger.info(
                        f"--- Finished evaluation loop for {input_path.name}. Starting post-processing. ---"
                    )
                    try:
                        # --- 4a. Read Evaluation Results from JSONL ---
                        parsed_jsonl_data = []
                        if not eval_jsonl_path.is_file():
                            logger.error(
                                f"Evaluation results file not found: {eval_jsonl_path}. Cannot generate final results."
                            )
                            raise FileNotFoundError(f"Missing {eval_jsonl_path}")

                        with eval_jsonl_path.open("r", encoding="utf-8") as f_jsonl:
                            for line_num, line in enumerate(f_jsonl, 1):
                                try:
                                    if line.strip():
                                        parsed_jsonl_data.append(json.loads(line))
                                except json.JSONDecodeError as json_err:
                                    logger.error(
                                        f"Error decoding JSON on line {line_num} in {eval_jsonl_path}: {json_err}"
                                    )
                                    raise  # Re-raise to indicate failure
                        logger.info(
                            f"  Successfully read {len(parsed_jsonl_data)} evaluation records from {eval_jsonl_path}"
                        )

                        # --- 4b. Combine Original Data with Evaluations (Revised Structure) ---
                        original_tasks_map = {}
                        for idx, task in enumerate(evaluation_tasks):
                            task_id = str(
                                task.get(
                                    "taskId",
                                    task.get(
                                        "id", task.get("task_id", f"unknown_{idx}")
                                    ),
                                )
                            )
                            original_tasks_map[task_id] = task

                        evaluations_by_task = {}
                        models_evaluated = set()
                        total_structuring_calls = 0
                        total_judging_calls = 0
                        total_time = 0.0
                        model_times = {}
                        model_eval_counts = {}

                        for eval_record in parsed_jsonl_data:
                            task_id = str(eval_record.get("task_id"))
                            model_id = eval_record.get("model_id")
                            if model_id:
                                models_evaluated.add(model_id)
                                model_eval_counts[model_id] = (
                                    model_eval_counts.get(model_id, 0) + 1
                                )
                            if task_id not in evaluations_by_task:
                                evaluations_by_task[task_id] = []
                            evaluations_by_task[task_id].append(eval_record)
                            total_structuring_calls += (
                                eval_record.get("structuring_api_calls") or 0
                            )
                            total_judging_calls += (
                                eval_record.get("judging_api_calls") or 0
                            )
                            eval_time = eval_record.get("total_time_seconds")
                            if eval_time is not None:
                                total_time += eval_time
                                if model_id:
                                    model_times[model_id] = (
                                        model_times.get(model_id, 0.0) + eval_time
                                    )

                        combined_results_structured = []
                        for task_id, original_task in original_tasks_map.items():
                            task_evals_raw = evaluations_by_task.get(task_id, [])
                            if not task_evals_raw:
                                continue

                            # Extract structured ideal response from the first evaluation for this task
                            structured_ideal_resp = task_evals_raw[0].get(
                                "structured_ideal_response"
                            )

                            processed_evaluations = []
                            for eval_record in task_evals_raw:
                                model_id = eval_record.get("model_id")
                                # Find corresponding human eval data
                                human_eval_data = {}
                                for he in original_task.get("human_evaluations", []):
                                    if he.get("model_id") == model_id:
                                        human_eval_data = he
                                        break

                                # Keys to exclude from the top level of eval_record when creating judge_evaluation
                                exclude_keys = {
                                    "task_id",
                                    "model_id",
                                    "structured_model_response",
                                    "structured_ideal_response",
                                    "human_evaluation",
                                }
                                judge_eval_data = {
                                    k: v
                                    for k, v in eval_record.items()
                                    if k not in exclude_keys
                                }

                                # Find the original raw model response text if available
                                raw_model_response_text = None
                                for mr in original_task.get("model_responses", []):
                                    if mr.get("model_id") == model_id:
                                        raw_model_response_text = mr.get(
                                            "response_text"
                                        )
                                        break

                                processed_evaluations.append(
                                    {
                                        "model_id": model_id,
                                        "model_response": raw_model_response_text,  # Add raw response text
                                        "structured_model_response": eval_record.get(
                                            "structured_model_response"
                                        ),
                                        "human_evaluation": human_eval_data,
                                        "judge_evaluation": judge_eval_data,
                                    }
                                )

                            combined_task_data = {
                                "task_id": int(task_id)
                                if task_id.isdigit()
                                else task_id,
                                "prompt": original_task.get("prompt"),
                                "ideal_response": original_task.get("ideal_response"),
                                "final_answer": original_task.get("final_answer"),
                                "metadata": original_task.get("metadata", {}),
                                "structured_ideal_response": structured_ideal_resp,
                                "evaluations": processed_evaluations,
                            }
                            combined_results_structured.append(combined_task_data)

                        # --- 4c. Calculate Summary Statistics (Revised) ---
                        avg_time_per_task = (
                            total_time / len(evaluation_tasks)
                            if evaluation_tasks
                            else 0
                        )
                        avg_time_per_eval = (
                            total_time / len(parsed_jsonl_data)
                            if parsed_jsonl_data
                            else 0
                        )
                        avg_time_per_model = {
                            model: round(
                                model_times.get(model, 0.0)
                                / model_eval_counts.get(model, 1),
                                2,
                            )
                            for model in models_evaluated
                        }

                        # Get structuring/judging model from config (should be named 'config' in this scope)
                        structuring_model = None
                        judging_model = None
                        try:
                            structuring_model = getattr(config.structuring_settings.llm_client, "model", None)
                        except Exception:
                            structuring_model = None
                        try:
                            judging_model = getattr(config.evaluation_settings.llm_client, "model", None)
                        except Exception:
                            judging_model = None

                        summary = {
                            "batch_id": cleaned_stem,  # Use cleaned stem as batch identifier
                            "total_tasks_processed": len(evaluation_tasks),
                            "total_evaluations_processed": len(parsed_jsonl_data),
                            "total_structuring_api_calls": total_structuring_calls,
                            "total_judging_api_calls": total_judging_calls,
                            "total_evaluation_time_seconds": round(total_time, 2),
                            "average_time_per_task_seconds": round(
                                avg_time_per_task, 2
                            ),
                            "average_time_per_evaluation_seconds": round(
                                avg_time_per_eval, 2
                            ),
                            "average_time_per_model_seconds": avg_time_per_model,
                            "models_evaluated": sorted(list(models_evaluated)),
                            "structuring_model": structuring_model or "N/A",
                            "judging_model": judging_model or "N/A",
                        }

                        # --- 4d. Assemble and Save Final JSON (Revised Structure) ---
                        final_data = {
                            "summary": summary,
                            "results": combined_results_structured,
                        }

                        with final_results_path.open("w", encoding="utf-8") as f_final:
                            json.dump(final_data, f_final, indent=2, ensure_ascii=False)

                        logger.info(
                            f"  Combined original tasks and evaluations saved to: {final_results_path}"
                        )
                        if (
                            str(eval_jsonl_path) in all_final_results_paths
                        ):  # Should not happen now
                            all_final_results_paths.remove(str(eval_jsonl_path))
                        all_final_results_paths.append(
                            str(final_results_path)
                        )  # Add correct path

                    except Exception as e:
                        logger.error(
                            f"Failed during result aggregation and saving step for {input_path.name}: {e}",
                            exc_info=True,
                        )
                        overall_batch_success = False
                        file_processing_success = False

                else:  # file_overall_success was False
                    logger.error(
                        f"--- Evaluation loop for {input_path.name} failed. Skipping post-processing for this file. ---"
                    )
                    overall_batch_success = False
                    file_processing_success = False

        except Exception as e:
            logger.error(
                f"Unexpected error during evaluation loop or post-processing for {input_path.name}: {e}",
                exc_info=True,
            )
            overall_batch_success = False
            file_processing_success = False

    # --- Submit Batch Job (if aggregating) ---
    if aggregate_mode:
        logger.info(
            f"Aggregated {len(structuring_requests)} structuring requests across all files."
        )
        logger.info(
            "Attempting to submit aggregated structuring requests as OpenAI Batch job..."
        )
        if not structuring_requests:
            logger.warning(
                "No structuring requests were aggregated. Skipping batch submission."
            )
            # Consider if this is an overall failure or just no work to do
            # overall_batch_success = False # Optional
        else:
            try:
                # 1. Format requests to JSONL
                logger.info(
                    f"Formatting {len(structuring_requests)} requests to JSONL..."
                )
                jsonl_content = format_requests_to_jsonl(structuring_requests)
                if not jsonl_content:
                    logger.error(
                        "Failed to format requests into JSONL content. Aborting batch submission."
                    )
                    overall_batch_success = False
                else:
                    logger.info("Successfully formatted requests to JSONL.")

                    # 2. Upload batch file
                    logger.info("Uploading batch file to OpenAI...")
                    file_id = await upload_batch_file(openai_client, jsonl_content)
                    if file_id:
                        logger.info(
                            f"Successfully uploaded batch file. File ID: {file_id}"
                        )

                        # 3. Create batch job
                        logger.info(f"Creating batch job with File ID: {file_id}...")
                        # Use default endpoint/window or get from config if added
                        batch_job = await create_batch_job(openai_client, file_id)
                        if batch_job:
                            batch_id = batch_job.id
                            logger.info(
                                f"Structuring batch job submitted successfully. Batch ID: {batch_id}"
                            )

                            # --- Task 13: Save Intermediate Data ---
                            intermediate_data_dir_path = Path(
                                config.batch_settings.intermediate_data_dir
                            )
                            try:
                                os.makedirs(intermediate_data_dir_path, exist_ok=True)
                                intermediate_file_path = (
                                    intermediate_data_dir_path
                                    / f"intermediate_data_{batch_id}.json"
                                )
                                with open(
                                    intermediate_file_path, "w", encoding="utf-8"
                                ) as f:
                                    json.dump(
                                        intermediate_data_map, f, indent=4
                                    )  # Save the map
                                logger.info(
                                    f"Successfully saved intermediate data map to: {intermediate_file_path}"
                                )
                            except (OSError, IOError, TypeError) as e:
                                logger.error(
                                    f"Failed to save intermediate data map for batch {batch_id}: {e}",
                                    exc_info=True,
                                )
                                overall_batch_success = (
                                    False  # Fail if map can't be saved
                                )

                        else:
                            logger.error("Failed to create batch job.")
                            overall_batch_success = False
                    else:
                        logger.error("Failed to upload batch file.")
                        overall_batch_success = False

            except Exception as e:
                logger.error(
                    f"An error occurred during batch submission: {e}", exc_info=True
                )
                overall_batch_success = False

    # --- Final Outcome ---
    if overall_batch_success:
        if aggregate_mode:
            logger.info("--- Batch Aggregation Core Process COMPLETED Successfully ---")
            # Return None as no final *evaluation* files were generated in this mode
            return None
        else:
            logger.info("--- Batch Evaluation Core Process COMPLETED Successfully ---")
            # Return the paths to the final combined result files
            return all_final_results_paths
    else:
        mode_msg = "Aggregation" if aggregate_mode else "Evaluation"
        logger.error(f"--- Batch {mode_msg} Core Process FAILED ---")
        return None  # Indicate failure


# --- Example Usage (if run directly) ---
# (Keep commented out as it's not the primary execution method)
# if __name__ == "__main__":
#     from .log_setup import setup_logging
#     import argparse
#
#     parser = argparse.ArgumentParser(description="Run Core Batch Evaluation Logic Directly")
#     # ... (rest of example usage code) ...
#     # ... (rest of example usage code) ...
#     # ... (rest of example usage code) ...
#     # ... (rest of example usage code) ...
#     # ... (rest of example usage code) ...
#     # ... (rest of example usage code) ...
