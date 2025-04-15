# CogniBench/core/evaluation_runner.py

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
    task_id = task_data.get("taskId", "unknown_task")  # Use 'taskId' from input JSON
    prompt_text = None
    ideal_response_text = None
    model_responses_raw = []
    structured_ideal_response = None  # Not present in this input format
    correct_answer = (
        None  # Not directly present, might be in signal.raw_preference_evaluation_form
    )

    # Extract data from messages array
    messages = task_data.get("messages", [])
    for message in messages:
        role = message.get("role")
        if role == "user":
            prompt_text = message.get("text")
            # Potentially extract metadata like subject/complexity from prompt_evaluation here if needed
        elif role == "assistant":
            model_responses_raw = message.get("response_options", [])
            signal_data = message.get("signal", {})
            ideal_response_text = signal_data.get("ideal_response")
            # Extract final_answer ground truth if available
            pref_eval_form = signal_data.get("raw_preference_evaluation_form", [])
            for item in pref_eval_form:
                if item.get("question") == "Final Answer":
                    correct_answer = item.get("human_input_value")
                    break

    # Transform model_responses_raw into the expected format
    model_responses = []
    for resp_option in model_responses_raw:
        model_id = resp_option.get("model_id")
        text = resp_option.get("text")
        if model_id and text is not None:  # Ensure both exist
            model_responses.append({"model_id": model_id, "response_text": text})
        else:
            logger.warning(
                "Task [%s]: Skipping invalid model response option: %s",
                task_id,
                resp_option,
            )

    task_results = []
    task_success = True

    # Check if essential data was found
    if not prompt_text or not ideal_response_text:
        logger.warning(
            "Task [%s]: Skipping task due to missing prompt ('%s') or ideal_response ('%s') in messages.",
            task_id,
            "found" if prompt_text else "missing",
            "found" if ideal_response_text else "missing",
        )  # Removed extra task_id and closed parenthesis
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

        if response_input is None or ideal_input is None:
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
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Using structured evaluation: {use_structured}")

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
            base_stem = input_path.stem
            common_suffixes = ["_ingested", "_tasks", ".json"]
            for suffix in common_suffixes:
                if base_stem.endswith(suffix):
                    base_stem = base_stem[: -len(suffix)]

            output_dir.mkdir(parents=True, exist_ok=True)

            eval_jsonl_path = output_dir / f"{base_stem}_evaluations.jsonl"
            eval_json_path = output_dir / f"{base_stem}_evaluations_formatted.json"
            final_results_path = output_dir / f"{base_stem}_final_results.json"

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

            # Check if JSONL was created, even if tasks failed (it might have partial results)
            if not eval_jsonl_path.is_file():
                logger.error(
                    f"Evaluation loop finished, but output JSONL file {eval_jsonl_path} was not created."
                )
                raise IOError(
                    f"JSONL file not created: {eval_jsonl_path}"
                )  # Raise to be caught

            # --- 4. Format JSONL Output ---
            logger.info(f"  Formatting evaluation results from {eval_jsonl_path}...")
            formatted_evaluations_list = []
            # This section needs its own try/except as it reads the potentially incomplete JSONL
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

            # --- 5. Combine Ingested Data and Formatted Evaluations ---
            logger.info(
                "  Combining ingested data with formatted evaluation results..."
            )
            # This section also needs its own try/except

            # Load ingested data again and extract the task list correctly
            with input_path.open("r", encoding="utf-8") as f:
                loaded_ingested_data = json.load(f)

            if (
                isinstance(loaded_ingested_data, dict)
                and "rlhf" in loaded_ingested_data
            ):
                ingested_tasks_list = loaded_ingested_data.get("rlhf", [])
                if not isinstance(ingested_tasks_list, list):
                    logger.error(
                        f"Data under 'rlhf' key in {input_path.name} is not a list (for combination step)."
                    )
                    raise TypeError("Expected a list under 'rlhf' key.")
            elif isinstance(loaded_ingested_data, list):
                ingested_tasks_list = loaded_ingested_data
            else:
                logger.error(
                    f"Unexpected JSON structure in {input_path.name} (for combination step)."
                )
                raise TypeError("Unexpected JSON structure.")

            # Now build the map from the extracted list
            ingested_data_map = {
                task.get("taskId", f"unknown_{i}"): task
                for i, task in enumerate(ingested_tasks_list)
            }  # Use .get() and taskId

            grouped_results_map = {}
            total_structuring_calls = 0
            total_judging_calls = 0
            total_eval_time = 0.0
            total_evaluations_processed = 0
            model_times = {}
            model_counts = {}

            for evaluation in formatted_evaluations_list:
                # (Logic for accumulating metrics and combining data as before)
                total_evaluations_processed += 1
                task_id = evaluation.get("task_id")
                model_id = evaluation.get("model_id")
                evaluation_id = evaluation.get("evaluation_id")

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

                # Extract prompt, ideal response, etc., from original_task_data messages
                original_prompt = None
                original_ideal_response = None
                original_model_responses_dict = {}  # Store model_id -> text
                original_final_answer_gt = None
                original_metadata = original_task_data.get(
                    "metadata", {}
                )  # Keep original metadata

                messages = original_task_data.get("messages", [])
                for message in messages:
                    role = message.get("role")
                    if role == "user":
                        original_prompt = message.get("text")
                        # Extract metadata from prompt_evaluation if needed
                        prompt_eval = message.get("prompt_evaluation", [])
                        for item in prompt_eval:
                            q = item.get("question", "").lower()
                            v = item.get("human_input_value")
                            if q and v:
                                original_metadata[q] = v  # Add subject/complexity etc.
                    elif role == "assistant":
                        signal_data = message.get("signal", {})
                        original_ideal_response = signal_data.get("ideal_response")
                        # Extract ground truth final answer
                        pref_eval_form = signal_data.get(
                            "raw_preference_evaluation_form", []
                        )
                        for item in pref_eval_form:
                            if item.get("question") == "Final Answer":
                                original_final_answer_gt = item.get("human_input_value")
                                break
                        # Extract model responses
                        for resp_option in message.get("response_options", []):
                            m_id = resp_option.get("model_id")
                            text = resp_option.get("text")
                            if m_id and text is not None:
                                original_model_responses_dict[m_id] = text

                # Get structured responses from the current evaluation result
                structured_ideal_response_obj = evaluation.get(
                    "structured_ideal_response"
                )
                structured_model_response_obj = evaluation.get(
                    "structured_model_response"
                )

                if task_id not in grouped_results_map:
                    grouped_results_map[task_id] = {
                        "task_id": task_id,
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
                    and grouped_results_map[task_id].get("structured_ideal_response")
                    is None
                ):
                    if isinstance(structured_ideal_response_obj, dict):
                        grouped_results_map[task_id]["structured_ideal_response"] = (
                            structured_ideal_response_obj
                        )
                    else:
                        logger.warning(
                            f"Task [{task_id}]: structured_ideal_response from evaluation was not a dict, type: {type(structured_ideal_response_obj)}. Skipping assignment."
                        )  # Close parenthesis
                    # Remove erroneous lines below

                if (
                    structured_ideal_response_obj is not None
                    and grouped_results_map[task_id].get("structured_ideal_response")
                    is None
                ):
                    if isinstance(structured_ideal_response_obj, dict):
                        grouped_results_map[task_id]["structured_ideal_response"] = (
                            structured_ideal_response_obj
                        )
                    # else: log warning (omitted for brevity)

                # Get the specific model response text from the extracted dictionary
                model_response_text = original_model_responses_dict.get(model_id)

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

                grouped_results_map[task_id]["evaluations"].append(
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
                "batch_id": base_stem,
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
            with final_results_path.open("w", encoding="utf-8") as outfile:
                json.dump(final_output, outfile, indent=2, ensure_ascii=False)
            logger.info(
                f"  Successfully combined results and wrote final output with summary to {final_results_path}"
            )
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
