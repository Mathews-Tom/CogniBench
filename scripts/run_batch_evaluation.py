import argparse
import json  # Import json for conversion step
import logging
import re  # Import re for snake_case conversion
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# --- Logging Setup ---
# Configure logging directly here to ensure DEBUG level is set for the main script
# This bypasses potential conflicts with setup_logging calls elsewhere.
# --- Logging Setup ---
# Add project root to sys.path to allow absolute import of core modules
APP_DIR = Path(__file__).parent
COGNIBENCH_ROOT = APP_DIR.parent
if str(COGNIBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(COGNIBENCH_ROOT))

from core.log_setup import setup_logging

# Setup logging using the centralized function
# Pass DEBUG level specifically for this verbose script if needed
setup_logging(log_level=logging.DEBUG)
logger = logging.getLogger("backend")
logger.info("Logging setup complete via core.log_setup.")
# --- End Logging Setup ---


# --- Helper Function for Parsing Embedded JSON ---
def _parse_json_string(
    json_string: Optional[str], task_id: str, field_name: str
) -> Union[Dict[str, Any], List[Any], str, None]:
    """Attempts to parse a JSON string, potentially embedded in markdown code fences.

    Args:
        json_string: The string value to parse.
        task_id: The task ID for logging context.
        field_name: The name of the field being parsed for logging context.

    Returns:
        The parsed Python object (dict or list) if successful,
        otherwise the original string. Returns None if input is None.
    """
    if json_string is None:
        return None
    if not isinstance(json_string, str):
        logger.warning(
            "Task [%s]: Field '%s' was not a string, skipping parsing. Value: %s",
            task_id,
            field_name,
            json_string,
        )
        return json_string  # Return the original non-string value

    # Attempt to remove markdown fences (```json ... ``` or ``` ... ```)
    extracted_string = json_string.strip()
    if extracted_string.startswith("```json"):
        extracted_string = extracted_string[len("```json") :]
    elif extracted_string.startswith("```"):
        extracted_string = extracted_string[len("```") :]

    if extracted_string.endswith("```"):
        extracted_string = extracted_string[: -len("```")]

    extracted_string = extracted_string.strip()

    # Handle potential escape sequences common in LLM outputs
    # Example: Convert \" to " - Add more rules if needed
    # extracted_string = extracted_string.replace('\\"', '"')
    # Note: json.loads usually handles standard JSON escapes correctly.
    # Be cautious adding custom replacements that might break valid JSON.

    try:
        # Attempt to parse the extracted string
        parsed_data = json.loads(extracted_string)
        # Optional: Log success at DEBUG level if needed
        # logger.debug("Task [%s]: Successfully parsed JSON for field '%s'.", task_id, field_name)
        return parsed_data
    except json.JSONDecodeError as e:
        logger.warning(
            "Task [%s]: Failed to parse JSON for field '%s'. Error: %s. Keeping original string: %s",
            task_id,
            field_name,
            e,
            json_string,  # Log the original string that failed
            exc_info=False,  # Don't log full traceback for expected errors
        )
        return json_string  # Return the original string on failure
    except Exception as e:  # Catch unexpected errors during parsing
        logger.error(
            "Task [%s]: Unexpected error parsing JSON for field '%s'. Error: %s. Keeping original string: %s",
            task_id,
            field_name,
            e,
            json_string,
            exc_info=True,  # Log traceback for unexpected errors
        )
        return json_string  # Return the original string on failure


# Removed redundant/commented out logging setup attempts


def run_command(command_list):
    """Runs a command using subprocess and handles errors."""
    logger.debug("Running command: %s", " ".join(command_list))  # Changed to debug
    try:
        process = subprocess.run(
            command_list, check=True, capture_output=True, text=True, encoding="utf-8"
        )
        logger.info("Command successful.")
        # Print stdout/stderr for debugging if needed, but be careful
        # if capturing stdout is critical (like for the ingestion script)
        # Log stdout/stderr only if needed for debugging, at DEBUG level
        if process.stdout:
            logger.debug("stdout:\n%s", process.stdout)
        if process.stderr:
            logger.debug("stderr:\n%s", process.stderr)  # Log stderr to the log file
        return process
    except subprocess.CalledProcessError as e:
        logger.error("Error running command: %s", " ".join(command_list))
        logger.error("Return code: %s", e.returncode)
        if e.stdout:
            logger.error("stdout:\n%s", e.stdout)
        if e.stderr:
            logger.error("stderr:\n%s", e.stderr)
        return None
    except FileNotFoundError:
        logger.error("Command not found (%s)", command_list[0])
        return None


# --- Helper Function for Snake Case ---
def to_snake_case(name):
    """Converts CamelCase, PascalCase, or space-separated string to snake_case."""
    if " " in name:  # Handle space-separated first
        return name.strip().lower().replace(" ", "_")
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def keys_to_snake_case(obj):
    """Recursively converts dictionary keys and specific string values to snake_case."""
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            # Convert the key itself to snake_case
            new_key = to_snake_case(key)
            # Recursively process the value
            new_dict[new_key] = keys_to_snake_case(value)
        return new_dict
    elif isinstance(obj, list):
        # Recursively process items in a list
        return [keys_to_snake_case(item) for item in obj]
    # Optional: Convert specific string values if needed (e.g., model IDs)
    # elif isinstance(obj, str):
    #     # Example: Convert string values that might represent keys elsewhere
    #     # return to_snake_case(obj)
    #     return obj # Keep original string values by default
    else:
        # Return non-dict/list types as is
        return obj


def load_config(config_path: Path) -> Optional[Dict[str, Any]]:
    """Loads configuration from a YAML file."""
    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                logger.error(
                    "Configuration file '%s' did not load as a dictionary.", config_path
                )
                return None
            return config
    except ImportError:
        logger.error(
            "PyYAML is required to load config.yaml. Please install it (`uv pip install pyyaml`)."
        )
        return None
    except Exception as e:
        logger.error("Error loading config file %s", config_path, exc_info=True)
        return None


def validate_config(config: Dict[str, Any]) -> bool:
    """Performs basic validation on the loaded configuration dictionary."""
    if not config:  # Check if config is None or empty
        logger.error(
            "Config validation failed: Configuration is empty or failed to load."
        )
        return False

    required_sections = ["llm_client", "evaluation_settings"]
    for section in required_sections:
        if section not in config or not isinstance(config[section], dict):
            logger.error(
                "Config validation failed: Missing or invalid section '%s'.", section
            )
            return False

    eval_settings = config["evaluation_settings"]
    required_eval_keys = [
        "judge_model",
        "prompt_template",
        "expected_criteria",
        "allowed_scores",
    ]
    for key in required_eval_keys:
        if key not in eval_settings:
            logger.error(
                "Config validation failed: Missing key '%s' in 'evaluation_settings'.",
                key,
            )
            return False
        # Specific type checks
        if key in ["expected_criteria", "allowed_scores"] and not isinstance(
            eval_settings[key], list
        ):
            logger.error(
                "Config validation failed: Key '%s' in 'evaluation_settings' must be a list.",
                key,
            )
            return False
        elif key in ["judge_model", "prompt_template"] and not isinstance(
            eval_settings[key], str
        ):
            logger.error(
                "Config validation failed: Key '%s' in 'evaluation_settings' must be a string.",
                key,
            )
            return False

    # Could add more checks (e.g., non-empty lists/strings) if needed
    logger.info("Configuration validation successful.")
    return True


if __name__ == "__main__":
    # Logging is configured via basicConfig at the top of the script
    # setup_logging() # Removed call
    logger.info("Starting end-to-end batch evaluation script.")
    parser = argparse.ArgumentParser(
        description="Run end-to-end CogniBench evaluation from a raw batch file."
    )
    parser.add_argument(
        "input_batch_file", type=str, help="Path to the input raw RLHF JSON batch file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the CogniBench configuration file (used by scripts/run_single_evaluation.py).",
    )
    # Add other arguments needed by run_single_evaluation.py if necessary

    args = parser.parse_args()

    input_batch_path = Path(args.input_batch_file)
    config_path = Path(args.config)

    if not input_batch_path.is_file():
        logger.error("Input batch file not found at %s", input_batch_path)
        sys.exit(1)

    if not config_path.is_file():
        logger.error("Config file not found at %s", config_path)
        sys.exit(1)

    # --- Load and Validate Config Early ---
    logger.info("Loading configuration from %s...", config_path)
    config = load_config(config_path)
    if not validate_config(config):  # Validate config after loading
        logger.error("Invalid configuration file. Aborting.")
        sys.exit(1)

    # --- Determine and Create Output Directory ---
    batch_stem = input_batch_path.stem  # e.g., "Batch-001"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_subdir_name = f"{batch_stem}_{timestamp}"
    output_dir = Path(__file__).parent.parent / "data" / output_subdir_name
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created output directory: %s", output_dir)
    except OSError as e:
        logger.error("Failed to create output directory %s: %s", output_dir, e)
        sys.exit(1)

    logger.debug("--- Starting Step 1: Ingestion ---")  # Changed to debug
    ingestion_script_path = Path(__file__).parent / "ingest_rlhf_data.py"
    ingestion_command = [
        sys.executable,  # Use the same python interpreter
        str(ingestion_script_path),
        str(input_batch_path),
        "--output-dir",  # Pass the determined output directory
        str(output_dir),
    ]

    ingestion_process = run_command(ingestion_command)

    if ingestion_process is None or ingestion_process.returncode != 0:
        logger.error("Ingestion script failed. Aborting.")
        sys.exit(1)

    # Capture the output path from stdout (remove potential trailing newline)
    # Capture the output path from stdout and strip thoroughly
    ingested_file_path_str = ingestion_process.stdout.strip()
    if not ingested_file_path_str:
        logger.error("Ingestion script did not output a file path.")
        sys.exit(1)

    ingested_file_path = Path(ingested_file_path_str)
    if not ingested_file_path.is_file():
        logger.error(
            "Ingested file path reported but not found: %s", ingested_file_path
        )
        sys.exit(1)

    logger.debug("--- Finished Step 1: Ingestion ---")  # Changed to debug
    logger.info(
        "Ingestion successful. Ingested data at: %s", ingested_file_path
    )  # Keep INFO for path

    # --- Define Other Output Filenames (within the subdirectory) ---
    # output_dir, batch_stem, and timestamp are already defined
    eval_jsonl_filename = (
        f"{batch_stem}_evaluations.jsonl"  # Simpler name within subdir
    )
    eval_json_filename = (
        f"{batch_stem}_evaluations_formatted.json"  # Simpler name within subdir
    )
    final_results_filename = (
        f"{batch_stem}_final_results.json"  # Simpler name within subdir
    )

    eval_jsonl_path = output_dir / eval_jsonl_filename
    eval_json_path = output_dir / eval_json_filename
    final_results_path = (
        output_dir / final_results_filename
    )  # Path for combined results

    logger.info("Evaluation results (.jsonl) will be saved to: %s", eval_jsonl_path)
    logger.info("Formatted evaluations (.json) will be saved to: %s", eval_json_path)
    logger.info(
        "Final combined results (.json) will be saved to: %s", final_results_path
    )

    logger.debug("--- Starting Step 2: Evaluation ---")  # Changed to debug
    # Path to the single evaluation script within the same scripts directory
    evaluation_script_path = Path(__file__).parent / "run_single_evaluation.py"
    evaluation_command = [
        sys.executable,  # Keep using the same python interpreter
        "-m",  # Add the module flag
        "scripts.run_single_evaluation",  # Use module path instead of file path
        "--config",
        str(config_path),
        "--input-data",
        str(
            ingested_file_path
        ),  # Assuming scripts/run_single_evaluation.py takes data path via an argument
        # Add other necessary arguments for scripts/run_single_evaluation.py here
        "--output-jsonl",  # Argument to pass the target .jsonl file path
        str(eval_jsonl_path),
    ]

    # Note: Adjust "--input-data" above based on the actual argument name
    # expected by scripts/run_single_evaluation.py to receive the ingested data path.

    # Run evaluation command without capturing output to allow tqdm to display
    logger.debug(
        "Running command: %s", " ".join(evaluation_command)
    )  # Changed to debug
    try:
        # Use subprocess.run directly without capture_output=True
        evaluation_process = subprocess.run(
            evaluation_command, check=True, text=True, encoding="utf-8"
        )
        logger.info("Evaluation command completed.")
        # Note: stdout/stderr will print directly to console now
    except subprocess.CalledProcessError as e:
        logger.error(
            "Error running evaluation command: %s", " ".join(evaluation_command)
        )
        logger.error("Return code: %s", e.returncode)
        # stdout/stderr would have printed directly
        evaluation_process = None  # Indicate failure
    except FileNotFoundError:
        logger.error("Evaluation script command not found (%s)", evaluation_command[0])
        evaluation_process = None  # Indicate failure

    if evaluation_process is None or evaluation_process.returncode != 0:
        logger.error("CogniBench evaluation script failed.")
        sys.exit(1)

    logger.debug("--- Finished Step 2: Evaluation ---")  # Changed to debug

    logger.debug("--- Starting Step 3: Formatting Output JSON ---")  # Changed to debug
    # Use the dynamically generated paths determined earlier
    jsonl_path = eval_jsonl_path
    json_path = eval_json_path

    if not jsonl_path.is_file():
        logger.warning(
            "Evaluations file not found at %s. Skipping conversion.", jsonl_path
        )
    else:
        try:
            evaluations_list = []
            with jsonl_path.open("r", encoding="utf-8") as infile:
                for line in infile:
                    try:
                        # Skip empty lines if any
                        if line.strip():
                            raw_evaluation_data = json.loads(line)
                            # Convert keys to snake_case
                            evaluation_data = keys_to_snake_case(raw_evaluation_data)
                            # Remove the raw_judge_output key (already snake_cased if present)
                            evaluation_data.pop("raw_judge_output", None)
                            evaluations_list.append(evaluation_data)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            "Skipping invalid JSON line in %s: %s - Error: %s",
                            jsonl_path,
                            line.strip(),
                            e,
                        )

            # Write the formatted JSON file
            with json_path.open("w", encoding="utf-8") as outfile:
                json.dump(
                    evaluations_list, outfile, indent=2, ensure_ascii=False
                )  # Use indent=2 for readability

            logger.info(
                "Successfully created formatted JSON (without raw_judge_output) at %s",
                json_path,
            )

        except IOError as e:
            logger.error("File I/O Error during conversion: %s", e)
        except Exception as e:
            logger.exception("An unexpected error occurred during JSON formatting.")

    logger.debug("--- Finished Step 3: Formatting Output JSON ---")  # Changed to debug

    # --- Step 4: Combine Ingested Data and Evaluation Results ---
    logger.debug("--- Starting Step 4: Combining Results ---")
    # final_results_path is already defined using the output_dir

    # Check if both necessary files exist
    if not ingested_file_path.is_file():
        logger.error(
            "Ingested data file not found at %s. Cannot combine results.",
            ingested_file_path,
        )
    elif not json_path.is_file():
        logger.error(
            "Formatted evaluation file not found at %s. Cannot combine results.",
            json_path,
        )
    else:
        try:
            # Load ingested data (original tasks)
            logger.debug("Loading ingested data from %s", ingested_file_path)
            with ingested_file_path.open("r", encoding="utf-8") as f:
                ingested_tasks = json.load(f)
            # Index ingested data by task_id for easy lookup
            ingested_data_map = {task["task_id"]: task for task in ingested_tasks}
            logger.debug(
                "Indexed %d ingested tasks by task_id.", len(ingested_data_map)
            )

            # Ensure evaluations_list is loaded if Step 3 was skipped or failed partially
            evaluations_list = []  # Initialize in case loading fails
            # The check 'if "evaluations_list" not in locals():' was removed as it was preventing loading.
            # Always attempt to load the formatted file created in Step 3.
            # This block loads the formatted evaluations file. It needs to be indented.
            if json_path.is_file():  # Check if formatted file exists
                logger.debug("Loading formatted evaluations from %s", json_path)
                try:
                    with json_path.open("r", encoding="utf-8") as f:
                        evaluations_list = json.load(f)
                except (IOError, json.JSONDecodeError) as e:
                    logger.error(
                        "Failed to load formatted evaluations file %s: %s. Cannot combine results.",
                        json_path,
                        e,
                    )
                    # evaluations_list remains empty
            else:
                logger.error(
                    "Formatted evaluation file not found at %s. Cannot combine results.",
                    json_path,
                )
                # evaluations_list remains empty

            grouped_results_map = {}  # Use a dict keyed by task_id to group results

            # --- Initialize Summary Statistics ---
            total_structuring_calls = 0
            total_judging_calls = 0
            total_eval_time = 0.0
            total_evaluations_processed = 0
            model_times = {}  # {model_id: total_time}
            model_counts = {}  # {model_id: count}

            logger.debug("Processing %d evaluation results...", len(evaluations_list))
            structuring_model_name = config.get("structuring_model", {}).get(
                "name", "unknown_structuring_model"
            )  # Get structuring model name
            for evaluation in evaluations_list:
                total_evaluations_processed += 1
                task_id = evaluation.get("task_id")
                model_id = evaluation.get("model_id")
                evaluation_id = evaluation.get("evaluation_id")

                if not task_id or not model_id:
                    logger.warning(
                        "Skipping evaluation %s due to missing task_id or model_id.",
                        evaluation_id,
                    )
                    continue
                # Removed debug log for task_id/model_id

                # --- Accumulate Metrics ---
                struct_calls = evaluation.get("structuring_api_calls", 0) or 0
                judge_calls = evaluation.get("judging_api_calls", 0) or 0
                eval_time = evaluation.get("total_time_seconds", 0.0) or 0.0

                total_structuring_calls += struct_calls
                total_judging_calls += judge_calls
                total_eval_time += eval_time

                if model_id:
                    model_times[model_id] = model_times.get(model_id, 0.0) + eval_time
                    model_counts[model_id] = model_counts.get(model_id, 0) + 1

                # --- Get Original Task Data ---
                original_task_data = ingested_data_map.get(task_id)
                if not original_task_data:
                    logger.warning(
                        "Could not find original task data for task_id %s in evaluation %s. Skipping.",
                        task_id,
                        evaluation_id,
                    )
                    continue

                # --- Get Structured Responses (Should already be dicts from JSONL) ---
                # These fields are expected to be dictionary objects loaded from the JSONL
                # generated by core/workflow.py, no further parsing needed here.
                structured_ideal_response_obj = evaluation.get(
                    "structured_ideal_response"
                )
                structured_model_response_obj = evaluation.get(
                    "structured_model_response"
                )

                # Optional: Add validation/logging if they are not dicts as expected
                if not isinstance(structured_ideal_response_obj, dict):
                    logger.warning(
                        f"Task [{task_id}]: Expected 'structured_ideal_response' to be a dict, but got {type(structured_ideal_response_obj)}. Value: {structured_ideal_response_obj}"
                    )
                if not isinstance(structured_model_response_obj, dict):
                    logger.warning(
                        f"Task [{task_id}]: Expected 'structured_model_response' to be a dict, but got {type(structured_model_response_obj)}. Value: {structured_model_response_obj}"
                    )

                # Removed debug log for original task data lookup

                # If first time seeing this task_id, initialize the top-level structure
                if task_id not in grouped_results_map:
                    grouped_results_map[task_id] = {
                        "task_id": task_id,
                        "prompt": original_task_data.get("prompt"),
                        "ideal_response": original_task_data.get("ideal_response"),
                        "final_answer": original_task_data.get("final_answer"),
                        # Filter out system_prompt from metadata before assigning
                        "metadata": {
                            k: v
                            for k, v in original_task_data.get("metadata", {}).items()
                            if k != "system_prompt"
                        },
                        "structured_ideal_response": None,  # Placeholder, will be populated by parsed value
                        "evaluations": [],
                    }

                # Find the specific model response and human eval for this model_id within the original task data
                model_response_text = None
                for resp in original_task_data.get("model_responses", []):
                    if resp.get("model_id") == model_id:
                        model_response_text = resp.get("response_text")
                        break

                human_evaluation_data = {}
                for human_eval in original_task_data.get("human_evaluations", []):
                    if human_eval.get("model_id") == model_id:
                        human_evaluation_data = {
                            k: v for k, v in human_eval.items() if k != "model_id"
                        }
                        break

                # --- Use parsed structured_ideal_response ---
                # The parsing happens earlier (around line 550) creating parsed_ideal_response
                # --- Assign Structured Ideal Response (Directly use the object) ---
                # Assign the object from the evaluation data if the task-level entry is still None
                if (
                    structured_ideal_response_obj is not None
                    and grouped_results_map[task_id].get("structured_ideal_response")
                    is None
                ):
                    # Directly assign the dictionary object loaded from the JSONL
                    grouped_results_map[task_id]["structured_ideal_response"] = (
                        structured_ideal_response_obj
                    )

                # Construct the judge_evaluation object
                excluded_judge_keys = {
                    "task_id",
                    "model_id",
                    "response_id",
                    "ideal_response_id",
                    "raw_judge_output",
                    "structured_ideal_response",  # Exclude this (original string key)
                    "structured_model_response",  # Exclude this (original string key)
                    "structuring_prompt",  # Exclude structuring prompt text
                    "system_prompt",  # Exclude system prompt text if present
                    "judging_prompt",  # Exclude judging prompt text if present
                    # Exclude new metrics
                    "structuring_api_calls",
                    "judging_api_calls",
                    "total_time_seconds",
                }
                judge_evaluation_data = {
                    k: v for k, v in evaluation.items() if k not in excluded_judge_keys
                }

                # Append the combined evaluation details
                # Removed debug log before appending evaluation
                grouped_results_map[task_id]["evaluations"].append(
                    {
                        "model_id": model_id,
                        # Directly assign the dictionary object loaded from the JSONL
                        "structured_model_response": structured_model_response_obj,
                        "human_evaluation": human_evaluation_data,
                        "judge_evaluation": judge_evaluation_data,
                        # Optionally keep raw text if needed, rename key e.g., "raw_model_response_text"
                        "model_response": model_response_text,
                    }
                )
            # --- End of loop processing evaluations ---

            # Convert the grouped results map values into a list
            # Removed debug log for final map size
            final_results_list = list(grouped_results_map.values())

            # --- Calculate Final Summary Statistics ---
            num_tasks = len(grouped_results_map)
            avg_time_per_task = total_eval_time / num_tasks if num_tasks > 0 else 0.0
            avg_time_per_evaluation = (
                total_eval_time / total_evaluations_processed
                if total_evaluations_processed > 0
                else 0.0
            )

            avg_time_per_model = {}
            for model_id, total_time in model_times.items():
                count = model_counts.get(model_id, 0)
                avg_time_per_model[model_id] = total_time / count if count > 0 else 0.0

            summary_stats = {
                "batch_id": batch_stem,
                "timestamp": timestamp,
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
                "models_evaluated": list(model_counts.keys()),
            }

            # --- Create Final Output Object with Summary ---
            final_output = {"summary": summary_stats, "results": final_results_list}

            # Write the final combined results including the summary
            logger.debug(
                "Writing final combined results with summary to %s", final_results_path
            )
            with final_results_path.open("w", encoding="utf-8") as outfile:
                json.dump(final_output, outfile, indent=2, ensure_ascii=False)

            logger.info(
                "Successfully combined results and wrote final output with summary to %s",
                final_results_path,
            )
            # Print the absolute path to stdout for the Streamlit app
            print(f"FINAL_RESULTS_PATH: {final_results_path.resolve()}")
            # Print the path to stdout for the Streamlit app to capture (if needed)
            # print(f"Successfully combined results into {final_results_path}") # Optional

        # Correctly indented except blocks for the try starting at line 391
        except IOError as e:
            logger.error("File I/O Error during result combination: %s", e)
        except Exception as e:
            logger.exception("An unexpected error occurred during result combination.")

        # Correctly indented debug log for finishing Step 4, still within the 'else' block
        logger.debug("--- Finished Step 4: Combining Results ---")

    # Correctly indented final log message at the end of the main script block
    logger.info("--- End-to-End Evaluation Complete ---")
