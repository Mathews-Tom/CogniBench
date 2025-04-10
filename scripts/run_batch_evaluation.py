import argparse
import json  # Import json for conversion step
import logging
import re  # Import re for snake_case conversion
import subprocess
import sys
import time  # For timestamp
from datetime import datetime
from pathlib import Path

# Assuming log_setup is in core directory relative to project root
try:
    from core.log_setup import setup_logging
except ImportError:
    # Fallback if running script directly and core isn't in path easily
    try:
        # Adjust path based on expected execution context
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from core.log_setup import setup_logging
    except ImportError:
        # If setup still fails, provide a basic config
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

        def setup_logging():
            pass  # No-op function


# Get logger for this module
logger = logging.getLogger(__name__)


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


if __name__ == "__main__":
    setup_logging()  # Setup logging
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
        help="Path to the CogniBench configuration file for run_single_evaluation.py.",
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
    # Assumes run_single_evaluation.py is in the parent directory (CogniBench/)
    evaluation_script_path = Path(__file__).parent.parent / "run_single_evaluation.py"
    evaluation_command = [
        sys.executable,
        str(evaluation_script_path),
        "--config",
        str(config_path),
        "--input-data",
        str(
            ingested_file_path
        ),  # Assuming run_single_evaluation takes data path via an argument
        # Add other necessary arguments for run_single_evaluation.py here
        "--output-jsonl",  # Argument to pass the target .jsonl file path
        str(eval_jsonl_path),
    ]

    # Note: Adjust "--input-data" above based on the actual argument name
    # expected by run_single_evaluation.py to receive the ingested data path.

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
            if "evaluations_list" not in locals():
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
                    evaluations_list = []  # Ensure it's an empty list if loading fails

            grouped_results_map = {}  # Use a dict keyed by task_id to group results

            logger.debug("Processing %d evaluation results...", len(evaluations_list))
            for evaluation in evaluations_list:
                task_id = evaluation.get("task_id")
                model_id = evaluation.get("model_id")
                evaluation_id = evaluation.get("evaluation_id")

                if not task_id or not model_id:
                    logger.warning(
                        "Skipping evaluation %s due to missing task_id or model_id.",
                        evaluation_id,
                    )
                    continue

                original_task_data = ingested_data_map.get(task_id)
                if not original_task_data:
                    logger.warning(
                        "Could not find original task data for task_id %s in evaluation %s. Skipping.",
                        task_id,
                        evaluation_id,
                    )
                    continue

                # If first time seeing this task_id, initialize the top-level structure
                if task_id not in grouped_results_map:
                    grouped_results_map[task_id] = {
                        "task_id": task_id,
                        "prompt": original_task_data.get("prompt"),
                        "ideal_response": original_task_data.get("ideal_response"),
                        "final_answer": original_task_data.get(
                            "final_answer"
                        ),  # Use renamed key final_answer
                        "metadata": original_task_data.get("metadata", {}),
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
                        # Exclude model_id itself from the human_evaluation object
                        human_evaluation_data = {
                            k: v for k, v in human_eval.items() if k != "model_id"
                        }
                        break

                # Construct the judge_evaluation object from the current evaluation record
                # Exclude fields already present at top level or handled separately
                excluded_judge_keys = {
                    "task_id",
                    "model_id",
                    "response_id",
                    "ideal_response_id",  # Already snake_case from conversion
                    "raw_judge_output",  # Already snake_case and removed earlier
                    # Add any other potential non-snake_case keys from judge output if known
                }
                judge_evaluation_data = {
                    k: v for k, v in evaluation.items() if k not in excluded_judge_keys
                }

                # Append the combined evaluation details for this model to the task's list
                grouped_results_map[task_id]["evaluations"].append(
                    {
                        "model_id": model_id,
                        "model_response": model_response_text,
                        "human_evaluation": human_evaluation_data,
                        "judge_evaluation": judge_evaluation_data,
                    }
                )

            # Convert the grouped results map values into a list for the final JSON output
            final_results_list = list(grouped_results_map.values())

            # Write the final combined results
            logger.debug("Writing final combined results to %s", final_results_path)
            with final_results_path.open("w", encoding="utf-8") as outfile:
                json.dump(final_results_list, outfile, indent=2, ensure_ascii=False)

            logger.info(
                "Successfully combined ingested data and evaluations into %s",
                final_results_path,
            )
            logger.debug("--- Finished Step 4: Combining Results ---")

        except IOError as e:
            logger.error("File I/O Error during result combination: %s", e)
        except Exception as e:
            logger.exception("An unexpected error occurred during result combination.")

    logger.info("--- End-to-End Evaluation Complete ---")
