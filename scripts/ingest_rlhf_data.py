"""
CogniBench RLHF Data Ingestion Script.

Loads raw RLHF JSON data (typically from external sources), extracts relevant
fields needed for CogniBench evaluations (prompt, ideal response, model responses,
metadata, human evaluations), performs basic transformations (like boolean conversion
for Yes/No scores), and saves the processed data into a new timestamped JSON file
suitable for use with the CogniBench evaluation runners.
"""

import argparse
import datetime
import json
import logging
import re
import sys
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
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

        def setup_logging():
            pass  # No-op function


# Get logger for this module
logger = logging.getLogger("backend")


from typing import Any, Dict, List, Optional


def robust_latex_conversion(text: Optional[str]) -> Optional[str]:
    """
    Robustly converts common LaTeX formatting issues to plain text.

    Args:
        text: The input string, potentially containing LaTeX.

    Returns:
        The processed string with common LaTeX elements removed, or None if input was None.
    """
    if not text:
        return text
    # Replace common LaTeX math delimiters
    text = re.sub(r"\$\$(.*?)\$\$", r"\1", text)
    text = re.sub(r"\$(.*?)\$", r"\1", text)
    # Remove LaTeX commands
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    # Remove curly braces
    text = re.sub(r"[{}]", "", text)
    return text.strip()


def enhanced_final_answer_extraction(
    raw_pref_form: List[Dict[str, Any]],
) -> Optional[str]:
    """
    Enhanced extraction logic for final answers from raw preference form data.

    Extracts the 'Final Answer' value and applies robust LaTeX conversion and stripping.

    Args:
        raw_pref_form: The list of dictionaries representing the raw preference evaluation form.

    Returns:
        The extracted and cleaned final answer string, or None if not found.
    """
    final_answer = extract_prompt_evaluation_value(raw_pref_form, "Final Answer")
    if final_answer:
        # Robust LaTeX conversion
        final_answer = robust_latex_conversion(final_answer)
        # Additional heuristic: remove leading/trailing whitespace and punctuation
        final_answer = final_answer.strip().strip(".")
    return final_answer


def extract_prompt_evaluation_value(
    prompt_eval_list: List[Dict[str, Any]], key_name: str
) -> Optional[Any]:
    """
    Helper function to extract a specific value from a list of prompt evaluation dictionaries.

    Searches for a dictionary where the 'question' key matches `key_name` and returns
    the corresponding 'human_input_value'.

    Args:
        prompt_eval_list: The list of prompt evaluation dictionaries.
        key_name: The 'question' value to search for.

    Returns:
        The corresponding 'human_input_value', or None if not found.
    """
    for item in prompt_eval_list:
        if item.get("question") == key_name:
            return item.get("human_input_value")
    return None


def ingest_rlhf_data(input_paths: List[Path], output_path: Path):
    """
    Loads RLHF JSON data from multiple input files, extracts relevant fields
    for CogniBench, combines them, and saves it to a single new JSON file.
    """
    ingested_data = []
    total_tasks_processed = 0
    total_tasks_skipped = 0
    output_path_obj = output_path  # Keep output path object

    for input_path_obj in input_paths:
        logger.info(f"--- Processing input file: {input_path_obj} ---")
        try:
            with input_path_obj.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.warning("Input file not found at %s. Skipping.", input_path_obj)
            continue  # Skip to the next file
        except json.JSONDecodeError:
            logger.warning(
                "Could not decode JSON from %s. Skipping.",
                input_path_obj,
                exc_info=True,
            )
            continue  # Skip to the next file
        except Exception as e:
            logger.warning(
                "Unexpected error reading file %s: %s. Skipping.",
                input_path_obj,
                e,
                exc_info=True,
            )
            continue  # Skip to the next file

        if "rlhf" not in data or not isinstance(data["rlhf"], list):
            logger.warning(
                "'rlhf' key not found or is not a list in %s. Skipping.", input_path_obj
            )
            continue  # Skip to the next file

        file_task_count = len(data["rlhf"])
        file_tasks_ingested = 0
        file_tasks_skipped = 0
        logger.info("Found %d tasks in %s.", file_task_count, input_path_obj)

        for idx, task_item in enumerate(data["rlhf"], start=1):
            logger.debug(
                "Processing task %d/%d from %s: Task ID %s",
                idx,
                file_task_count,
                input_path_obj.name,
                task_item.get("taskId"),
            )
            task_id = task_item.get("taskId")
            messages = task_item.get("messages", [])

            user_prompt = None
            ideal_response = None
            subject = None
            complexity = None
            final_answer = None
            system_prompt = None

            # Find user prompt and prompt evaluation metadata
            for message in messages:
                if message.get("role") == "user":
                    user_prompt = message.get("text")
                    prompt_evaluation = message.get("prompt_evaluation", [])
                    subject = extract_prompt_evaluation_value(
                        prompt_evaluation, "Subject"
                    )
                    complexity = extract_prompt_evaluation_value(
                        prompt_evaluation, "Complexity"
                    )
                elif message.get("role") == "system":
                    system_prompt = message.get("text")

            # Find ideal response, model responses, and human evals in assistant message
            model_responses = []
            human_evaluations = []
            for message in messages:
                if message.get("role") == "assistant":
                    signal = message.get("signal", {})
                    ideal_response = signal.get("ideal_response")
                    human_evaluations = signal.get("human_evals", [])

                    raw_pref_form = signal.get("raw_preference_evaluation_form", [])
                    final_answer = enhanced_final_answer_extraction(raw_pref_form)

                    response_options = message.get("response_options", [])
                    for resp_opt in response_options:
                        model_id = resp_opt.get("model_id")
                        text = resp_opt.get("text")
                        if model_id and text:
                            model_responses.append(
                                {"model_id": model_id, "response_text": text}
                            )

                    if not ideal_response:
                        raw_pref_form = signal.get("raw_preference_evaluation_form", [])
                        final_answer_value = extract_prompt_evaluation_value(
                            raw_pref_form, "Final Answer"
                        )
                        if final_answer_value:
                            ideal_response = final_answer_value
                    break

            if user_prompt and ideal_response and task_id is not None:
                # Transform human_evaluations format
                transformed_human_evals = []
                for eval_item in human_evaluations:
                    evaluation_details = {}
                    original_eval_form = eval_item.get("evaluation_form", [])
                    for form_item in original_eval_form:
                        question_str = form_item.get("question")
                        value = form_item.get("human_input_value")
                        if question_str:
                            standardized_key = question_str.lower().replace(" ", "_")
                            if isinstance(value, str):
                                if value.lower() == "yes":
                                    value = True
                                elif value.lower() == "no":
                                    value = False
                            evaluation_details[standardized_key] = value

                    transformed_eval_item = {"model_id": eval_item.get("model_id")}
                    transformed_eval_item.update(evaluation_details)
                    transformed_human_evals.append(transformed_eval_item)

                ingested_item = {
                    "task_id": task_id,
                    "prompt": user_prompt,
                    "ideal_response": ideal_response,
                    "final_answer": final_answer,
                    "model_responses": model_responses,
                    "human_evaluations": transformed_human_evals,
                    "metadata": {
                        "subject": subject,
                        "complexity": complexity,
                        "system_prompt": system_prompt,
                    },
                }
                ingested_data.append(ingested_item)
                file_tasks_ingested += 1
                logger.debug(
                    "Successfully ingested task ID %s from %s",
                    task_id,
                    input_path_obj.name,
                )
            else:
                file_tasks_skipped += 1
                logger.warning(
                    "Skipping task ID %s from %s due to missing prompt or ideal response. Prompt: %s, Ideal Response: %s",
                    task_id,
                    input_path_obj.name,
                    user_prompt is not None,
                    ideal_response is not None,
                )
        logger.info(
            f"Finished processing {input_path_obj.name}: Ingested {file_tasks_ingested}, Skipped {file_tasks_skipped}"
        )
        total_tasks_processed += file_tasks_ingested
        total_tasks_skipped += file_tasks_skipped

    # Ensure the output directory exists using pathlib
    output_dir = output_path_obj.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the combined ingested data
    if not ingested_data:
        logger.warning(
            "No tasks were successfully ingested from any input file. Output file will be empty."
        )

    try:
        with output_path_obj.open("w", encoding="utf-8") as f:
            json.dump(ingested_data, f, indent=2, ensure_ascii=False)
        logger.info("--- Ingestion Summary ---")
        logger.info(f"Input files processed: {len(input_paths)}")
        logger.info(f"Total tasks ingested: {total_tasks_processed}")
        logger.info(f"Total tasks skipped: {total_tasks_skipped}")
        logger.info(
            "Successfully saved combined ingested data (%d tasks) to %s",
            len(ingested_data),
            output_path_obj,
        )
        # Print output path to stdout for capture by other scripts
        print(output_path_obj.resolve())
    except IOError:
        logger.error(
            "Error writing combined output file to %s", output_path_obj, exc_info=True
        )
        sys.exit(1)  # Exit if final write fails


if __name__ == "__main__":
    setup_logging()  # Setup logging first
    logger.info("Starting RLHF data ingestion script.")
    parser = argparse.ArgumentParser(
        description="Ingest RLHF JSON data for CogniBench."
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        type=str,
        help="Paths to one or more input RLHF JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Optional directory to save the combined ingested file.",
    )
    args = parser.parse_args()

    input_paths = [Path(f) for f in args.input_files]
    valid_inputs = True
    for p in input_paths:
        if not p.is_file():
            logger.error("Input file not found at %s", p)
            valid_inputs = False
    if not valid_inputs:
        sys.exit(1)

    # Generate timestamp string YYYYMMDD_HHMMSS (added seconds)
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default to data directory relative to the script's parent (CogniBench/)
        # This might need adjustment if the script is called from elsewhere
        output_dir = Path(__file__).parent.parent / "data"
        logger.warning(f"No output directory specified, defaulting to: {output_dir}")

    # Construct output path using combined stems
    input_stems = [p.stem for p in input_paths]
    combined_stem = "_".join(sorted(input_stems))  # Sort for consistency
    output_filename = (
        f"{combined_stem}_{timestamp}_ingested.json"  # Timestamp before suffix
    )
    output_path = output_dir / output_filename

    logger.info("Input files: %s", [str(p) for p in input_paths])
    logger.info("Output file will be: %s", output_path)

    ingest_rlhf_data(input_paths, output_path)
