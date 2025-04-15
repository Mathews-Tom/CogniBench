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


def robust_latex_conversion(text):
    """
    Robustly converts common LaTeX formatting issues to plain text.
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


def enhanced_final_answer_extraction(raw_pref_form):
    """
    Enhanced extraction logic for final answers using improved regex patterns.
    """
    final_answer = extract_prompt_evaluation_value(raw_pref_form, "Final Answer")
    if final_answer:
        # Robust LaTeX conversion
        final_answer = robust_latex_conversion(final_answer)
        # Additional heuristic: remove leading/trailing whitespace and punctuation
        final_answer = final_answer.strip().strip(".")
    return final_answer


def extract_prompt_evaluation_value(prompt_eval_list, key_name):
    """Helper function to extract specific values from prompt_evaluation."""
    for item in prompt_eval_list:
        if item.get("question") == key_name:
            return item.get("human_input_value")
    return None


def ingest_rlhf_data(input_path: Path, output_path: Path):
    """
    Loads RLHF JSON data, extracts relevant fields for CogniBench,
    and saves it to a new JSON file.
    """
    ingested_data = []
    # input_path and output_path are already Path objects from the caller
    input_path_obj = input_path
    output_path_obj = output_path

    try:
        with input_path_obj.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error("Input file not found at %s", input_path_obj)
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(
            "Could not decode JSON from %s", input_path_obj, exc_info=True
        )  # Include exception info
        sys.exit(1)

    if "rlhf" not in data or not isinstance(data["rlhf"], list):
        logger.error("'rlhf' key not found or is not a list in the input JSON.")
        sys.exit(1)

    logger.info("Starting ingestion of %d tasks.", len(data["rlhf"]))
    for idx, task_item in enumerate(data["rlhf"], start=1):
        logger.debug(
            "Processing task %d/%d: Task ID %s",
            idx,
            len(data["rlhf"]),
            task_item.get("taskId"),
        )
        task_id = task_item.get("taskId")
        messages = task_item.get("messages", [])

        user_prompt = None
        ideal_response = None
        subject = None
        complexity = None
        final_answer = None  # Initialize final_answer
        system_prompt = None  # Initialize system_prompt
        # Find user prompt and prompt evaluation metadata
        for message in messages:
            if message.get("role") == "user":
                user_prompt = message.get("text")
                prompt_evaluation = message.get("prompt_evaluation", [])
                subject = extract_prompt_evaluation_value(prompt_evaluation, "Subject")
                complexity = extract_prompt_evaluation_value(
                    prompt_evaluation, "Complexity"
                )
                # Don't break yet, look for system prompt too
            elif message.get("role") == "system":
                system_prompt = message.get("text")
                # Don't break, might be other messages

        # Find ideal response, model responses, and human evals in assistant message
        model_responses = []
        human_evaluations = []
        for message in messages:
            if message.get("role") == "assistant":
                signal = message.get("signal", {})
                ideal_response = signal.get("ideal_response")
                human_evaluations = signal.get("human_evals", [])  # Extract human evals

                # Extract final answer from raw_preference_evaluation_form
                raw_pref_form = signal.get("raw_preference_evaluation_form", [])
                final_answer = enhanced_final_answer_extraction(raw_pref_form)

                # Extract model responses
                response_options = message.get("response_options", [])
                for resp_opt in response_options:
                    model_id = resp_opt.get("model_id")
                    text = resp_opt.get("text")
                    if model_id and text:
                        model_responses.append(
                            {"model_id": model_id, "response_text": text}
                        )

                # Optional Fallback for ideal_response: Check raw_preference_evaluation_form if ideal_response is missing
                if not ideal_response:
                    raw_pref_form = signal.get("raw_preference_evaluation_form", [])
                    final_answer_value = extract_prompt_evaluation_value(
                        raw_pref_form, "Final Answer"
                    )
                    if final_answer_value:
                        ideal_response = final_answer_value

                break  # Assuming one assistant message with signal and response options

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
                        # Standardize key to snake_case
                        standardized_key = question_str.lower().replace(" ", "_")

                        # Convert "Yes"/"No" to boolean
                        if isinstance(value, str):
                            if value.lower() == "yes":
                                value = True
                            elif value.lower() == "no":
                                value = False

                        evaluation_details[standardized_key] = value

                # Create the base item, keeping only model_id (and potentially others if needed later)
                # Exclude 'evaluation_form' and 'processed__score'
                transformed_eval_item = {
                    "model_id": eval_item.get("model_id")
                    # Add other keys here if they should be preserved explicitly
                }
                # Merge the extracted evaluation details directly into the item
                transformed_eval_item.update(evaluation_details)
                transformed_human_evals.append(transformed_eval_item)

            ingested_item = {
                "task_id": task_id,
                "prompt": user_prompt,
                "ideal_response": ideal_response,  # Already extracted
                "final_answer": final_answer,  # Add the renamed field
                "model_responses": model_responses,
                "human_evaluations": transformed_human_evals,  # Use transformed list
                "metadata": {
                    "subject": subject,
                    "complexity": complexity,
                    "system_prompt": system_prompt,  # Add extracted system prompt
                },
            }
            ingested_data.append(ingested_item)
            logger.debug("Successfully ingested task ID %s", task_id)
        else:
            # Print warnings to stderr
            logger.warning(
                "Skipping task ID %s due to missing prompt or ideal response. Prompt: %s, Ideal Response: %s",
                task_id,
                user_prompt is not None,
                ideal_response is not None,
            )

    # Ensure the output directory exists using pathlib
    output_dir = output_path_obj.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the ingested data
    try:
        with output_path_obj.open("w", encoding="utf-8") as f:
            json.dump(ingested_data, f, indent=2, ensure_ascii=False)
        # Log success message
        logger.info(
            "Successfully ingested %d tasks to %s", len(ingested_data), output_path_obj
        )
        # Print output path to stdout for capture by other scripts
        print(output_path_obj.resolve())
    except IOError as e:
        logger.error("Error writing output file to %s", output_path_obj, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    setup_logging()  # Setup logging first
    logger.info("Starting RLHF data ingestion script.")
    parser = argparse.ArgumentParser(
        description="Ingest RLHF JSON data for CogniBench."
    )
    parser.add_argument(
        "input_file", type=str, help="Path to the input RLHF JSON file."
    )
    parser.add_argument(
        "--output-dir", type=str, help="Optional directory to save the ingested file."
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.is_file():
        logger.error("Input file not found at %s", input_path)
        sys.exit(1)

    # Generate timestamp string YYYYMMDD_HHMM
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default to data directory relative to the script's parent (CogniBench/)
        output_dir = Path(__file__).parent.parent / "data"

    # Construct output path
    output_filename = f"{input_path.stem}_ingested_{timestamp}.json"
    output_path = output_dir / output_filename

    logger.info("Input file: %s", input_path)
    logger.info("Output file will be: %s", output_path)

    ingest_rlhf_data(input_path, output_path)
    # Removed duplicate call
