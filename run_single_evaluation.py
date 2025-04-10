# CogniBench - Single Evaluation Test Harness
# Version: 0.1 (Phase 1)

import argparse
import argparse
import json
import sys
import logging # Import logging
from pathlib import Path
from tqdm import tqdm # Import tqdm
# Assuming workflow and config loading utilities exist
from core.workflow import run_evaluation_workflow
# Assuming log_setup is in core directory relative to project root
try:
    from core.log_setup import setup_logging
except ImportError:
    # Fallback if running script directly and core isn't in path easily
    try:
        # Adjust path based on expected execution context
        sys.path.insert(0, str(Path(__file__).parent))
        from core.log_setup import setup_logging
    except ImportError:
        # If setup still fails, provide a basic config
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        def setup_logging(): pass # No-op function

# Get logger for this module
logger = logging.getLogger(__name__)

# Configuration will be loaded from the config file passed via args


# --- Main Execution Logic ---
def load_config(config_path: Path):
    """Loads configuration from a YAML file."""
    # Basic placeholder - requires PyYAML (pip install pyyaml)
    # Add error handling
    try:
        import yaml
        with config_path.open('r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except ImportError:
        logger.error("PyYAML is required to load config.yaml. Please install it (`uv pip install pyyaml`).")
        sys.exit(1)
    except Exception as e:
        logger.error("Error loading config file %s", config_path, exc_info=True)
        sys.exit(1)


def main(args):
    """Loads data, iterates through tasks/responses, and runs evaluation workflow."""
    config_path = Path(args.config)
    input_data_path = Path(args.input_data)

    if not input_data_path.is_file():
        logger.error("Input data file not found at %s", input_data_path)
        sys.exit(1)

    config = load_config(config_path)
    if not config:
        sys.exit(1) # Error handled in load_config

    try:
        with input_data_path.open("r", encoding="utf-8") as f:
            evaluation_tasks = json.load(f)
    except Exception as e:
        logger.error("Error loading or parsing input data file %s", input_data_path, exc_info=True)
        sys.exit(1)

    all_results = []
    overall_success = True

    logger.debug("--- Starting Evaluation Run ---") # Changed to debug
    logger.debug("Config: %s", config_path) # Changed to debug
    logger.debug("Input Data: %s", input_data_path) # Changed to debug

    # Wrap the main loop with tqdm for progress bar
    for task in tqdm(evaluation_tasks, desc="Evaluating Tasks"):
        task_id = task.get("task_id", "unknown_task")
        prompt_text = task.get("prompt")
        ideal_response_text = task.get("ideal_response")
        model_responses = task.get("model_responses", [])

        if not prompt_text or not ideal_response_text:
            logger.warning("Skipping task %s due to missing prompt or ideal response.", task_id)
            continue

        # Logging within the loop might be too verbose with tqdm, consider logging level adjustments
        # logger.debug("--- Evaluating Task ID: %s ---", task_id)

        for model_response in model_responses:
            response_text = model_response.get("response_text")
            model_id = model_response.get("model_id", "unknown_model")

            if not response_text:
                logger.warning("Skipping response from model %s in task %s due to missing text.", model_id, task_id)
                continue

            # logger.debug("  - Evaluating response from Model: %s", model_id)

            # --- Call the workflow function ---
            # Assumes run_evaluation_workflow is updated to accept text and config
            result = run_evaluation_workflow(
                prompt=prompt_text,
                response=response_text,
                ideal_response=ideal_response_text,
                config=config, # Pass loaded config
                task_id=task_id, # Pass identifiers for context
                model_id=model_id,
                output_jsonl_path=Path(args.output_jsonl) if args.output_jsonl else None # Pass output path
            )
            # --- End workflow call ---

            # logger.debug("    Result Status: %s", result.get('status'))
            all_results.append(result)

            if result.get("status") != "success":
                overall_success = False
                logger.error("    Workflow Error for task %s, model %s: %s", task_id, model_id, result.get('message'))


    # --- Save overall results (optional) ---
    # Example: save to a file specified in config or a default location
    results_output_path_str = config.get("output_options", {}).get("results_file", "data/evaluation_results.json")
    results_output_path = Path(results_output_path_str)
    results_output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with results_output_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        logger.debug("--- Overall results saved to: %s ---", results_output_path) # Changed to debug
    except Exception as e:
        logger.error("Error saving overall results to %s", results_output_path, exc_info=True)


    if not overall_success:
        logger.error("--- Evaluation Run Completed with Errors ---")
        sys.exit(1) # Exit with error if any workflow failed
    else:
        logger.info("--- Evaluation Run Completed Successfully ---")


if __name__ == "__main__":
    setup_logging() # Setup logging
    logger.info("Starting CogniBench evaluation script.")
    parser = argparse.ArgumentParser(description="Run CogniBench evaluation on ingested data.")
    parser.add_argument("--input-data", required=True, help="Path to the ingested JSON data file.")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--output-jsonl", help="Path to the output JSONL file for detailed results.")
    # Add other potential arguments here

    args = parser.parse_args()
    main(args)
