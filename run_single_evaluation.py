# CogniBench - Single Evaluation Test Harness
# Version: 0.2 (Phase 5 - Structured Input Support)

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

from core.workflow import run_evaluation_workflow
from tqdm import tqdm

try:
    from core.log_setup import setup_logging
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from core.log_setup import setup_logging

logger = logging.getLogger(__name__)


def load_config(config_path: Path):
    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error("Error loading config file %s", config_path, exc_info=True)
        sys.exit(1)


def validate_config(config: Dict[str, Any]) -> bool:
    required_sections = ["llm_client", "evaluation_settings"]
    for section in required_sections:
        if section not in config or not isinstance(config[section], dict):
            logger.error(
                "Config validation failed: Missing or invalid section '%s'.", section
            )
            return False
    return True


def main(args):
    config_path = Path(args.config)
    input_data_path = Path(args.input_data)

    if not input_data_path.is_file():
        logger.error("Input data file not found at %s", input_data_path)
        sys.exit(1)

    config = load_config(config_path)
    if not validate_config(config):
        sys.exit(1)

    try:
        with input_data_path.open("r", encoding="utf-8") as f:
            evaluation_tasks = json.load(f)
    except Exception as e:
        logger.error("Error loading input data file %s", input_data_path, exc_info=True)
        sys.exit(1)

    all_results = []
    overall_success = True

    total_tasks = len(evaluation_tasks)
    logger.info(f"Starting evaluation for {total_tasks} tasks.")

    for i, task in enumerate(
        tqdm(evaluation_tasks, desc="Evaluating Tasks", file=sys.stdout)
    ):
        print(f"PROGRESS: Task {i + 1}/{total_tasks}", file=sys.stdout, flush=True)
        task_id = task.get("task_id", "unknown_task")
        prompt_text = task.get("prompt")
        correct_answer = task.get("correct_answer", "")

        ideal_response_text = task.get("ideal_response")
        structured_ideal_response = task.get("structured_ideal_response")

        model_responses = task.get("model_responses", [])

        if not prompt_text or not ideal_response_text:
            logger.warning(
                "Skipping task %s due to missing prompt or ideal response.", task_id
            )
            continue

        for model_response in model_responses:
            response_text = model_response.get("response_text")
            structured_model_response = model_response.get("structured_model_response")
            model_id = model_response.get("model_id", "unknown_model")

            if (
                args.use_structured
                and structured_model_response
                and structured_ideal_response
            ):
                response_input = structured_model_response
                ideal_input = structured_ideal_response
                structured = True
            else:
                response_input = response_text
                ideal_input = ideal_response_text
                structured = False

            if not response_input:
                logger.warning(
                    "Skipping response from model %s in task %s due to missing text.",
                    model_id,
                    task_id,
                )
                continue

            result = run_evaluation_workflow(
                prompt=prompt_text,
                response=response_input,
                ideal_response=ideal_input,
                correct_answer=correct_answer,
                config=config,
                task_id=task_id,
                model_id=model_id,
                structured=structured,
                output_jsonl_path=Path(args.output_jsonl)
                if args.output_jsonl
                else None,
            )

            all_results.append(result)

            if result.get("status") != "success":
                overall_success = False
                logger.error(
                    "Workflow Error for task %s, model %s: %s",
                    task_id,
                    model_id,
                    result.get("message"),
                )

    results_output_path_str = config.get("output_options", {}).get(
        "results_file", "data/evaluation_results.json"
    )
    results_output_path = Path(results_output_path_str)
    results_output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with results_output_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Overall results saved to: %s", results_output_path)
    except Exception as e:
        logger.error(
            "Error saving overall results to %s", results_output_path, exc_info=True
        )

    if not overall_success:
        logger.error("Evaluation Run Completed with Errors")
        sys.exit(1)
    else:
        logger.info("Evaluation Run Completed Successfully")


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Run CogniBench evaluation on ingested data."
    )
    parser.add_argument(
        "--input-data", required=True, help="Path to the ingested JSON data file."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--output-jsonl", help="Path to the output JSONL file for detailed results."
    )
    parser.add_argument(
        "--use-structured",
        action="store_true",
        help="Use structured responses if available.",
    )
    args = parser.parse_args()
    main(args)
