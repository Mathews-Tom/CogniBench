#!/usr/bin/env python3
# CogniBench/scripts/prepare_judging_batch.py
"""
Prepares and submits the judging batch job for CogniBench evaluations.

This script loads the structured outputs generated by the structuring batch,
formats judging requests based on these outputs and ideal responses,
and submits a new batch job to the OpenAI Batch API for judging.
"""

import argparse
import asyncio
import json
import logging
import os
import sys

# Adjust path to import core modules
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Import core components
from core.batch_processor import (create_batch_job, format_requests_to_jsonl,
                                  upload_batch_file)
from core.config import load_config
from core.llm_clients.openai_client import OpenAIClient
from core.prompt_templates import load_prompt_template

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Prepare and submit OpenAI Batch API job for judging."
    )
    parser.add_argument(
        "--structured-input",
        "-i",
        required=True,
        help="Path to the JSON file containing structured outputs from the previous stage.",
    )
    parser.add_argument(
        "--config-file",
        "-c",
        default="./config.yaml",  # Assuming config.yaml is in the CogniBench root
        help="Path to the configuration file (default: ./config.yaml).",
    )
    # Add other arguments as needed later, e.g., output directory
    args = parser.parse_args()
    logger.info("Starting judging batch preparation...")
    logger.info(f"Loading structured input from: {args.structured_input}")
    logger.info(f"Using configuration file: {args.config_file}")

    # --- Load Config and Judge Prompt Template ---
    try:
        config = load_config(args.config_file)
        judge_config = config.get("judging", {})
        judge_model = judge_config.get("model")
        judge_prompt_template_path = judge_config.get("prompt_template_path")
        judge_temperature = judge_config.get(
            "temperature", 0.0
        )  # Default temp if not specified

        if not judge_model or not judge_prompt_template_path:
            logger.critical(
                "Judge model or prompt template path not found in config file under 'judging' section."
            )
            return

        # Construct full path relative to config file location if necessary
        if not os.path.isabs(judge_prompt_template_path):
            config_dir = os.path.dirname(os.path.abspath(args.config_file))
            judge_prompt_template_path = os.path.join(
                config_dir, judge_prompt_template_path
            )

        judge_prompt_template = load_prompt_template(judge_prompt_template_path)
        logger.info(f"Loaded judge prompt template from: {judge_prompt_template_path}")
        logger.info(f"Using judge model: {judge_model}")

    except FileNotFoundError:
        logger.critical(f"Configuration file not found: {args.config_file}")
        return
    except Exception as e:
        logger.critical(f"Error loading configuration or judge prompt template: {e}")
        return

    # --- Initialize OpenAI Client ---
    try:
        openai_client = OpenAIClient()
        logger.info("OpenAI client initialized.")
    except Exception as e:
        logger.critical(f"Failed to initialize OpenAI client: {e}")
        return

    # --- Load Structured Outputs ---
    structured_items = None
    structured_input_path = args.structured_input
    if not os.path.exists(structured_input_path):
        logger.critical(f"Input file not found: {structured_input_path}")
        return  # Stop execution if file is missing

    try:
        with open(structured_input_path, "r", encoding="utf-8") as f:
            structured_items = json.load(f)
        if not isinstance(structured_items, list):
            logger.warning(
                f"Loaded data from {structured_input_path} is not a list. Type: {type(structured_items)}"
            )
            # Depending on expected format, might need to raise error or handle differently
        logger.info(
            f"Successfully loaded {len(structured_items) if isinstance(structured_items, list) else 'item'} from {structured_input_path}"
        )
    except json.JSONDecodeError as e:
        logger.critical(f"Error decoding JSON from {structured_input_path}: {e}")
        return  # Stop execution on JSON error
    except Exception as e:
        logger.critical(f"Error reading file {structured_input_path}: {e}")
        return  # Stop execution on other file errors

    if not structured_items:
        logger.critical("No structured items loaded or file was empty.")
        return  # Stop if loading failed or file empty

    # --- Generate Judging Requests ---
    judging_requests = []
    skipped_items_count = 0
    logger.info("Generating judging requests...")

    for item in structured_items:
        try:
            # Assuming structure added in Task 19/22 includes results like this:
            # item = { ..., 'model_structuring_result': {'status': 'succeeded', 'batch_response_body': {...}}, 'ideal_structuring_result': {'status': 'succeeded', 'batch_response_body': {...}} }
            model_structuring_result = item.get("model_structuring_result", {})
            ideal_structuring_result = item.get("ideal_structuring_result", {})

            model_status = model_structuring_result.get("status")
            ideal_status = ideal_structuring_result.get("status")

            if model_status != "succeeded" or ideal_status != "succeeded":
                logger.warning(
                    f"Skipping item {item.get('task_id', 'UNKNOWN')}_{item.get('model_id', 'UNKNOWN')} due to failed structuring. Model status: {model_status}, Ideal status: {ideal_status}"
                )
                skipped_items_count += 1
                continue

            # Extract necessary data
            original_prompt = item.get("prompt")
            correct_answer = item.get(
                "correct_answer", ""
            )  # Handle cases where it might be missing
            task_id = item.get("task_id")
            model_id = item.get("model_id")

            # Extract structured content - adjust path based on actual OpenAI response structure within batch_response_body
            # Assuming the structured content is within the message content of the first choice
            try:
                structured_model_response_content = (
                    model_structuring_result.get("batch_response_body", {})
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content")
                )
                structured_ideal_response_content = (
                    ideal_structuring_result.get("batch_response_body", {})
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content")
                )

                # The content might be a stringified JSON or already parsed. Ensure it's a string for the template.
                if isinstance(structured_model_response_content, (dict, list)):
                    structured_model_response_str = json.dumps(
                        structured_model_response_content, indent=2
                    )
                elif isinstance(structured_model_response_content, str):
                    # Attempt to parse and re-stringify for consistent formatting, but handle if it's not valid JSON
                    try:
                        structured_model_response_str = json.dumps(
                            json.loads(structured_model_response_content), indent=2
                        )
                    except json.JSONDecodeError:
                        structured_model_response_str = structured_model_response_content  # Use as is if not valid JSON
                else:
                    logger.warning(
                        f"Unexpected type for structured model response content in item {task_id}_{model_id}: {type(structured_model_response_content)}. Using empty string."
                    )
                    structured_model_response_str = ""

                if isinstance(structured_ideal_response_content, (dict, list)):
                    structured_ideal_response_str = json.dumps(
                        structured_ideal_response_content, indent=2
                    )
                elif isinstance(structured_ideal_response_content, str):
                    try:
                        structured_ideal_response_str = json.dumps(
                            json.loads(structured_ideal_response_content), indent=2
                        )
                    except json.JSONDecodeError:
                        structured_ideal_response_str = (
                            structured_ideal_response_content
                        )
                else:
                    logger.warning(
                        f"Unexpected type for structured ideal response content in item {task_id}_{model_id}: {type(structured_ideal_response_content)}. Using empty string."
                    )
                    structured_ideal_response_str = ""

            except (IndexError, KeyError, TypeError) as e:
                logger.warning(
                    f"Skipping item {task_id}_{model_id} due to error extracting structured content: {e}"
                )
                skipped_items_count += 1
                continue

            if not all(
                [
                    original_prompt,
                    task_id,
                    model_id,
                    structured_model_response_str,
                    structured_ideal_response_str,
                ]
            ):
                logger.warning(
                    f"Skipping item {task_id}_{model_id} due to missing essential data after extraction."
                )
                skipped_items_count += 1
                continue

            # Format the judge prompt
            # Assuming template variables like {prompt}, {model_structured_output}, {ideal_structured_output}, {correct_answer}
            try:
                judge_prompt_formatted = judge_prompt_template.format(
                    prompt=original_prompt,
                    model_structured_output=structured_model_response_str,
                    ideal_structured_output=structured_ideal_response_str,
                    correct_answer=correct_answer,
                )
            except KeyError as e:
                logger.error(
                    f"Missing key in judge prompt template ({judge_prompt_template_path}): {e}. Cannot format prompt for item {task_id}_{model_id}."
                )
                skipped_items_count += 1
                continue

            # Create the request dictionary for the batch API
            custom_id = f"judge_{task_id}_{model_id}"
            request_body = {
                "model": judge_model,
                "messages": [
                    # Assuming a system prompt might be part of the template or added here if needed
                    # {"role": "system", "content": "You are a helpful assistant..."}
                    {"role": "user", "content": judge_prompt_formatted},
                ],
                "temperature": judge_temperature,
                # Add other parameters like max_tokens if needed from config
            }

            batch_request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": request_body,
            }
            judging_requests.append(batch_request)

        except Exception as e:
            logger.error(
                f"Unexpected error processing item {item.get('task_id', 'UNKNOWN')}_{item.get('model_id', 'UNKNOWN')}: {e}",
                exc_info=True,
            )
            skipped_items_count += 1
            continue  # Skip to next item on unexpected error

    logger.info(f"Generated {len(judging_requests)} judging requests.")
    if skipped_items_count > 0:
        logger.warning(
            f"Skipped {skipped_items_count} items due to errors or failed structuring."
        )

    # --- Submit Judging Batch ---
    batch_id = None
    if not judging_requests:
        logger.warning(
            "No valid judging requests were generated. Skipping batch submission."
        )
    else:
        try:
            logger.info("Formatting requests to JSONL...")
            jsonl_content = format_requests_to_jsonl(judging_requests)
            logger.info("Uploading batch file...")
            file_id = await upload_batch_file(openai_client, jsonl_content)

            if file_id:
                logger.info(f"Batch file uploaded successfully. File ID: {file_id}")
                logger.info("Creating batch job...")
                batch_id = await create_batch_job(openai_client, file_id)
                if batch_id:
                    logger.info(f"Batch job created successfully. Batch ID: {batch_id}")
                else:
                    logger.error("Failed to create batch job.")
            else:
                logger.error("Failed to upload batch file.")

        except Exception as e:
            logger.critical(
                f"An error occurred during batch submission: {e}", exc_info=True
            )

    if not batch_id:  # Check if batch_id is still None after the try block
        logger.warning("Judging batch job submission failed or was skipped.")

    logger.info("Judging batch preparation process finished.")


if __name__ == "__main__":
    asyncio.run(main())
