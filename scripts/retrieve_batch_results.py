#!/usr/bin/env python3
# CogniBench/scripts/retrieve_batch_results.py
"""
Retrieves and processes results from a completed OpenAI Batch API job.

This script checks the status of a given batch ID, downloads the results
file upon completion, parses the results, maps them back to the original
evaluation items using intermediate data, handles errors, and saves the
processed outputs (e.g., structured data or final evaluations).
"""

import argparse
import asyncio
import json  # Added for intermediate data loading
import logging
import os
import sys
import time  # Added for polling delay
import uuid  # Added for evaluation ID generation

# Adjust path to import core modules
# This might need refinement based on actual project structure
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Import core components
from core.batch_processor import (
    check_batch_status,
    create_batch_job,
    download_batch_result_file,
    format_requests_to_jsonl,
    parse_batch_result_file,
    upload_batch_file,
)
from core.config import load_config  # Needed for post-processing parameters
from core.llm_clients.openai_client import OpenAIClient
from core.output_writer import save_evaluation_result  # Added for final saving
from core.workflow import process_judging_output  # Needed for judging stage

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants for polling
POLLING_INTERVAL_SECONDS = 30
MAX_POLLING_ATTEMPTS = 480  # e.g., 480 * 30s = 4 hours


async def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Retrieve and process OpenAI Batch API results."
    )
    parser.add_argument(
        "--batch-id",
        "-b",
        required=True,
        help="OpenAI Batch ID for the job to retrieve.",
    )
    parser.add_argument(
        "--stage",
        "-s",
        required=True,
        choices=["structuring", "judging"],
        help="Processing stage ('structuring' or 'judging'). Determines how results are processed and saved.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        required=True,
        help="File path to save the processed output for this stage.",
    )
    parser.add_argument(
        "--intermediate-data-dir",
        "-i",
        default="./batch_intermediate_data",
        help="Directory containing intermediate data files (e.g., mapping files).",
    )
    parser.add_argument(
        "--config-file",
        "-c",
        default="./config.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=POLLING_INTERVAL_SECONDS,
        help=f"Polling interval in seconds (default: {POLLING_INTERVAL_SECONDS}).",
    )
    parser.add_argument(
        "--max-poll-attempts",
        type=int,
        default=MAX_POLLING_ATTEMPTS,
        help=f"Maximum number of polling attempts (default: {MAX_POLLING_ATTEMPTS}).",
    )

    args = parser.parse_args()
    logger.info("Starting batch result retrieval...")
    logger.info(f"  Batch ID: {args.batch_id}")
    logger.info(f"  Stage: {args.stage}")
    logger.info(f"  Output Path: {args.output_path}")
    logger.info(f"  Intermediate Data Dir: {args.intermediate_data_dir}")
    logger.info(f"  Config File: {args.config_file}")
    logger.info(f"  Poll Interval: {args.poll_interval}s")
    logger.info(f"  Max Poll Attempts: {args.max_poll_attempts}")

    # Use batch_id from args
    batch_id = args.batch_id

    # --- Initialize Clients/Config ---
    try:
        config = load_config(args.config_file)
        logger.info(f"Configuration loaded successfully from {args.config_file}")
    except FileNotFoundError:
        logger.critical(f"Configuration file not found at {args.config_file}. Exiting.")
        return
    except Exception as e:
        logger.critical(
            f"Error loading configuration from {args.config_file}: {e}. Exiting."
        )
        return

    openai_client = OpenAIClient()
    logger.info("OpenAI client initialized.")

    # --- Check Batch Status (with polling) ---
    logger.info(f"Checking status for batch: {batch_id}")
    status_info = None
    for attempt in range(args.max_poll_attempts):
        logger.info(f"Polling attempt {attempt + 1}/{args.max_poll_attempts}...")
        try:
            status_info = await check_batch_status(openai_client, batch_id)
        except Exception as e:
            logger.error(f"API error checking batch status: {e}")
            status_info = None  # Ensure status_info is None on exception

        if status_info is None:
            logger.warning("Failed to retrieve batch status (API error?). Retrying...")
            # Optional: Implement more robust retry logic or backoff here
        elif status_info.get("status") == "completed":
            logger.info(
                f"Batch {batch_id} completed successfully. Status info: {status_info}"
            )
            break
        elif status_info.get("status") in ["failed", "cancelled", "expired"]:
            logger.error(
                f"Batch job {batch_id} ended with status: {status_info.get('status')}. Details: {status_info}"
            )
            # TODO: Decide on final error handling (e.g., raise exception, specific exit code)
            return  # Exit script on terminal failure status
        else:
            logger.info(
                f"Attempt {attempt + 1}: Batch status is '{status_info.get('status', 'unknown')}'. Waiting {args.poll_interval}s..."
            )

        await asyncio.sleep(args.poll_interval)
    else:
        # This block executes if the loop completes without a 'break'
        logger.error(
            f"Batch job {batch_id} did not complete after {args.max_poll_attempts} attempts. Last known status: {status_info.get('status', 'unknown') if status_info else 'unavailable'}."
        )
        return  # Exit script if polling times out

    # Ensure status_info is available and status is completed before proceeding
    if not status_info or status_info.get("status") != "completed":
        logger.error("Exiting due to incomplete batch status after polling loop.")
        return

    # --- Download Results ---
    output_file_id = status_info.get("output_file_id")
    if not output_file_id:
        logger.error(
            f"Batch {batch_id} completed but no output_file_id found in status info: {status_info}"
        )
        return

    # Define a temporary path for the downloaded file
    # Consider using tempfile module for better temporary file management
    result_file_path = f"./temp_results_{batch_id}.jsonl"
    logger.info(f"Downloading result file {output_file_id} to {result_file_path}...")

    try:
        download_success = await download_batch_result_file(
            openai_client, output_file_id, result_file_path
        )
    except Exception as e:
        logger.error(f"Error during result file download: {e}")
        download_success = False

    if not download_success:
        logger.error(f"Failed to download result file for batch {batch_id}.")
        # Clean up potentially partially downloaded file?
        if os.path.exists(result_file_path):
            try:
                os.remove(result_file_path)
                logger.info(f"Removed potentially incomplete file: {result_file_path}")
            except OSError as e:
                logger.error(f"Error removing temporary file {result_file_path}: {e}")
        return
    logger.info(f"Successfully downloaded result file to {result_file_path}")

    # --- Parse Results ---
    logger.info(f"Parsing result file: {result_file_path}")
    parsed_results = None
    try:
        parsed_results = parse_batch_result_file(result_file_path)
    except Exception as e:
        logger.error(f"Error parsing result file {result_file_path}: {e}")
        # Decide if we should proceed or exit

    if parsed_results is None:
        logger.error("Parsing failed, result file might be corrupted or empty.")
        # Keep the downloaded file for inspection? Or delete?
        # For now, let's keep it and exit.
        return
    elif not parsed_results:
        logger.warning("Result file parsed, but contained no valid results.")
        # Decide if this is an error or just an empty batch result
    else:
        logger.info(f"Successfully parsed {len(parsed_results)} results from the file.")

    # Clean up the temporary file after successful parsing
    try:
        os.remove(result_file_path)
        logger.info(f"Removed temporary result file: {result_file_path}")
    except OSError as e:
        logger.error(f"Error removing temporary file {result_file_path}: {e}")
        # Log error but continue, as parsing was successful

    # Store parsed results for the next steps
    # (This variable 'parsed_results' will be used in subsequent tasks)

    # --- Load Intermediate Data ---
    # Task 17: Load the mapping file created during batch submission
    logger.info("Loading intermediate data map...")
    intermediate_data_map = None
    intermediate_file_name = f"intermediate_data_{args.batch_id}.json"
    intermediate_data_path = os.path.join(
        args.intermediate_data_dir, intermediate_file_name
    )

    if not os.path.exists(intermediate_data_path):
        logger.critical(
            f"Intermediate data file not found at expected path: {intermediate_data_path}. "
            f"This file is required to map results back to original tasks. "
            f"Ensure the directory '{args.intermediate_data_dir}' contains the file '{intermediate_file_name}'."
        )
        # Cannot proceed without the mapping
        return
    else:
        try:
            with open(intermediate_data_path, "r") as f:
                intermediate_data_map = json.load(f)
            logger.info(
                f"Successfully loaded intermediate data from {intermediate_data_path}"
            )
            # Optional: Add a check if the loaded map is empty or not a dict if needed
            if not isinstance(intermediate_data_map, dict) or not intermediate_data_map:
                logger.warning(
                    f"Intermediate data file {intermediate_data_path} loaded, but it's empty or not a dictionary."
                )
                # Depending on requirements, might need to treat this as an error too
                # return # Uncomment if an empty map is critical

        except FileNotFoundError:
            # This case should theoretically be caught by os.path.exists, but good practice to include
            logger.critical(
                f"Intermediate data file disappeared unexpectedly: {intermediate_data_path}"
            )
            return
        except IOError as e:
            logger.critical(
                f"Error reading intermediate data file {intermediate_data_path}: {e}"
            )
            return
        except json.JSONDecodeError as e:
            logger.critical(
                f"Error decoding JSON from intermediate data file {intermediate_data_path}: {e}"
            )
            return
        except Exception as e:
            logger.critical(
                f"An unexpected error occurred while loading intermediate data from {intermediate_data_path}: {e}"
            )
            return

    # --- Map Results & Handle Errors ---
    # Task 18: Map parsed results back to original data using the intermediate map.
    # Task 19 (partially addressed by capturing error): Basic error capture.
    logger.info("Mapping batch results to original data...")
    processed_items = []

    if not parsed_results:
        logger.warning("No parsed results available to process. Skipping mapping.")
        # Depending on requirements, might need different handling if intermediate_data_map exists but results are empty.
    elif not intermediate_data_map:
        logger.error(
            "Intermediate data map is not loaded. Cannot map results. Exiting."
        )
        # No point continuing if the map is missing.
        return
    else:
        results_mapped = 0
        results_missing_custom_id = 0
        results_orphaned = 0

        for result_item in parsed_results:
            custom_id = result_item.get("custom_id")

            if not custom_id:
                logger.error(f"Found a result item missing 'custom_id': {result_item}")
                results_missing_custom_id += 1
                continue  # Skip this item

            original_data = intermediate_data_map.get(custom_id)

            if original_data:
                # Found the corresponding original data
                combined_item = {}
                # Copy all original data first
                combined_item.update(original_data)

                # Extract response and error from the batch result
                response_data = result_item.get("response")
                error_data = result_item.get(
                    "error"
                )  # Explicit error object from OpenAI

                # Initialize fields consistently
                combined_item["status"] = "unknown"  # Default status, will be updated
                combined_item["batch_response_status_code"] = None
                combined_item["batch_response_body"] = None
                combined_item["batch_error"] = None  # Explicitly initialize

                # --- Process Response and Check for HTTP Errors ---
                http_status_code = None
                has_http_error = False
                if response_data and isinstance(response_data, dict):
                    http_status_code = response_data.get("status_code")
                    combined_item["batch_response_status_code"] = http_status_code
                    combined_item["batch_response_body"] = response_data.get(
                        "body"
                    )  # Capture body regardless of status
                    if http_status_code is not None and http_status_code != 200:
                        has_http_error = True
                elif response_data is not None:
                    # Log if response exists but isn't the expected dictionary format
                    logger.warning(
                        f"Unexpected format for 'response' field for custom_id '{custom_id}'. Expected dict, got: {type(response_data)}. Value: {response_data}"
                    )
                    # Consider treating this as an error depending on strictness requirements
                    # combined_item["status"] = "error"
                    # combined_item["batch_error"] = {"code": "unexpected_response_format", "message": f"Response was type {type(response_data)}"}

                # --- Check for Explicit Batch Errors ---
                has_explicit_error = error_data is not None

                # --- Determine Final Status and Log ---
                if has_explicit_error:
                    combined_item["status"] = "error"
                    combined_item["batch_error"] = (
                        error_data  # Store the actual error object
                    )
                    logger.error(
                        f"Explicit batch error for custom_id '{custom_id}'. "
                        f"Code: {error_data.get('code', 'N/A')}, "
                        f"Message: {error_data.get('message', 'N/A')}"
                    )
                    # Note: Response might still contain useful info even with an error
                elif has_http_error:
                    combined_item["status"] = "error"
                    # No explicit 'error' object, but HTTP status indicates failure.
                    # Log details from the response. batch_error remains None unless we synthesize one.
                    logger.error(
                        f"HTTP error status ({http_status_code}) received for custom_id '{custom_id}'. "
                        f"Response Body (if any): {combined_item.get('batch_response_body', 'N/A')}"
                    )
                    # Optional: Synthesize an error object if needed downstream
                    # combined_item["batch_error"] = {"code": "http_error", "message": f"Received status code {http_status_code}"}
                else:
                    # No explicit error and HTTP status is 200 (or None/missing, treated as success if no explicit error)
                    combined_item["status"] = "success"
                    # batch_error remains None (initialized above)

                processed_items.append(combined_item)
                results_mapped += 1
            else:
                # custom_id from result file not found in the intermediate map
                logger.warning(
                    f"Orphaned result found: custom_id '{custom_id}' from batch results "
                    f"was not found in the intermediate data map ({intermediate_data_path}). "
                    f"Result details: {result_item}"
                )
                results_orphaned += 1
                # Optionally store orphaned results separately if needed for debugging
                # orphaned_results.append(result_item)

        logger.info(f"Mapping complete. Processed {len(parsed_results)} result items:")
        logger.info(f"  - Successfully mapped: {results_mapped}")
        logger.info(f"  - Results missing custom_id: {results_missing_custom_id}")
        logger.info(f"  - Orphaned results (custom_id not in map): {results_orphaned}")

    # 'processed_items' now holds the combined data.
    # Task 19 (further refinement) and Task 20 will use this list.

    # --- Save Processed Outputs / Finalize ---
    if args.stage == "structuring":
        logger.info(
            f"Saving structured outputs for stage '{args.stage}' to: {args.output_path}"
        )
        try:
            # Ensure the output directory exists
            output_dir = os.path.dirname(args.output_path)
            if output_dir:  # Only create if path includes a directory
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Ensured output directory exists: {output_dir}")

            # Write the processed items to the output file
            with open(args.output_path, "w") as f:
                json.dump(processed_items, f, indent=4)
            logger.info(f"Successfully saved structured outputs to {args.output_path}")

        except OSError as e:
            logger.critical(f"Error creating output directory {output_dir}: {e}")
            return  # Exit if directory creation fails
        except IOError as e:
            logger.critical(
                f"Error writing structured outputs to file {args.output_path}: {e}"
            )
            return  # Exit if file writing fails
        except Exception as e:
            logger.critical(f"An unexpected error occurred during saving: {e}")
            return  # Exit on unexpected errors during saving

    elif args.stage == "judging":
        logger.info("Processing judging results...")
        # 'processed_items' contains the raw batch results mapped to original data

        final_results = []
        items_processed = 0
        items_skipped_due_to_error = 0
        items_postprocessing_failed = 0

        # Extract necessary config values once
        try:
            expected_criteria = config["evaluation"]["judging"]["criteria"]
            allowed_scores = config["evaluation"]["judging"]["allowed_scores"]
            logger.info("Extracted judging criteria and allowed scores from config.")
        except KeyError as e:
            logger.critical(
                f"Missing required judging configuration key: {e}. Check config file: {args.config_file}"
            )
            return  # Cannot proceed without judging config

        for item in processed_items:
            custom_id = item.get("custom_id", "N/A")  # Get custom_id for logging

            if item.get("status") == "error":
                logger.error(
                    f"Skipping post-processing for item {custom_id} due to batch processing error: {item.get('batch_error')}"
                )
                # Append original item with error status to final results
                final_results.append(item)
                items_skipped_due_to_error += 1
                continue

            try:
                # Extract inputs for process_judging_output
                raw_judge_output = item.get("batch_response_body")
                # Assuming intermediate data contains these keys from structuring/preparation
                structured_model_response = item.get("structured_response")
                correct_final_answer = item.get("correct_final_answer")

                # Basic validation of required inputs from the item
                if raw_judge_output is None:
                    raise ValueError(
                        "Missing 'batch_response_body' (raw judge output) in item."
                    )
                if structured_model_response is None:
                    raise ValueError("Missing 'structured_response' in item.")
                # correct_final_answer might be optional depending on config/use case, but let's assume required for now
                if correct_final_answer is None:
                    raise ValueError("Missing 'correct_final_answer' in item.")

                # Call the post-processing function from core.workflow
                judging_processed_data = process_judging_output(
                    raw_judge_output=raw_judge_output,
                    expected_criteria=expected_criteria,
                    allowed_scores=allowed_scores,
                    structured_model_response=structured_model_response,
                    correct_final_answer=correct_final_answer,
                    config=config,  # Pass the whole config for flexibility
                )

                # Combine original item data with the processed judging data
                # Start with the original item, then update/add the processed fields
                final_item_result = item.copy()
                final_item_result.update(
                    judging_processed_data
                )  # Overwrites/adds keys like 'parsed_scores', 'final_score', 'reasoning', 'errors', etc.
                final_item_result["postprocessing_status"] = (
                    "success"  # Add status for this stage
                )
                # Add a unique evaluation ID
                final_item_result["evaluation_id"] = f"eval_{uuid.uuid4()}"

                final_results.append(final_item_result)
                items_processed += 1

            except Exception as e:
                logger.error(
                    f"Post-processing failed for item {custom_id}: {e}", exc_info=True
                )
                # Append original item with post-processing error status to final results
                item["postprocessing_status"] = "error"
                item["postprocessing_error"] = str(e)
                final_results.append(item)
                items_postprocessing_failed += 1
                continue  # Continue to the next item

        logger.info(
            f"Judging post-processing complete. Processed {items_processed} items."
        )
        if items_skipped_due_to_error > 0:
            logger.warning(
                f"Skipped {items_skipped_due_to_error} items due to initial batch errors."
            )
        if items_postprocessing_failed > 0:
            logger.warning(
                f"Post-processing failed for {items_postprocessing_failed} items."
            )

        # Save the final combined results
        logger.info(f"Saving final judging results to: {args.output_path}")
        try:
            # Ensure the output directory exists
            output_dir = os.path.dirname(args.output_path)
            if output_dir:  # Only create if path includes a directory
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Ensured output directory exists: {output_dir}")

            # Write the final results to the output file
            save_evaluation_result(
                final_results, args.output_path
            )  # Use the helper function
            logger.info(
                f"Successfully saved final judging results to {args.output_path}"
            )

        except OSError as e:
            logger.critical(f"Error creating output directory {output_dir}: {e}")
            sys.exit(1)  # Exit if directory creation fails
        except IOError as e:
            logger.critical(
                f"Error writing final judging results to file {args.output_path}: {e}"
            )
            sys.exit(1)  # Exit if file writing fails
        except Exception as e:
            logger.critical(
                f"An unexpected error occurred during saving final results: {e}"
            )
            sys.exit(1)  # Exit on unexpected errors during saving

    logger.info("Batch result retrieval and processing finished.")


if __name__ == "__main__":
    asyncio.run(main())
