# CogniBench/core/batch_processor.py
"""
Handles interactions with the OpenAI Batch API, including:
- Formatting requests to JSONL.
- Uploading batch files.
- Creating and monitoring batch jobs.
- Downloading and parsing results.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

# Use ABSOLUTE imports for all core modules
from core.llm_clients.openai_client import OpenAIClient

# Import specific types if needed for clarity, otherwise use Any/Dict
# from openai.types.batch import Batch

logger = logging.getLogger("backend")


def format_request_for_batch(
    custom_id: str, method: str, url: str, body: Dict[str, Any]
) -> Dict[str, Any]:
    """Helper to create the dictionary structure for a single batch request line."""
    return {"custom_id": custom_id, "method": method, "url": url, "body": body}


def format_requests_to_jsonl(requests: List[Dict[str, Any]]) -> str:
    """
    Formats a list of API request dictionaries into a JSONL string.

    Each dictionary in the input list should represent a single API request
    and conform to the structure expected by format_request_for_batch helper,
    or directly contain 'custom_id', 'method', 'url', 'body'.

    Args:
        requests: A list of dictionaries, each representing an API request.

    Returns:
        A string containing all requests formatted as JSON Lines.
        Returns an empty string if the input list is empty or if errors occur.
    """
    if not requests:
        logger.warning("Received empty request list for JSONL formatting.")
        return ""

    jsonl_lines = []
    for i, req_data in enumerate(requests):
        try:
            # Ensure the request has the required keys for the batch format
            batch_request_line = {
                "custom_id": req_data.get(
                    "custom_id", f"request_{i + 1}"
                ),  # Add default ID if missing
                "method": req_data.get("method", "POST"),
                "url": req_data.get("url", "/v1/chat/completions"),
                "body": req_data.get("body", {}),
            }
            if not batch_request_line["body"]:
                logger.warning(
                    f"Request {batch_request_line['custom_id']} has empty body."
                )

            json_line = json.dumps(batch_request_line, ensure_ascii=False)
            jsonl_lines.append(json_line)
        except (TypeError, ValueError) as e:
            logger.error(
                f"Error serializing request at index {i} to JSON: {e}. Request data: {req_data}",
                exc_info=True,
            )
            # Optionally skip this request or return empty string to indicate failure
            return ""  # Fail fast if any request cannot be serialized

    return "\n".join(jsonl_lines)


async def upload_batch_file(
    openai_client: OpenAIClient, jsonl_content: str
) -> Optional[str]:
    """
    Saves JSONL content to a temporary file and uploads it using the OpenAIClient.

    Args:
        openai_client: An instance of the OpenAIClient.
        jsonl_content: The JSONL string content to upload.

    Returns:
        The OpenAI file ID string if successful, otherwise None.
    """
    if not jsonl_content:
        logger.error("Cannot upload empty JSONL content.")
        return None

    file_id = None
    temp_file_path = None  # Initialize path variable
    try:
        # Create a temporary file that is automatically deleted
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".jsonl", encoding="utf-8", delete=False
        ) as temp_file:
            temp_file_path = temp_file.name
            logger.debug(f"Writing JSONL content to temporary file: {temp_file_path}")
            temp_file.write(jsonl_content)
            temp_file.flush()  # Ensure content is written before upload

            # Upload the file using the client
            logger.info(f"Uploading temporary file {temp_file_path} to OpenAI...")
            file_id = await openai_client.upload_file(
                file_path=temp_file_path, purpose="batch"
            )

    except IOError as e:
        logger.error(f"Error writing to temporary file: {e}", exc_info=True)
        return None
    except Exception as e:
        # Catch potential errors during upload_file call
        logger.error(f"Error during batch file upload: {e}", exc_info=True)
        return None
    finally:
        # Ensure temporary file is deleted even if upload fails
        if temp_file_path and os.path.exists(
            temp_file_path
        ):  # Check if path was assigned
            try:
                os.remove(temp_file_path)
                logger.debug(f"Removed temporary file: {temp_file_path}")
            except OSError as e:
                logger.error(f"Error removing temporary file {temp_file_path}: {e}")

    return file_id


async def create_batch_job(
    openai_client: OpenAIClient,
    file_id: str,
    endpoint: str = "/v1/chat/completions",
    completion_window: str = "24h",
) -> Optional[Any]:  # Return the Batch object or None
    """
    Creates an OpenAI batch job using a previously uploaded file ID.

    Args:
        openai_client: An instance of the OpenAIClient.
        file_id: The ID of the uploaded JSONL file.
        endpoint: The target API endpoint for the batch requests.
        completion_window: The requested completion window.

    Returns:
        The OpenAI Batch object if successful, otherwise None.
    """
    if not file_id:
        logger.error("Cannot create batch job without a file ID.")
        return None

    try:
        logger.info(
            f"Creating batch job with file_id={file_id}, endpoint={endpoint}, completion_window={completion_window}"
        )
        batch_job = await openai_client.create_batch(
            file_id=file_id, endpoint=endpoint, completion_window=completion_window
        )
        logger.info(
            f"Batch job creation initiated successfully: {batch_job.id if batch_job else 'N/A'}"
        )
        return batch_job
    except Exception as e:
        logger.error(
            f"Error creating batch job for file ID {file_id}: {e}", exc_info=True
        )
        return None


async def check_batch_status(
    openai_client: OpenAIClient, batch_id: str
) -> Optional[Dict[str, Any]]:
    """
    Checks the status of an OpenAI batch job.

    Args:
        openai_client: An instance of the OpenAIClient.
        batch_id: The ID of the batch job to check.

    Returns:
        A dictionary containing batch status information if successful, otherwise None.
        Example keys: 'id', 'status', 'output_file_id', 'error_file_id', 'request_counts'.
    """
    if not batch_id:
        logger.error("Cannot check status without a batch ID.")
        return None

    try:
        logger.debug(f"Retrieving status for batch ID: {batch_id}")
        batch_info = await openai_client.retrieve_batch(batch_id=batch_id)

        if batch_info:
            # Extract relevant fields into a standard dictionary
            status_dict = {
                "id": batch_info.id,
                "status": batch_info.status,
                "endpoint": batch_info.endpoint,
                "input_file_id": batch_info.input_file_id,
                "output_file_id": batch_info.output_file_id,
                "error_file_id": batch_info.error_file_id,
                "completion_window": batch_info.completion_window,
                "cancelled_at": batch_info.cancelled_at,
                "completed_at": batch_info.completed_at,
                "created_at": batch_info.created_at,
                "expired_at": batch_info.expired_at,
                "expires_at": batch_info.expires_at,
                "failed_at": batch_info.failed_at,
                "finalizing_at": batch_info.finalizing_at,
                "in_progress_at": batch_info.in_progress_at,
                "request_counts": batch_info.request_counts.model_dump()
                if batch_info.request_counts
                else None,  # Convert Pydantic model
                "metadata": batch_info.metadata,
                "errors": batch_info.errors.model_dump()
                if batch_info.errors
                else None,  # Convert Pydantic model
            }
            logger.debug(f"Batch {batch_id} status: {status_dict.get('status')}")
            return status_dict
        else:
            logger.warning(f"Received no information for batch ID: {batch_id}")
            return None
    except Exception as e:
        logger.error(
            f"Error retrieving status for batch ID {batch_id}: {e}", exc_info=True
        )
        return None


async def download_batch_result_file(
    openai_client: OpenAIClient, file_id: str, destination_path: str
) -> bool:
    """
    Downloads the result file (output or error) for a completed OpenAI batch job.

    Args:
        openai_client: An instance of the OpenAIClient.
        file_id: The ID of the file to download.
        destination_path: The local file path where the downloaded content should be saved.

    Returns:
        True if the file was successfully downloaded and saved, otherwise False.
    """
    if not file_id:
        logger.error("Cannot download file without a file ID.")
        return False

    logger.info(
        f"Attempting to download result file ID: {file_id} to {destination_path}"
    )
    try:
        # Ensure destination directory exists
        dest_dir = os.path.dirname(destination_path)
        if dest_dir:  # Check if dirname returned non-empty string
            os.makedirs(dest_dir, exist_ok=True)

        # Use the client's download method directly (assuming it handles content/URL)
        # Note: The previous implementation assumed get_file_content returned a URL.
        # The openai_client.py implementation should handle the actual download.
        # We pass the destination path directly to the client helper.
        # Assuming download_file_content exists in openai_client and handles download
        await openai_client.download_file_content(
            file_id=file_id, destination_path=destination_path
        )

        # Verify file was created (basic check)
        if os.path.exists(destination_path) and os.path.getsize(destination_path) > 0:
            logger.info(f"Successfully downloaded result file to: {destination_path}")
            return True
        elif os.path.exists(destination_path):
            logger.warning(f"Result file downloaded but is empty: {destination_path}")
            return (
                True  # Consider empty file a success? Or False? Let's say True for now.
            )
        else:
            logger.error(
                f"Result file download failed (file not found after attempt): {destination_path}"
            )
            return False

    except AttributeError:
        logger.error(
            "OpenAIClient does not have a 'download_file_content' method. Check implementation.",
            exc_info=True,
        )
        return False
    except Exception as e:
        logger.error(
            f"Error downloading or saving result file ID {file_id} to {destination_path}: {e}",
            exc_info=True,
        )
        return False


def parse_batch_result_file(result_file_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parses a downloaded OpenAI batch result file (JSONL format).

    Args:
        result_file_path: The local path to the downloaded JSONL result file.

    Returns:
        A list of successfully parsed result dictionaries, or None if the file
        cannot be read. Returns an empty list if the file is empty or all
        lines fail parsing.
    """
    logger.info(f"Parsing batch result file: {result_file_path}")
    if not os.path.exists(result_file_path):
        logger.error(f"Result file not found: {result_file_path}")
        return None

    parsed_results = []
    try:
        with open(result_file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                try:
                    result_item = json.loads(line)
                    parsed_results.append(result_item)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse JSON on line {i + 1} of {result_file_path}. Error: {e}. Line: '{line[:100]}...'"
                    )
                    # Optionally add an error placeholder to parsed_results
                    # parsed_results.append({"parse_error": True, "line": i+1, "content": line})
                    continue  # Skip lines that fail to parse

    except IOError as e:
        logger.error(f"Error reading result file {result_file_path}: {e}")
        return None
    except Exception as e:  # Catch other potential errors during file processing
        logger.error(
            f"Unexpected error processing file {result_file_path}: {type(e).__name__} - {e}"
        )
        return None  # Return None on unexpected file processing errors

    logger.info(
        f"Successfully parsed {len(parsed_results)} lines from {result_file_path}"
    )
    return parsed_results
