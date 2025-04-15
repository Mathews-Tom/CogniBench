"""
CogniBench API Layer.

Provides a FastAPI interface for submitting evaluation tasks and retrieving results.
Handles API key authentication, background task processing for evaluations,
and interaction with the core CogniBench workflow and configuration.

Version: 0.1.1 (Phase 3 - Cleanup Pass)
"""

import json
import logging  # Added
import os
import uuid
from pathlib import Path  # Re-add Path import

from core.config import AppConfig, load_config  # Import new config loader and model
from core.log_setup import setup_logging  # Use relative import

# Assuming workflow and config loading utilities exist
from core.workflow import run_evaluation_workflow
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader

# Import schemas after other dependencies but before local code execution
from .schemas import EvaluationRequest, EvaluationResponse, EvaluationResultData

# Setup logging first
setup_logging()
# Note: FastAPI has its own logging, but we use 'backend' for our app logic.
logger = logging.getLogger("backend")
logging.basicConfig(level=logging.INFO)  # Basic config for startup logs

# --- Configuration Loading (Now handled by core.config) ---
# Old load_config and validate_config functions removed.


# --- Security Setup ---
load_dotenv()  # Load .env file for API_KEY
API_KEY_NAME = "X-API-KEY"  # Standard header name
API_KEY = os.getenv("API_KEY")  # Get the expected key from environment

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


async def get_api_key(key: str = Security(api_key_header)):
    """Dependency to verify the API key."""
    if not API_KEY:
        # This is a server configuration error
        print("API Security Error: API_KEY environment variable not set on the server.")
        raise HTTPException(
            status_code=500, detail="Internal server error: API Key not configured."
        )
    if key == API_KEY:
        return key
    else:
        raise HTTPException(status_code=403, detail="Could not validate credentials")


# --- End Security Setup ---

# Create FastAPI app instance
app = FastAPI(
    title="CogniBench API",
    description="API for submitting and managing CogniBench evaluations.",
    version="0.1.0",
)


# --- FastAPI Startup Event (Removed config loading) ---
# @app.on_event("startup")
# async def startup_event():
#     """Load and validate configuration on API startup."""
#     # Configuration is now loaded on demand using load_config() from core.config
#     # which includes caching.
#     pass # Keep startup event structure if needed for other things later


@app.get("/health", tags=["Status"])
async def health_check():
    """
    Simple health check endpoint to confirm the API is running.
    """
    return {"status": "ok"}


# Removed append_to_json_file helper function as it's no longer needed.
# Input data is assumed to exist, and output is handled by the workflow/output_writer.


# --- API Endpoints ---


@app.post(
    "/evaluate",
    response_model=EvaluationResponse,
    tags=["Evaluation"],
    dependencies=[Depends(get_api_key)],  # Apply security dependency
)
async def submit_evaluation(
    request: EvaluationRequest, background_tasks: BackgroundTasks
) -> EvaluationResponse:
    """
    Accept evaluation request data and trigger the workflow asynchronously.

    Accepts IDs for prompt, model response, and ideal response. Retrieves the actual
    data (currently using placeholders - TODO) and queues the core evaluation
    workflow using background tasks.

    Args:
        request: The evaluation request data containing necessary IDs.
        background_tasks: FastAPI background tasks manager.

    Returns:
        An EvaluationResponse indicating the task has been queued.

    Raises:
        HTTPException: If configuration fails to load or an internal error occurs.
    """
    print(
        f"Received evaluation request for prompt_id: {request.prompt_id}, response_id: {request.model_response_id}"
    )

    # Generate a temporary ID for logging this specific API request/queueing action
    api_request_id = f"api_req_{uuid.uuid4()}"
    print(f"API Request {api_request_id}: Queuing evaluation workflow...")

    # Load configuration on demand
    try:
        app_config: AppConfig = load_config()
    except Exception as e:  # Catch potential loading/validation errors
        logger.error("API Error: Failed to load server configuration.", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: Invalid server configuration. {e}",
        )

    try:
        # --- Data Retrieval (Placeholder - Needs Implementation) ---
        # TODO: Implement logic to retrieve actual prompt, response, and ideal_response
        #       text content based on the IDs provided in the request.
        #       This likely involves reading from files or a database based on how
        #       data is stored/managed.
        # Example Placeholder:
        prompt_text = f"Placeholder prompt for ID {request.prompt_id}"
        response_text = f"Placeholder response for ID {request.model_response_id}"
        ideal_response_text = (
            f"Placeholder ideal response for ID {request.ideal_response_id}"
        )
        logger.warning(
            "Using placeholder data for prompt/response text. Implement actual data retrieval."
        )
        # --- End Placeholder ---

        # Schedule the workflow to run in the background
        background_tasks.add_task(
            run_evaluation_workflow,
            prompt=prompt_text,  # Pass actual text content
            response=response_text,
            ideal_response=ideal_response_text,
            config=app_config,  # Pass the loaded and validated config object
            task_id=request.task_id,  # Pass optional task_id if provided
            model_id=request.model_id,  # Pass optional model_id if provided
            # output_jsonl_path is handled within the workflow based on config
        )

        print(f"API Request {api_request_id}: Evaluation workflow successfully queued.")
        # Return immediately, indicating the task is queued
        # Note: We don't have the final evaluation_id from the workflow yet.
        # The client would need to poll the GET endpoint later if they need the result.
        return EvaluationResponse(
            status="queued",
            message="Evaluation workflow successfully queued for background processing.",
            evaluation_id=None,  # No final ID available immediately
        )

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions from helper function
        raise http_exc
    except Exception as e:
        print(f"API Error in /evaluate endpoint: {e}")
        # Return a generic server error response
        raise HTTPException(
            status_code=500, detail=f"An internal server error occurred: {e}"
        )


@app.get(
    "/evaluate/{evaluation_id}",
    response_model=EvaluationResultData,
    tags=["Evaluation"],
    dependencies=[Depends(get_api_key)],  # Apply security dependency
)
async def get_evaluation_result(evaluation_id: str) -> EvaluationResultData:
    """
    Retrieve the results of a specific evaluation by its ID.

    Loads the evaluation results file (path defined in config) and searches
    for the entry matching the provided evaluation_id.

    Args:
        evaluation_id: The unique ID of the evaluation to retrieve.

    Returns:
        The EvaluationResultData for the specified ID.

    Raises:
        HTTPException: If the configuration fails, the results file is not found,
                       the ID is not found, or an error occurs during reading.
    """
    print(f"Received request for evaluation_id: {evaluation_id}")

    # Load configuration on demand
    try:
        app_config: AppConfig = load_config()
    except Exception as e:  # Catch potential loading/validation errors
        logger.error("API Error: Failed to load server configuration.", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: Invalid server configuration. {e}",
        )

    # Determine the results file path from the loaded config object
    # Use Path object for consistency, assuming paths in config are relative to project root
    project_root = Path(__file__).resolve().parent.parent
    results_file_path = project_root / app_config.output_options.results_file

    # Check if the resolved path exists
    if not results_file_path.is_file():
        logger.warning(
            "Evaluations result file not found at resolved path: %s", results_file_path
        )
        raise HTTPException(status_code=404, detail="Evaluations data file not found.")

    # Read JSONL file line by line
    try:
        with results_file_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    evaluation = json.loads(line)
                    if evaluation.get("evaluation_id") == evaluation_id:
                        # Found the evaluation, return it
                        # FastAPI will validate against the response_model
                        return evaluation
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping invalid JSON line in %s: %s",
                        results_file_path,
                        line.strip(),
                    )
                    continue  # Skip malformed lines
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Evaluations data file not found.")
    except IOError as e:
        raise HTTPException(
            status_code=500, detail=f"Error reading evaluations data file: {e}"
        )
    except Exception as e:  # Catch unexpected errors during processing
        logger.error(
            "API Error: Unexpected error reading %s: %s",
            results_file_path,
            e,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Internal server error reading evaluation data."
        )

    # If loop completes without finding the ID
    raise HTTPException(
        status_code=404, detail=f"Evaluation ID '{evaluation_id}' not found."
    )

    # Removed redundant try/except block and validation logic here.
    # The return statement above handles returning the found dict.
    # FastAPI's response_model handles the validation.


# To run the API locally (from the CogniBench directory):
# uvicorn api.main:app --reload
