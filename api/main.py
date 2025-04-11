# CogniBench - API Layer
# Version: 0.1 (Phase 3 - Basic Setup)

import json
import logging  # Added
import os
import sys  # Added
import uuid
from datetime import datetime
from pathlib import Path  # Added
from typing import Any, Dict, Optional  # Added

# Assuming workflow and config loading utilities exist
from core.workflow import run_evaluation_workflow
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader

# Setup logger for this module
# Note: FastAPI has its own logging, but we can use this for config loading etc.
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # Basic config for startup logs

# Import schemas and potentially core functions if needed later
from .schemas import EvaluationRequest, EvaluationResponse, EvaluationResultData

# --- Configuration Loading and Validation ---
# Determine config path relative to this file (api/main.py -> CogniBench/config.yaml)
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
APP_CONFIG: Optional[Dict[str, Any]] = None  # Global variable to hold validated config


def load_config(config_path: Path) -> Optional[Dict[str, Any]]:
    """Loads configuration from a YAML file."""
    if not config_path.is_file():
        logger.error("Configuration file not found at %s", config_path)
        return None
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


def validate_config(config: Optional[Dict[str, Any]]) -> bool:
    """Performs basic validation on the loaded configuration dictionary."""
    if not config:  # Check if config is None or empty
        logger.error(
            "Config validation failed: Configuration is empty or failed to load."
        )
        return False

    required_sections = [
        "llm_client",
        "evaluation_settings",
        "output_options",
    ]  # Added output_options
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

    # Validate output_options has results_file (used by get endpoint)
    output_options = config["output_options"]
    if "results_file" not in output_options or not isinstance(
        output_options["results_file"], str
    ):
        logger.error(
            "Config validation failed: Missing or invalid 'results_file' key in 'output_options'."
        )
        return False

    logger.info("Configuration validation successful.")
    return True


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


# --- FastAPI Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Load and validate configuration on API startup."""
    global APP_CONFIG
    logger.info("Loading API configuration from: %s", CONFIG_PATH)
    loaded_config = load_config(CONFIG_PATH)
    if validate_config(loaded_config):
        APP_CONFIG = loaded_config
        logger.info("API Configuration loaded and validated successfully.")
    else:
        APP_CONFIG = None  # Ensure config is None if validation fails
        logger.error(
            "API startup failed due to invalid configuration. Please check config.yaml."
        )
        # Optionally, raise an exception here to prevent startup,
        # but logging the error might be sufficient depending on desired behavior.
        # raise RuntimeError("API Configuration failed validation.")


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
):
    """
    Accepts evaluation request data containing IDs for existing prompt, model response,
    and ideal response. Triggers the evaluation workflow asynchronously in the background.
    Returns a message indicating the evaluation has been queued.
    """
    print(
        f"Received evaluation request for prompt_id: {request.prompt_id}, response_id: {request.model_response_id}"
    )

    # Generate a temporary ID for logging this specific API request/queueing action
    api_request_id = f"api_req_{uuid.uuid4()}"
    print(f"API Request {api_request_id}: Queuing evaluation workflow...")

    # Check if config loaded successfully during startup
    if APP_CONFIG is None:
        logger.error(
            "API Error: Server configuration is invalid or missing. Cannot process request."
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error: Invalid server configuration.",
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
            config=APP_CONFIG,  # Pass the loaded and validated config
            task_id=request.task_id,  # Pass optional task_id if provided
            model_id=request.model_id,  # Pass optional model_id if provided
            # Determine output path from config or default
            # Note: run_evaluation_workflow handles saving to JSONL now
            # output_jsonl_path=Path(APP_CONFIG["output_options"]["results_file"]).parent / f"{APP_CONFIG['output_options']['results_file']}.jsonl" # Example construction
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
async def get_evaluation_result(evaluation_id: str):
    """
    Retrieves the results of a specific evaluation run by its ID.
    Assumes the evaluation has been processed offline and saved to evaluations.json.
    """
    print(f"Received request for evaluation_id: {evaluation_id}")
    # Determine the results file path from config (validated at startup)
    # Use Path object for consistency
    results_file_path = Path(
        APP_CONFIG["output_options"]["results_file"]
    )  # Assumes validation passed

    if not results_file_path.is_file():
        logger.warning("Evaluations result file '%s' not found.", results_file_path)
        raise HTTPException(status_code=404, detail=f"Evaluations data file not found.")

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
