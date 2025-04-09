# CogniBench - API Layer
# Version: 0.1 (Phase 3 - Basic Setup)

import json
import os
import uuid
from datetime import datetime

from core.workflow import \
    run_evaluation_workflow  # Import the workflow function
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader

# Import schemas and potentially core functions if needed later
from .schemas import (EvaluationRequest,  # Added EvaluationResultData
                      EvaluationResponse, EvaluationResultData)

# --- Configuration (Should match workflow.py or be centralized) ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROMPTS_FILE = os.path.join(DATA_DIR, "prompts.json")
MODEL_RESPONSES_FILE = os.path.join(DATA_DIR, "model_responses.json")
IDEAL_RESPONSES_FILE = os.path.join(DATA_DIR, "ideal_responses.json")
EVALUATIONS_FILE = os.path.join(DATA_DIR, "evaluations.jsonl") # Updated to .jsonl

# --- Security Setup ---
load_dotenv() # Load .env file for API_KEY
API_KEY_NAME = "X-API-KEY" # Standard header name
API_KEY = os.getenv("API_KEY") # Get the expected key from environment

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(key: str = Security(api_key_header)):
    """Dependency to verify the API key."""
    if not API_KEY:
        # This is a server configuration error
        print("API Security Error: API_KEY environment variable not set on the server.")
        raise HTTPException(
            status_code=500,
            detail="Internal server error: API Key not configured."
        )
    if key == API_KEY:
        return key
    else:
        raise HTTPException(
            status_code=403, detail="Could not validate credentials"
        )
# --- End Security Setup ---

# Create FastAPI app instance
app = FastAPI(
    title="CogniBench API",
    description="API for submitting and managing CogniBench evaluations.",
    version="0.1.0",
)


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
    dependencies=[Depends(get_api_key)], # Apply security dependency
)
async def submit_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
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

    try:
        # Schedule the workflow to run in the background
        # Pass necessary IDs and parameters from the request
        background_tasks.add_task(
            run_evaluation_workflow,
            prompt_id=request.prompt_id,
            response_id=request.model_response_id,
            ideal_response_id=request.ideal_response_id,
            # Optional: Pass judge model/prompt version from request if needed
            # judge_llm_model=request.judge_llm_model,
            # judge_prompt_version=request.judge_prompt_version,
        )

        print(f"API Request {api_request_id}: Evaluation workflow successfully queued.")
        # Return immediately, indicating the task is queued
        # Note: We don't have the final evaluation_id from the workflow yet.
        # The client would need to poll the GET endpoint later if they need the result.
        return EvaluationResponse(
            status="queued",
            message="Evaluation workflow successfully queued for background processing.",
            evaluation_id=None, # No final ID available immediately
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
    dependencies=[Depends(get_api_key)], # Apply security dependency
)
async def get_evaluation_result(evaluation_id: str):
    """
    Retrieves the results of a specific evaluation run by its ID.
    Assumes the evaluation has been processed offline and saved to evaluations.json.
    """
    print(f"Received request for evaluation_id: {evaluation_id}")
    evaluations = []
    if not os.path.exists(EVALUATIONS_FILE):
        raise HTTPException(status_code=404, detail=f"Evaluations data file not found.")

    # Read JSONL file line by line
    try:
        with open(EVALUATIONS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    evaluation = json.loads(line)
                    if evaluation.get("evaluation_id") == evaluation_id:
                        # Found the evaluation, return it
                        # FastAPI will validate against the response_model
                        return evaluation
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in {EVALUATIONS_FILE}: {line.strip()}")
                    continue # Skip malformed lines
    except FileNotFoundError:
         raise HTTPException(status_code=404, detail=f"Evaluations data file not found.")
    except IOError as e:
        raise HTTPException(
            status_code=500, detail=f"Error reading evaluations data file: {e}"
        )
    except Exception as e: # Catch unexpected errors during processing
         print(f"API Error: Unexpected error reading {EVALUATIONS_FILE}: {e}")
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
