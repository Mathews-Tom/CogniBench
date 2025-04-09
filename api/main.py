# CogniBench - API Layer
# Version: 0.1 (Phase 3 - Basic Setup)

import json
import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader

# Import schemas and potentially core functions if needed later
from .schemas import (  # Added EvaluationResultData
    EvaluationRequest,
    EvaluationResponse,
    EvaluationResultData,
)

# from core.workflow import run_evaluation_workflow # Not calling directly in this phase

# --- Configuration (Should match workflow.py or be centralized) ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROMPTS_FILE = os.path.join(DATA_DIR, "prompts.json")
MODEL_RESPONSES_FILE = os.path.join(DATA_DIR, "model_responses.json")
IDEAL_RESPONSES_FILE = os.path.join(DATA_DIR, "ideal_responses.json")
EVALUATIONS_FILE = os.path.join(DATA_DIR, "evaluations.json")  # Need to read this now

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


# --- Helper function to append data to JSON list file ---
def append_to_json_file(file_path: str, data_item: dict):
    """Reads a JSON list file, appends an item, and writes it back."""
    items = []
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    items = json.load(f)
                    if not isinstance(items, list):
                        print(f"Warning: {file_path} is not a list. Overwriting.")
                        items = []
                except json.JSONDecodeError:
                    print(
                        f"Warning: Could not decode JSON from {file_path}. Overwriting."
                    )
                    items = []

        # Check for duplicates based on ID (simple check)
        id_key = list(data_item.keys())[0]  # Assume first key is the ID
        if any(item.get(id_key) == data_item[id_key] for item in items):
            print(
                f"Warning: Item with ID {data_item[id_key]} already exists in {file_path}. Skipping append."
            )
            # Or raise an error / update existing? For now, just skip.
            return

        items.append(data_item)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"API Error: File I/O error writing to {file_path}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write data to {os.path.basename(file_path)}",
        )
    except Exception as e:
        print(f"API Error: Unexpected error writing to {file_path}: {e}")
        raise HTTPException(
            status_code=500, detail="An internal error occurred while saving data."
        )


# --- API Endpoints ---


@app.post(
    "/evaluate",
    response_model=EvaluationResponse,
    tags=["Evaluation"],
    # dependencies=[Depends(get_api_key)], # Dependency added below
) # Apply security dependency
async def submit_evaluation(request: EvaluationRequest):
    """
    Accepts evaluation data, saves the input components (prompt, model response,
    ideal response) to their respective JSON files.
    Does NOT trigger the evaluation workflow directly in this phase.
    Returns a unique evaluation ID for tracking (though processing is offline).
    """
    print(
        f"Received evaluation request for prompt_id: {request.prompt_id}, response_id: {request.model_response_id}"
    )

    # Generate a unique ID for this evaluation *run* attempt
    # Note: This ID isn't directly linked to the workflow result yet in this phase
    evaluation_run_id = f"eval_api_{uuid.uuid4()}"

    try:
        # 1. Save Prompt Data
        prompt_record = {
            "prompt_id": request.prompt_id,
            "content": request.prompt_content,
            "metadata": request.prompt_metadata,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        append_to_json_file(PROMPTS_FILE, prompt_record)

        # 2. Save Model Response Data
        model_response_record = {
            "response_id": request.model_response_id,
            "prompt_id": request.prompt_id,
            "model_name": request.model_name,
            "response_text": request.model_response_text,
            "extracted_final_answer": None,  # Will be populated by offline workflow
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        append_to_json_file(MODEL_RESPONSES_FILE, model_response_record)

        # 3. Save Ideal Response Data
        ideal_response_record = {
            "ideal_response_id": request.ideal_response_id,
            "prompt_id": request.prompt_id,
            "response_text": request.ideal_response_text,
            "correct_answer": request.correct_answer,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        append_to_json_file(IDEAL_RESPONSES_FILE, ideal_response_record)

        # TODO: In a real system, this might enqueue a job (e.g., Celery, RQ, Kafka)
        # which would then pick up the data using the IDs and run the workflow.
        # For now, we just save the inputs.

        print(f"Successfully saved input data for evaluation run {evaluation_run_id}")
        return EvaluationResponse(
            status="submitted",
            message="Evaluation input data received and saved. Processing occurs offline.",
            evaluation_id=evaluation_run_id,  # Return the API submission ID
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
) # Apply security dependency
async def get_evaluation_result(evaluation_id: str):
    """
    Retrieves the results of a specific evaluation run by its ID.
    Assumes the evaluation has been processed offline and saved to evaluations.json.
    """
    print(f"Received request for evaluation_id: {evaluation_id}")
    evaluations = []
    if not os.path.exists(EVALUATIONS_FILE):
        raise HTTPException(status_code=404, detail=f"Evaluations data file not found.")

    try:
        with open(EVALUATIONS_FILE, "r", encoding="utf-8") as f:
            evaluations = json.load(f)
            if not isinstance(evaluations, list):
                raise HTTPException(
                    status_code=500, detail="Invalid evaluations data format."
                )
    except (json.JSONDecodeError, IOError) as e:
        raise HTTPException(
            status_code=500, detail=f"Error reading evaluations data: {e}"
        )

    # Find the evaluation by ID
    result_data = None
    for evaluation in evaluations:
        if evaluation.get("evaluation_id") == evaluation_id:
            result_data = evaluation
            break

    if result_data is None:
        raise HTTPException(
            status_code=404, detail=f"Evaluation ID '{evaluation_id}' not found."
        )

    # Validate data against the Pydantic model (optional but good practice)
    try:
        # We return the raw dict here, FastAPI handles validation against response_model
        return result_data
    except Exception as e:  # Catch potential issues if data is severely malformed before FastAPI validation
        print(
            f"API Error: Unexpected issue processing data for evaluation {evaluation_id}: {e}"
        )
        raise HTTPException(
            status_code=500, detail="Internal server error processing evaluation data."
        )


# To run the API locally (from the CogniBench directory):
# uvicorn api.main:app --reload
