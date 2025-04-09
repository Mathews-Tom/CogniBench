# CogniBench - Evaluation Workflow
# Version: 0.2 (Phase 4 - External Prompts & Enhanced Postprocessing)

import functools
import json
import os
import uuid
from typing import Any, Dict, List, Optional

from .llm_clients.base import BaseLLMClient  # Import base class
from .llm_clients.openai_client import (
    OpenAIClient,
)  # Import specific client for default
from .output_writer import save_evaluation_result
from .postprocessing import perform_postprocessing  # Updated import

# Import core components
from .preprocessing import extract_final_answer, normalize_text_formats

# Removed: from .prompt_templates import FULL_L1_JUDGE_PROMPT_TEMPLATE
from .response_parser import parse_judge_response

# --- Configuration (Consider moving to a dedicated config module later) ---
# Define base directory relative to this file
BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
)  # Goes up one level from core to CogniBench
DATA_DIR = os.path.join(BASE_DIR, "data")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")  # New prompts directory

PROMPTS_FILE = os.path.join(DATA_DIR, "prompts.json")  # Input prompts data
MODEL_RESPONSES_FILE = os.path.join(DATA_DIR, "model_responses.json")
IDEAL_RESPONSES_FILE = os.path.join(DATA_DIR, "ideal_responses.json")

# Default LLM settings (can be overridden)
DEFAULT_JUDGE_LLM_PROVIDER = "openai"
DEFAULT_JUDGE_LLM_MODEL = "gpt-4o"
DEFAULT_JUDGE_PROMPT_VERSION = "v0.2-full-L1"


# --- Helper Functions ---
@functools.lru_cache(maxsize=None)  # Cache loaded data in memory
def load_json_data(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """Loads data from a JSON file."""
    if not os.path.exists(file_path):
        print(f"Workflow Error: Data file not found at {file_path}")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Workflow Error loading data from {file_path}: {e}")
        return None


# Removed find_item_by_id as we will use dict lookups


# Helper to create indexed dictionaries from loaded lists
@functools.lru_cache(maxsize=8)  # Cache a few indexed dicts
def _index_data_by_id(data_list_tuple: tuple, id_key: str) -> Dict[str, Any]:
    """Converts a list of dicts (passed as tuple for caching) to a dict indexed by id_key."""
    if not data_list_tuple:
        return {}
    return {item.get(id_key): item for item in data_list_tuple if item.get(id_key)}

    # These lines were remnants of the old find_item_by_id function and are now removed.
    return None


@functools.lru_cache(maxsize=32)  # Cache loaded prompt templates
def load_prompt_template(version: str) -> Optional[str]:
    """Loads a specific prompt template version from the prompts directory."""
    # Basic sanitization of version to prevent path traversal
    safe_version = os.path.basename(version)
    template_filename = f"{safe_version}.txt"  # Assuming .txt extension
    template_path = os.path.join(PROMPTS_DIR, template_filename)

    if not os.path.exists(template_path):
        print(f"Workflow Error: Prompt template file not found at {template_path}")
        return None
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    except IOError as e:
        print(f"Workflow Error loading prompt template {template_path}: {e}")
        return None


# --- Workflow Function ---


def run_evaluation_workflow(
    prompt_id: str,
    response_id: str,
    ideal_response_id: str,
    judge_llm_provider: str = DEFAULT_JUDGE_LLM_PROVIDER,
    judge_llm_model: str = DEFAULT_JUDGE_LLM_MODEL,
    judge_prompt_version: str = DEFAULT_JUDGE_PROMPT_VERSION,
    llm_client: Optional[
        BaseLLMClient
    ] = None,  # Allow passing a pre-initialized client
) -> Dict[str, Any]:
    """
    Executes the end-to-end evaluation workflow for a single response.

    Args:
        prompt_id: ID of the prompt to use.
        response_id: ID of the model response to evaluate.
        ideal_response_id: ID of the ideal response for comparison.
        judge_llm_provider: The provider of the Judge LLM (e.g., "openai").
        judge_llm_model: The specific model name of the Judge LLM.
        judge_prompt_version: The version of the prompt template used.
        llm_client: An optional pre-initialized LLM client instance.

    Returns:
        A dictionary containing the status and results of the workflow.
        Example success: {"status": "success", "evaluation_id": "eval_...", "result": {...}}
        Example failure: {"status": "error", "message": "Error details..."}
    """
    print(f"\n--- Starting Workflow for Response ID: {response_id} ---")
    workflow_result = {"status": "error", "message": "Workflow did not complete."}

    # 1. Load Data
    print("Workflow Step: Loading and Indexing data...")
    # Load data (will hit cache after first load)
    prompts_list = load_json_data(PROMPTS_FILE)
    model_responses_list = load_json_data(MODEL_RESPONSES_FILE)
    ideal_responses_list = load_json_data(IDEAL_RESPONSES_FILE)

    if not all([prompts_list, model_responses_list, ideal_responses_list]):
        workflow_result["message"] = "Failed to load necessary data files."
        print(f"Error: {workflow_result['message']}")  # Added print
        return workflow_result

    # Index data for fast lookup (will also hit cache)
    # Note: Convert lists to tuples for caching as lists are not hashable
    try:
        prompts_dict = _index_data_by_id(tuple(prompts_list), "prompt_id")
        model_responses_dict = _index_data_by_id(
            tuple(model_responses_list), "response_id"
        )
        ideal_responses_dict = _index_data_by_id(
            tuple(ideal_responses_list), "ideal_response_id"
        )
    except Exception as e:
        workflow_result["message"] = f"Failed to index loaded data: {e}"
        print(f"Error: {workflow_result['message']}")  # Added print
        return workflow_result

    # Retrieve specific items using dictionary lookup
    prompt_data = prompts_dict.get(prompt_id)
    model_response_data = model_responses_dict.get(response_id)
    ideal_response_data = ideal_responses_dict.get(ideal_response_id)

    if not all([prompt_data, model_response_data, ideal_response_data]):
        missing = []
        if not prompt_data:
            missing.append(f"prompt_id={prompt_id}")
        if not model_response_data:
            missing.append(f"response_id={response_id}")
        if not ideal_response_data:
            missing.append(f"ideal_response_id={ideal_response_id}")
        workflow_result["message"] = (
            f"Could not find specified data IDs in cached/indexed data: {', '.join(missing)}."
        )
        print(f"Error: {workflow_result['message']}")  # Added print
        return workflow_result

    # Basic validation (can be expanded)
    if (
        model_response_data.get("prompt_id") != prompt_id
        or ideal_response_data.get("prompt_id") != prompt_id
    ):
        print(
            f"Workflow Warning: Mismatch between prompt_id ({prompt_id}), response prompt_id ({model_response_data.get('prompt_id')}), and ideal response prompt_id ({ideal_response_data.get('prompt_id')})."
        )
        # Decide if this should be a fatal error for the workflow

    # 2. Preprocessing
    print("Workflow Step: Preprocessing...")
    norm_model_response_text = normalize_text_formats(
        model_response_data["response_text"]
    )
    norm_ideal_response_text = normalize_text_formats(
        ideal_response_data["response_text"]
    )
    norm_prompt_content = prompt_data["content"]  # Assuming prompt is clean
    norm_correct_answer = normalize_text_formats(ideal_response_data["correct_answer"])
    extracted_answer = extract_final_answer(norm_model_response_text)
    print(f"  - Extracted Answer: {extracted_answer}")

    # 3. Initialize LLM Client (if not provided)
    if llm_client is None:
        print(f"Workflow Step: Initializing LLM client ({judge_llm_provider})...")
        if judge_llm_provider == "openai":
            try:
                llm_client = OpenAIClient()  # Assumes API key in env/.env
            except Exception as e:
                workflow_result["message"] = f"Failed to initialize OpenAI client: {e}"
                return workflow_result
        else:
            workflow_result["message"] = (
                f"Unsupported LLM provider '{judge_llm_provider}'."
            )
            return workflow_result

    # 4. Load and Format Prompt
    print(
        f"Workflow Step: Loading prompt template (version: {judge_prompt_version})..."
    )
    prompt_template = load_prompt_template(judge_prompt_version)
    if prompt_template is None:
        workflow_result["message"] = (
            f"Failed to load prompt template version '{judge_prompt_version}'."
        )
        return workflow_result

    print("Workflow Step: Formatting prompt...")
    try:
        filled_prompt = prompt_template.format(
            prompt_content=norm_prompt_content,
            model_response_text=norm_model_response_text,
            ideal_response_text=norm_ideal_response_text,
            correct_answer=norm_correct_answer,
        )
    except KeyError as e:
        workflow_result["message"] = (
            f"Error formatting prompt template '{judge_prompt_version}': Missing key {e}."
        )
        return workflow_result
    except Exception as e:  # Catch other potential formatting errors
        workflow_result["message"] = (
            f"Unexpected error formatting prompt template '{judge_prompt_version}': {e}."
        )
        return workflow_result

    # 5. Invoke LLM Judge
    print(f"Workflow Step: Invoking Judge LLM ({judge_llm_model})...")
    judge_system_prompt = "You are an expert mathematician and rigorous evaluator assessing an AI model's response."
    llm_response = llm_client.invoke(
        prompt=filled_prompt,
        model_name=judge_llm_model,
        system_prompt=judge_system_prompt,
        temperature=0.0,
    )
    if "error" in llm_response:
        workflow_result["message"] = f"LLM Invocation failed: {llm_response['error']}"
        return workflow_result
    raw_llm_output_content = llm_response.get("raw_content", "")

    # 6. Parse Response
    print("Workflow Step: Parsing LLM response...")
    parsed_data = parse_judge_response(raw_llm_output_content)
    # Parsing errors are now handled within postprocessing, but we can log here
    if "error" in parsed_data:
        print(f"  - Parsing Warning/Error: {parsed_data['error']}")
        # Continue to postprocessing, which will flag for review
    # parsed_scores = parsed_data.get("evaluation", {}) # This is handled by postprocessing now

    # 7. Postprocessing
    # 7. Postprocessing (using the enhanced function)
    print("Workflow Step: Postprocessing...")
    postprocessing_results = perform_postprocessing(
        parsed_judge_response=parsed_data,  # Pass the full parser output
        extracted_final_answer=extracted_answer,
        correct_final_answer=norm_correct_answer,
    )
    print(
        f"  - Final Answer Verification: {postprocessing_results.get('verification_message')}"
    )
    print(f"  - Aggregated Score: {postprocessing_results.get('aggregated_score')}")
    print(f"  - Needs Human Review: {postprocessing_results.get('needs_human_review')}")
    if postprocessing_results.get("review_reasons"):
        print(
            f"  - Review Reasons: {'; '.join(postprocessing_results.get('review_reasons', []))}"
        )

    # 8. Save Result
    print("Workflow Step: Saving evaluation result...")
    evaluation_id = f"eval_{uuid.uuid4()}"
    save_status = save_evaluation_result(
        evaluation_id=evaluation_id,
        response_id=response_id,
        ideal_response_id=ideal_response_id,
        judge_llm_model=judge_llm_model,
        judge_prompt_template_version=judge_prompt_version,
        raw_judge_output={"raw_content": raw_llm_output_content},  # Keep raw output
        parsed_rubric_scores=parsed_data.get(
            "evaluation", {}
        ),  # Pass parsed scores if available
        # Pass fields from the enhanced postprocessing results
        final_answer_verified=postprocessing_results.get("final_answer_verified"),
        aggregated_score=postprocessing_results.get("aggregated_score"),
        needs_human_review=postprocessing_results.get(
            "needs_human_review", False
        ),  # Default to False if missing
        review_reasons=postprocessing_results.get(
            "review_reasons"
        ),  # Pass the list of reasons
    )

    if save_status.get("status") == "success":
        print(f"--- Workflow Complete for Response ID: {response_id} ---")
        workflow_result = {
            "status": "success",
            "evaluation_id": evaluation_id,
            "result": save_status,  # Contains the saved data structure implicitly
        }
    else:
        workflow_result["message"] = (
            f"Failed to save evaluation: {save_status.get('message')}"
        )

    return workflow_result


if __name__ == "__main__":
    test_prompt_id = "math_prompt_001"
    test_response_id = "resp_001_modelA"
    test_ideal_response_id = "ideal_resp_001"

    result = run_evaluation_workflow(
        prompt_id=test_prompt_id,
        response_id=test_response_id,
        ideal_response_id=test_ideal_response_id,
    )

    print("\n--- Workflow Result ---")
    print(json.dumps(result, indent=2))
    print("-----------------------")
    print("-----------------------")
