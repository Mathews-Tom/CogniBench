# CogniBench - Evaluation Workflow
# Version: 0.1 (Phase 2 - Basic Orchestration)

import json
import os
import uuid
from typing import Any, Dict, Optional

from .llm_clients.base import BaseLLMClient  # Import base class
from .llm_clients.openai_client import (
    OpenAIClient,  # Import specific client for default
)
from .output_writer import save_evaluation_result
from .postprocessing import aggregate_scores, verify_final_answer

# Import core components
from .preprocessing import extract_final_answer, normalize_text_formats
from .prompt_templates import FULL_L1_JUDGE_PROMPT_TEMPLATE
from .response_parser import parse_judge_response

# --- Configuration (Consider moving to a dedicated config module later) ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROMPTS_FILE = os.path.join(DATA_DIR, "prompts.json")
MODEL_RESPONSES_FILE = os.path.join(DATA_DIR, "model_responses.json")
IDEAL_RESPONSES_FILE = os.path.join(DATA_DIR, "ideal_responses.json")

# Default LLM settings (can be overridden)
DEFAULT_JUDGE_LLM_PROVIDER = "openai"
DEFAULT_JUDGE_LLM_MODEL = "gpt-4o"
DEFAULT_JUDGE_PROMPT_VERSION = "v0.2-full-L1"


# --- Helper Functions (Copied from run_single_evaluation.py - consider utils module) ---
def load_json_data(file_path):
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


def find_item_by_id(data_list, id_key, target_id):
    """Finds an item in a list of dictionaries by its ID."""
    if not data_list:
        return None
    for item in data_list:
        if item.get(id_key) == target_id:
            return item
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
    print("Workflow Step: Loading data...")
    prompts = load_json_data(PROMPTS_FILE)
    model_responses = load_json_data(MODEL_RESPONSES_FILE)
    ideal_responses = load_json_data(IDEAL_RESPONSES_FILE)
    if not all([prompts, model_responses, ideal_responses]):
        workflow_result["message"] = "Failed to load necessary data files."
        return workflow_result

    prompt_data = find_item_by_id(prompts, "prompt_id", prompt_id)
    model_response_data = find_item_by_id(model_responses, "response_id", response_id)
    ideal_response_data = find_item_by_id(
        ideal_responses, "ideal_response_id", ideal_response_id
    )
    if not all([prompt_data, model_response_data, ideal_response_data]):
        workflow_result["message"] = "Could not find specified data IDs in JSON files."
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

    # 4. Format Prompt
    print("Workflow Step: Formatting prompt...")
    try:
        filled_prompt = FULL_L1_JUDGE_PROMPT_TEMPLATE.format(
            prompt_content=norm_prompt_content,
            model_response_text=norm_model_response_text,
            ideal_response_text=norm_ideal_response_text,
            correct_answer=norm_correct_answer,
        )
    except KeyError as e:
        workflow_result["message"] = f"Error formatting prompt: Missing key {e}."
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
    if "error" in parsed_data:
        workflow_result["message"] = f"Response Parsing failed: {parsed_data['error']}"
        # TODO: Decide how to handle parsing errors - save raw output anyway?
        return workflow_result
    parsed_scores = parsed_data.get("evaluation", {})

    # 7. Postprocessing
    print("Workflow Step: Postprocessing...")
    is_answer_correct = verify_final_answer(extracted_answer, norm_correct_answer)
    print(f"  - Final Answer Verification Result: {is_answer_correct}")
    aggregated_score_result = aggregate_scores(parsed_scores)
    print(f"  - Aggregated Score: {aggregated_score_result}")

    # 8. Save Result
    print("Workflow Step: Saving evaluation result...")
    evaluation_id = f"eval_{uuid.uuid4()}"
    save_status = save_evaluation_result(
        evaluation_id=evaluation_id,
        response_id=response_id,
        ideal_response_id=ideal_response_id,
        judge_llm_model=judge_llm_model,
        judge_prompt_template_version=judge_prompt_version,
        raw_judge_output={"raw_content": raw_llm_output_content},
        parsed_rubric_scores=parsed_scores,
        final_answer_verified=is_answer_correct,
        aggregated_score=aggregated_score_result,
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
