# CogniBench - Single Evaluation Test Harness
# Version: 0.1 (Phase 1)

import json
import os
import uuid
from datetime import datetime

# Import core components
from core.llm_clients.openai_client import OpenAIClient
from core.output_writer import save_evaluation_result
from core.prompt_templates import (
    INITIAL_JUDGE_PROMPT_TEMPLATE,  # Using the initial template
)
from core.response_parser import parse_judge_response

# --- Configuration ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PROMPTS_FILE = os.path.join(DATA_DIR, "prompts.json")
MODEL_RESPONSES_FILE = os.path.join(DATA_DIR, "model_responses.json")
IDEAL_RESPONSES_FILE = os.path.join(DATA_DIR, "ideal_responses.json")

# Select which data point to test (using IDs from example files)
TEST_PROMPT_ID = "math_prompt_001"
TEST_RESPONSE_ID = "resp_001_modelA"  # Response to evaluate
TEST_IDEAL_RESPONSE_ID = "ideal_resp_001"  # Corresponding ideal response

# LLM Configuration
JUDGE_LLM_PROVIDER = "openai"  # Hardcoded for now, could be config driven later
JUDGE_LLM_MODEL = "gpt-4o"  # Or "gpt-4-turbo" or other suitable model
JUDGE_PROMPT_VERSION = "v0.1-initial"  # Matches the template used


# --- Helper Function to Load Data ---
def load_json_data(file_path):
    """Loads data from a JSON file."""
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


def find_item_by_id(data_list, id_key, target_id):
    """Finds an item in a list of dictionaries by its ID."""
    if not data_list:
        return None
    for item in data_list:
        if item.get(id_key) == target_id:
            return item
    return None


# --- Main Execution Logic ---
def main():
    print("--- Starting Single Evaluation Test ---")

    # 1. Load Data
    print("Loading data...")
    prompts = load_json_data(PROMPTS_FILE)
    model_responses = load_json_data(MODEL_RESPONSES_FILE)
    ideal_responses = load_json_data(IDEAL_RESPONSES_FILE)

    if not all([prompts, model_responses, ideal_responses]):
        print("Failed to load necessary data files. Exiting.")
        return

    prompt_data = find_item_by_id(prompts, "prompt_id", TEST_PROMPT_ID)
    model_response_data = find_item_by_id(
        model_responses, "response_id", TEST_RESPONSE_ID
    )
    ideal_response_data = find_item_by_id(
        ideal_responses, "ideal_response_id", TEST_IDEAL_RESPONSE_ID
    )

    if not all([prompt_data, model_response_data, ideal_response_data]):
        print("Could not find the specified test data IDs in the JSON files. Exiting.")
        return

    # Check if the response corresponds to the prompt
    if model_response_data.get("prompt_id") != prompt_data.get("prompt_id"):
        print(
            f"Warning: Model response {TEST_RESPONSE_ID} does not match prompt {TEST_PROMPT_ID}."
        )
        # Decide whether to exit or continue based on requirements
        # return

    # Check if the ideal response corresponds to the prompt
    if ideal_response_data.get("prompt_id") != prompt_data.get("prompt_id"):
        print(
            f"Warning: Ideal response {TEST_IDEAL_RESPONSE_ID} does not match prompt {TEST_PROMPT_ID}."
        )
        # return

    print("Data loaded successfully.")

    # 2. Initialize LLM Client
    print(f"Initializing LLM client for provider: {JUDGE_LLM_PROVIDER}")
    llm_client = None
    if JUDGE_LLM_PROVIDER == "openai":
        try:
            # Assumes OPENAI_API_KEY is set in environment
            llm_client = OpenAIClient()
        except (ValueError, Exception) as e:  # Catch init errors
            print(f"Error initializing OpenAI client: {e}")
            return
    else:
        print(f"Error: Unsupported LLM provider '{JUDGE_LLM_PROVIDER}' configured.")
        return

    # 3. Format Prompt (using the initial template)
    print("Formatting prompt...")
    try:
        filled_prompt = INITIAL_JUDGE_PROMPT_TEMPLATE.format(
            prompt_content=prompt_data["content"],
            model_response_text=model_response_data["response_text"],
            ideal_response_text=ideal_response_data["response_text"],
            correct_answer=ideal_response_data["correct_answer"],
        )
    except KeyError as e:
        print(f"Error formatting prompt: Missing key {e}. Check data files.")
        return

    # 4. Invoke LLM Judge
    print(f"Invoking Judge LLM ({JUDGE_LLM_MODEL})...")
    # Define the system prompt specifically for the judge role
    judge_system_prompt = "You are an expert mathematician and rigorous evaluator assessing an AI model's response."
    llm_response = llm_client.invoke(
        prompt=filled_prompt,
        model_name=JUDGE_LLM_MODEL,
        system_prompt=judge_system_prompt,
        temperature=0.0,
        # Add other kwargs if needed, e.g., response_format={"type": "json_object"}
    )

    if "error" in llm_response:
        print(f"LLM Invocation failed: {llm_response['error']}")
        return
    raw_llm_output_content = llm_response.get("raw_content", "")
    print("LLM invocation successful.")
    # print(f"Raw LLM Output:\n---\n{raw_llm_output_content}\n---") # Optional: print raw output

    # 5. Parse Response
    print("Parsing LLM response...")
    parsed_data = parse_judge_response(
        raw_llm_output_content
    )  # Uses default expected criteria

    if "error" in parsed_data:
        print(f"Response Parsing failed: {parsed_data['error']}")
        # Optionally save the raw output anyway for debugging
        # save_evaluation_result(...) with parsed_rubric_scores=None or error info
        return
    print("Response parsed successfully.")
    # print(f"Parsed Data:\n---\n{json.dumps(parsed_data, indent=2)}\n---") # Optional: print parsed data

    # 6. Save Result
    print("Saving evaluation result...")
    evaluation_id = f"eval_{uuid.uuid4()}"  # Generate a unique ID
    save_result = save_evaluation_result(
        evaluation_id=evaluation_id,
        response_id=TEST_RESPONSE_ID,
        ideal_response_id=TEST_IDEAL_RESPONSE_ID,
        judge_llm_model=JUDGE_LLM_MODEL,
        judge_prompt_template_version=JUDGE_PROMPT_VERSION,
        raw_judge_output={
            "raw_content": raw_llm_output_content
        },  # Store the raw response
        parsed_rubric_scores=parsed_data.get(
            "evaluation", {}
        ),  # Extract the inner evaluation dict
        # aggregated_score and final_answer_verified are None initially
    )

    if save_result.get("status") == "success":
        print(f"Evaluation saved successfully with ID: {evaluation_id}")
    else:
        print(f"Failed to save evaluation: {save_result.get('message')}")

    print("--- Single Evaluation Test Complete ---")


if __name__ == "__main__":
    # Note: This script requires the OPENAI_API_KEY environment variable to be set.
    main()
