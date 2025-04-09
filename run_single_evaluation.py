# CogniBench - Single Evaluation Test Harness
# Version: 0.1 (Phase 1)

import json
import os
import uuid
from datetime import datetime

# Import the workflow function
from core.workflow import run_evaluation_workflow
import json # For printing the result

# --- Configuration ---
# Define which specific evaluation to run

# Select which data point to test (using IDs from example files)
TEST_PROMPT_ID = "math_prompt_001"
TEST_RESPONSE_ID = "resp_001_modelA"  # Response to evaluate
TEST_IDEAL_RESPONSE_ID = "ideal_resp_001"  # Corresponding ideal response

# LLM Configuration
JUDGE_LLM_PROVIDER = "openai"  # Hardcoded for now, could be config driven later
JUDGE_LLM_MODEL = "gpt-4o"  # Or "gpt-4-turbo" or other suitable model
JUDGE_PROMPT_VERSION = "v0.2-full-L1" # Update version to reflect template change


# (Helper functions load_json_data and find_item_by_id are now within workflow.py)


# --- Main Execution Logic ---
def main():
    print(f"--- Running Evaluation Workflow for Prompt: {TEST_PROMPT_ID}, Response: {TEST_RESPONSE_ID} ---")

    # Call the workflow function
    # We pass the specific IDs and let the workflow handle loading, processing, etc.
    # We can also override LLM settings here if needed, otherwise defaults from workflow.py are used.
    result = run_evaluation_workflow(
        prompt_id=TEST_PROMPT_ID,
        response_id=TEST_RESPONSE_ID,
        ideal_response_id=TEST_IDEAL_RESPONSE_ID,
        # judge_llm_model=JUDGE_LLM_MODEL, # Optional override
        # judge_prompt_version=JUDGE_PROMPT_VERSION # Optional override
    )

    print("\n--- Test Harness: Workflow Result ---")
    print(json.dumps(result, indent=2))
    print("-------------------------------------")

    if result.get("status") != "success":
        print("Workflow execution reported an error.")
    else:
        print("Workflow executed successfully.")


if __name__ == "__main__":
    # Note: This script requires the OPENAI_API_KEY environment variable to be set.
    main()
