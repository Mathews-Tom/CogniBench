import json
import logging
import os

from CogniBench.core.llm_clients.openai_client import OpenAIClient
from CogniBench.core.prompt_templates import load_prompt_template

# setup_logging() should be called by the entry point (e.g., run_batch_evaluation)
logger = logging.getLogger('backend')

STRUCTURING_PROMPT_PATH = "prompts/structuring/Math-L1-structuring-v1.0.txt"
STRUCTURED_OUTPUT_DIR = "data/structured_responses"


def load_json(filepath):
    with open(filepath, "r") as file:
        return json.load(file)


def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as file:
        json.dump(data, file, indent=2)


def structure_response(client, prompt_template, response_text):
    prompt = f"{prompt_template}\n\n{response_text}"
    structured_response = client.get_completion(prompt)
    return json.loads(structured_response)


def main(input_json_path):
    data = load_json(input_json_path)
    client = OpenAIClient(model="gpt-4o")
    prompt_template = load_prompt_template(STRUCTURING_PROMPT_PATH)

    structured_data = []
    for entry in data:
        task_id = entry.get("task_id")
        model_response = entry.get("model_response")
        ideal_response = entry.get("ideal_response")

        try:
            structured_model_response = structure_response(
                client, prompt_template, model_response
            )
            structured_ideal_response = structure_response(
                client, prompt_template, ideal_response
            )

            structured_entry = {
                "task_id": task_id,
                "structured_model_response": structured_model_response,
                "structured_ideal_response": structured_ideal_response,
            }
            structured_data.append(structured_entry)
            logger.info(f"Structured responses for task_id {task_id} successfully.")
        except Exception as e:
            logger.error(f"Error structuring responses for task_id {task_id}: {e}")

    output_path = os.path.join(STRUCTURED_OUTPUT_DIR, os.path.basename(input_json_path))
    save_json(structured_data, output_path)
    logger.info(f"Structured data saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run structuring on ingested JSON data."
    )
    parser.add_argument(
        "input_json_path", help="Path to the input JSON file containing responses."
    )
    args = parser.parse_args()
    main(args.input_json_path)
    main(args.input_json_path)
