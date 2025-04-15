"""
CogniBench Structuring Script (Standalone - Potentially Outdated).

Loads ingested data, applies a structuring prompt using an LLM
to both model and ideal responses, and saves the structured results.

NOTE: This script might be outdated based on recent refactoring,
especially regarding LLM client usage (`get_completion` vs `invoke`).
It's kept for reference but may need updates to work with the current core logic.
"""

import json
import logging
import sys  # Import sys
from pathlib import Path  # Import Path
from typing import Any, Dict, List, Optional  # Import typing

# Add project root to sys.path to allow absolute import of core modules
APP_DIR = Path(__file__).resolve().parent
COGNIBENCH_ROOT = APP_DIR.parent
if str(COGNIBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(COGNIBENCH_ROOT))

# Use relative imports consistent with other scripts
try:
    from core.llm_clients.openai_client import OpenAIClient
    from core.prompt_templates import load_prompt_template

    # Assuming log setup is handled elsewhere or basic config is sufficient
    # from core.log_setup import setup_logging
except ImportError as e:
    print(f"Error importing core modules: {e}", file=sys.stderr)
    print(
        "Ensure the script is run from the project root or the PYTHONPATH is set correctly.",
        file=sys.stderr,
    )
    sys.exit(1)

# setup_logging() should be called by the entry point (e.g., run_batch_evaluation)
logger = logging.getLogger("backend")

# Define paths using pathlib relative to project root
STRUCTURING_PROMPT_PATH = (
    COGNIBENCH_ROOT / "prompts/structuring/Math-L1-structuring-v1.0.txt"
)
STRUCTURED_OUTPUT_DIR = COGNIBENCH_ROOT / "data/structured_responses"


def load_json(filepath: Path) -> Optional[List[Dict[str, Any]]]:
    """Loads JSON data (expected as a list of dicts) from a file."""
    try:
        with filepath.open("r", encoding="utf-8") as file:
            data = json.load(file)
            if isinstance(data, list):
                return data
            else:
                logger.error(f"Data in {filepath} is not a JSON list.")
                return None
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return None
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading/parsing {filepath}: {e}")
        return None


def save_json(data: List[Dict[str, Any]], filepath: Path) -> None:
    """Saves data to a JSON file."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
    except IOError as e:
        logger.error(f"Error writing output file to {filepath}: {e}")


def structure_response(
    client: OpenAIClient, prompt_template: str, response_text: Optional[str]
) -> Optional[Dict[str, Any]]:
    """
    Uses the LLM client to apply the structuring prompt to the response text.

    NOTE: Uses `client.get_completion` which may be outdated. Should likely use `client.invoke`.
    """
    if not response_text:
        return None  # Cannot structure empty text

    prompt = f"{prompt_template}\n\n--- Unstructured Solution to Convert ---\n{response_text}"
    # TODO: Replace get_completion with invoke and handle the dictionary response format
    # structured_response_dict = client.invoke(prompt=prompt, model_name="gpt-4o", temperature=0.0) # Example using invoke
    # raw_content = structured_response_dict.get("raw_content")
    # if raw_content:
    #     try:
    #         return json.loads(raw_content) # Attempt to parse the raw content
    #     except json.JSONDecodeError:
    #         logger.error(f"Failed to parse structured response JSON: {raw_content[:100]}...")
    #         return None # Return None if parsing fails
    # else:
    #     logger.error(f"LLM invocation failed or returned no content: {structured_response_dict.get('error')}")
    #     return None

    # --- Keeping potentially outdated logic for now ---
    try:
        # This method likely doesn't exist or work as expected anymore
        structured_response_str = client.get_completion(prompt)
        return json.loads(structured_response_str)
    except AttributeError:
        logger.error(
            "`get_completion` method not found on client. Script likely needs update to use `invoke`."
        )
        raise  # Re-raise to stop execution clearly
    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to parse LLM response as JSON: {e}. Response: {structured_response_str[:100]}..."
        )
        return None  # Return None if parsing fails
    # --- End outdated logic ---


def main(input_json_path: str) -> None:
    """Loads data, structures responses using LLM, and saves results."""
    data = load_json(input_json_path)
    # Initialize client (assuming API key is in env)
    try:
        client = OpenAIClient()
    except ValueError as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        sys.exit(1)
    try:
        prompt_template = load_prompt_template(str(STRUCTURING_PROMPT_PATH))
    except FileNotFoundError as e:
        logger.error(f"Failed to load structuring prompt template: {e}")
        sys.exit(1)

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

    # Use pathlib for path joining
    input_filename = Path(input_json_path).name
    output_path = STRUCTURED_OUTPUT_DIR / input_filename
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
    # main(args.input_json_path) # Remove duplicate call
