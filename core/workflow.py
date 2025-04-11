# CogniBench - Evaluation Workflow
# Version: 0.2 (Phase 4 - External Prompts & Enhanced Postprocessing)

import functools
import json
import logging  # Import logging
import sys

# import os # May not be needed if paths come from config/args
import uuid  # Keep for evaluation_id
from pathlib import Path  # Use pathlib
from typing import Any, Dict, List, Optional

from .llm_clients.base import BaseLLMClient  # Import base class
from .llm_clients.openai_client import (
    OpenAIClient,  # Import specific client for default
)
from .output_writer import save_evaluation_result
from .postprocessing import perform_postprocessing  # Updated import

# Import core components
from .preprocessing import extract_final_answer, normalize_text_formats

# Removed: from .prompt_templates import FULL_L1_JUDGE_PROMPT_TEMPLATE
from .response_parser import parse_judge_response

# --- Configuration (Consider moving to a dedicated config module later) ---
# Default settings (can be overridden by config)
DEFAULT_JUDGE_LLM_PROVIDER = "openai"
DEFAULT_JUDGE_LLM_MODEL = "gpt-4o"
# DEFAULT_JUDGE_PROMPT_VERSION = "v0.2-full-L1" # Version comes from config now
DEFAULT_PROMPT_TEMPLATE_PATH = (
    "prompts/judge_template_v1.txt"  # Example default path in config
)

# Get logger for this module
logger = logging.getLogger(__name__)


# --- Helper Functions ---
@functools.lru_cache(maxsize=32)  # Cache loaded prompt templates by path
def load_prompt_template(template_path_str: str) -> Optional[str]:
    """Loads a prompt template from the given path."""
    template_path = Path(template_path_str)
    # Resolve relative to BASE_DIR if not absolute? Or assume path is correct.
    # Let's assume the path passed is relative to the project root (CogniBench/) or absolute.
    if not template_path.is_file():
        # Try resolving relative to project root (assuming workflow.py is in core/)
        project_root = Path(__file__).parent.parent
        template_path = project_root / template_path_str
        if not template_path.is_file():
            logger.error(
                "Prompt template file not found at %s (or resolved to %s)",
                template_path_str,
                template_path,
            )
            return None

    try:
        with template_path.open("r", encoding="utf-8") as f:
            return f.read()
    except IOError as e:
        logger.error("Error loading prompt template %s", template_path, exc_info=True)
        return None


# --- Workflow Function ---
def run_evaluation_workflow(
    prompt: str,
    response: str,
    ideal_response: str,
    config: Dict[str, Any],
    task_id: Optional[str] = None,  # For context
    model_id: Optional[str] = None,  # For context
    llm_client: Optional[
        BaseLLMClient
    ] = None,  # Allow passing a pre-initialized client
    output_jsonl_path: Optional[Path] = None,  # Path for the output JSONL file
) -> Dict[str, Any]:
    """
    Executes the end-to-end evaluation workflow for a single response.

    Args:
        prompt: The text content of the prompt.
        response: The text content of the model's response to evaluate.
        ideal_response: The text content of the ideal/reference response.
        config: The loaded configuration dictionary.
        task_id: Optional identifier for the task.
        model_id: Optional identifier for the model that generated the response.
        llm_client: An optional pre-initialized LLM client instance.
        output_jsonl_path: Optional Path object for the target output JSONL file.

    Returns:
        A dictionary containing the status and results of the workflow.
        Example success: {"status": "success", "evaluation_id": "eval_...", "result": {...}}
        Example failure: {"status": "error", "message": "Error details..."}
    """
    # Construct a unique identifier for this specific evaluation instance
    eval_instance_id = f"task_{task_id or 'unknown'}_model_{model_id or 'unknown'}"
    logger.debug(  # Changed to debug
        f"--- Starting Workflow for Evaluation Instance: {eval_instance_id} ---"
    )
    workflow_result = {"status": "error", "message": "Workflow did not complete."}

    # 1. Data is passed directly, no loading/indexing needed here.
    # Get config values
    llm_config = config.get("llm_client", {})
    eval_config = config.get("evaluation_settings", {})

    judge_llm_provider = llm_config.get("provider", DEFAULT_JUDGE_LLM_PROVIDER)
    judge_llm_model = eval_config.get(
        "judge_model", llm_config.get("model", DEFAULT_JUDGE_LLM_MODEL)
    )  # Prefer judge_model if set
    prompt_template_path = eval_config.get(
        "prompt_template", DEFAULT_PROMPT_TEMPLATE_PATH
    )

    # Validation (optional): Check if required config values are present
    if not judge_llm_model or not prompt_template_path:
        workflow_result["message"] = (
            "Missing judge_model or prompt_template in configuration."
        )
        logger.error(workflow_result["message"])
        return workflow_result

    # 2. Preprocessing
    logger.debug("Workflow Step: Preprocessing...")  # Changed to debug
    norm_model_response_text = normalize_text_formats(response)
    norm_ideal_response_text = normalize_text_formats(ideal_response)
    norm_prompt_content = normalize_text_formats(
        prompt
    )  # Normalize prompt too for consistency
    # Use ideal response as the 'correct answer' for comparison and template filling for now
    norm_correct_answer_text = norm_ideal_response_text
    extracted_answer = extract_final_answer(norm_model_response_text)
    logger.debug("  - Extracted Answer: %s", extracted_answer)  # Changed to debug

    # 3. Initialize LLM Client (if not provided)
    if llm_client is None:
        logger.debug(
            f"Workflow Step: Initializing LLM client ({judge_llm_provider})..."
        )  # Changed to debug
        # Convert provider name to lowercase for case-insensitive comparison
        provider_lower = judge_llm_provider.lower()
        if provider_lower == "openai":
            try:
                llm_client = OpenAIClient()  # Assumes API key in env/.env
            except Exception as e:
                workflow_result["message"] = f"Failed to initialize OpenAI client: {e}"
                logger.error(workflow_result["message"], exc_info=True)
                return workflow_result
        # Add elif blocks for other providers, comparing with lowercase
        # elif provider_lower == "anthropic":
        #     # Initialize Anthropic client
        #     pass
        # elif provider_lower == "google":
        #     # Initialize Google client
        #     pass
        else:
            # Handle unsupported providers
            workflow_result["message"] = (
                f"Unsupported LLM provider '{judge_llm_provider}'."  # Original case in error message
            )
            logger.error(workflow_result["message"])
            return workflow_result

    # 4. Load and Format Prompt
    logger.debug(  # Changed to debug
        f"Workflow Step: Loading prompt template from path: {prompt_template_path}..."
    )
    prompt_template = load_prompt_template(prompt_template_path)
    if prompt_template is None:
        # Error logged in load_prompt_template
        workflow_result["message"] = (
            f"Failed to load prompt template from '{prompt_template_path}'."
        )
        return workflow_result

    logger.debug("Workflow Step: Formatting prompt...")  # Changed to debug
    try:
        # Use ideal response text also as the 'correct_answer' placeholder for now
        filled_prompt = prompt_template.format(
            prompt_content=norm_prompt_content,
            model_response_text=norm_model_response_text,
            ideal_response_text=norm_ideal_response_text,
            correct_answer=norm_correct_answer_text,
        )
    except KeyError as e:
        workflow_result["message"] = (
            f"Error formatting prompt template '{prompt_template_path}': Missing key {e}."
        )
        logger.error(workflow_result["message"])
        return workflow_result
    except Exception as e:  # Catch other potential formatting errors
        workflow_result["message"] = (
            f"Unexpected error formatting prompt template '{prompt_template_path}': {e}."
        )
        logger.error(workflow_result["message"], exc_info=True)
        return workflow_result

    # 5. Invoke LLM Judge
    logger.debug(
        f"Workflow Step: Invoking Judge LLM ({judge_llm_model})..."
    )  # Changed to debug
    judge_system_prompt = "You are an expert mathematician and rigorous evaluator assessing an AI model's response."
    llm_response = llm_client.invoke(
        prompt=filled_prompt,
        model_name=judge_llm_model,
        system_prompt=judge_system_prompt,
        temperature=0.0,
    )
    if "error" in llm_response:
        workflow_result["message"] = f"LLM Invocation failed: {llm_response['error']}"
        logger.error(workflow_result["message"])
        return workflow_result
    raw_llm_output_content = llm_response.get("raw_content", "")

    # 6. Parse Response
    logger.debug("Workflow Step: Parsing LLM response...")  # Changed to debug
    parsed_data = parse_judge_response(raw_llm_output_content)
    # Parsing errors are now handled within postprocessing, but we can log here
    if "error" in parsed_data:
        logger.warning("Parsing Warning/Error: %s", parsed_data["error"])
        # Continue to postprocessing, which will flag for review
    # parsed_scores = parsed_data.get("evaluation", {}) # This is handled by postprocessing now

    # 7. Postprocessing
    # 7. Postprocessing (using the enhanced function)
    logger.debug("Workflow Step: Postprocessing...")  # Changed to debug
    postprocessing_results = perform_postprocessing(
        parsed_judge_response=parsed_data,  # Pass the full parser output
        extracted_final_answer=extracted_answer,
        correct_final_answer=norm_correct_answer_text,  # Use ideal response text here for now
    )
    logger.debug(  # Changed to debug
        "  - Final Answer Verification: %s",
        postprocessing_results.get("verification_message"),
    )
    logger.debug(  # Changed to debug
        "  - Aggregated Score: %s", postprocessing_results.get("aggregated_score")
    )
    needs_review = postprocessing_results.get("needs_human_review")
    logger.debug("  - Needs Human Review: %s", needs_review)  # Changed to debug
    if needs_review and postprocessing_results.get("review_reasons"):
        logger.debug(  # Changed to debug
            "  - Review Reasons: %s",
            "; ".join(postprocessing_results.get("review_reasons", [])),
        )

    # 8. Save Result
    logger.debug("Workflow Step: Saving evaluation result...")  # Changed to debug
    evaluation_id = f"eval_{uuid.uuid4()}"
    save_status = save_evaluation_result(
        evaluation_id=evaluation_id,  # Keep unique ID for this specific eval run
        task_id=task_id,  # Store original task ID
        model_id=model_id,  # Store model ID
        judge_llm_model=judge_llm_model,
        judge_prompt_template_path=prompt_template_path,  # Store path instead of version
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
        output_jsonl_path=output_jsonl_path,  # Pass the output path down
    )

    if save_status.get("status") == "success":
        logger.debug(  # Changed to debug
            "--- Workflow Complete for Evaluation Instance: %s ---", eval_instance_id
        )
        workflow_result = {
            "status": "success",
            "evaluation_id": evaluation_id,
            "result": save_status,  # Contains the saved data structure implicitly
        }
    else:
        workflow_result["message"] = (
            f"Failed to save evaluation: {save_status.get('message')}"
        )
        logger.error(workflow_result["message"])

    return workflow_result


# Remove the __main__ block as this module is not intended to be run directly anymore
