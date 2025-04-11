# -*- coding: utf-8 -*-
"""
CogniBench Evaluation Workflow Module.

This module orchestrates the end-to-end evaluation process for a single
prompt-response pair using a judge LLM. It handles preprocessing,
LLM invocation, response parsing, postprocessing, and result saving.

Version: 0.3 (Phase 5 - Code Quality Enhancements)
"""

import functools
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup logger for this module
logger = logging.getLogger(__name__)

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

# --- Module Constants ---
# Default settings (can be overridden by config file or arguments)
# TODO: Consider moving these defaults entirely into a config schema/loading mechanism.
DEFAULT_JUDGE_LLM_PROVIDER: str = "openai"
DEFAULT_JUDGE_LLM_MODEL: str = "gpt-4o"
DEFAULT_PROMPT_TEMPLATE_PATH: str = "prompts/full_l1_v0.2.txt"  # Updated default


# --- Helper Functions ---
@functools.lru_cache(maxsize=32)  # Cache loaded prompt templates by path
def load_prompt_template(template_path_str: str) -> Optional[str]:
    """
    Loads a prompt template from the specified file path.

    Attempts to resolve the path relative to the project root if not found directly.
    Uses LRU caching to avoid reloading the same template multiple times.

    Args:
        template_path_str: The relative or absolute path to the template file as a string.

    Returns:
        The content of the template file as a string, or None if the file
        cannot be found or read.
    """
    template_path = Path(template_path_str)
    # Assume the path passed is relative to the project root (CogniBench/) or absolute.
    if not template_path.is_file():
        # If not found directly, try resolving relative to the project root
        # (assuming this script is in CogniBench/core/)
        project_root = Path(__file__).resolve().parent.parent
        resolved_path = project_root / template_path_str
        logger.debug(
            "Template not found at %s, trying resolved path: %s",
            template_path,
            resolved_path,
        )
        if resolved_path.is_file():
            template_path = resolved_path
        else:
            logger.error(
                "Prompt template file not found at original path '%s' or resolved path '%s'",
                template_path_str,
                resolved_path,
            )
            return None

    try:
        with template_path.open("r", encoding="utf-8") as f:
            return f.read()
    except IOError as e:
        logger.error(
            "Error loading prompt template from %s", template_path, exc_info=True
        )
        return None


# --- Workflow Function ---
def run_evaluation_workflow(
    prompt: str,
    response: str,
    ideal_response: str,
    config: Dict[str, Any],
    task_id: Optional[str] = None,  # For context
    model_id: Optional[str] = None,  # For context
    llm_client: Optional[BaseLLMClient] = None,
    output_jsonl_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executes the end-to-end evaluation workflow for a single prompt-response pair.

    This function performs the following steps:
    1.  Retrieves configuration settings.
    2.  Preprocesses the input prompt, model response, and ideal response.
    3.  Initializes the appropriate LLM client if not provided.
    4.  Loads and formats the judge prompt template.
    5.  Invokes the judge LLM to get an evaluation.
    6.  Parses the judge LLM's response.
    7.  Performs postprocessing (answer verification, score aggregation, review flagging).
    8.  Saves the final evaluation result.

    Args:
        prompt: The text content of the prompt given to the model under evaluation.
        response: The text content of the model's response to the prompt.
        ideal_response: The text content of the ideal or reference response.
        config: A dictionary containing the loaded configuration settings from
            `config.yaml` or command-line arguments. Expected keys include
            `llm_client` and `evaluation_settings`.
        task_id: An optional identifier for the task or prompt being evaluated.
            Used for logging and result association. Defaults to None.
        model_id: An optional identifier for the model that generated the
            `response`. Used for logging and result association. Defaults to None.
        llm_client: An optional pre-initialized LLM client instance that conforms
            to the `BaseLLMClient` interface. If None, a new client will be
            initialized based on the `config`. Defaults to None.
        output_jsonl_path: An optional `pathlib.Path` object specifying the file
            to append the JSONL evaluation result to. If None, the default path
            from `output_writer` might be used, or saving might be skipped
            depending on `output_writer`'s implementation. Defaults to None.

    Returns:
        A dictionary indicating the outcome of the workflow.
        - On success: `{"status": "success", "evaluation_id": "eval_...", "result": {...}}`
            where `result` contains details from `save_evaluation_result`.
        - On failure: `{"status": "error", "message": "Detailed error message"}`.
    """
    # Construct a unique identifier for this specific evaluation instance
    eval_instance_id = f"task_{task_id or 'unknown'}_model_{model_id or 'unknown'}"
    logger.info(
        "--- Starting Evaluation Workflow for Instance: %s ---", eval_instance_id
    )
    workflow_result = {"status": "error", "message": "Workflow did not complete."}

    # --- 1. Configuration Extraction ---
    logger.debug("Step 1: Extracting configuration...")
    llm_config: Dict[str, Any] = config.get("llm_client", {})
    eval_config: Dict[str, Any] = config.get("evaluation_settings", {})

    # Determine Judge LLM provider, model, and prompt template path from config, using defaults if necessary
    judge_llm_provider: str = llm_config.get("provider", DEFAULT_JUDGE_LLM_PROVIDER)
    # Prefer 'judge_model' in eval_config, fallback to general 'model' in llm_config, then default
    judge_llm_model: str = eval_config.get(
        "judge_model", llm_config.get("model", DEFAULT_JUDGE_LLM_MODEL)
    )
    prompt_template_path: str = eval_config.get(
        "prompt_template", DEFAULT_PROMPT_TEMPLATE_PATH
    )
    # Get evaluation rubric details from config
    expected_criteria: Optional[List[str]] = eval_config.get("expected_criteria")
    allowed_scores: Optional[List[str]] = eval_config.get("allowed_scores")

    logger.debug("  - Judge Provider: %s", judge_llm_provider)
    logger.debug("  - Judge Model: %s", judge_llm_model)
    logger.debug("  - Prompt Template Path: %s", prompt_template_path)
    logger.debug("  - Expected Criteria: %s", expected_criteria)
    logger.debug("  - Allowed Scores: %s", allowed_scores)

    # Basic validation of essential config values
    # TODO: Implement more robust config validation, possibly using Pydantic.
    if not judge_llm_model:
        msg = "Configuration error: 'judge_model' (or fallback 'llm_client.model') is not specified."
        logger.error(msg)
        workflow_result["message"] = msg
        return workflow_result
    if not prompt_template_path:
        msg = "Configuration error: 'evaluation_settings.prompt_template' is not specified."
        logger.error(msg)
        workflow_result["message"] = msg
        return workflow_result
    # Validate that expected_criteria and allowed_scores are present and are lists
    if not expected_criteria or not isinstance(expected_criteria, list):
        msg = "Configuration error: 'evaluation_settings.expected_criteria' is missing or not a list."
        logger.error(msg)
        workflow_result["message"] = msg
        return workflow_result
    if not allowed_scores or not isinstance(allowed_scores, list):
        msg = "Configuration error: 'evaluation_settings.allowed_scores' is missing or not a list."
        logger.error(msg)
        workflow_result["message"] = msg
        return workflow_result

    # --- 2. Preprocessing ---
    logger.debug("Step 2: Preprocessing inputs...")
    norm_model_response_text = normalize_text_formats(response)
    norm_ideal_response_text = normalize_text_formats(ideal_response)
    norm_prompt_content = normalize_text_formats(
        prompt
    )  # Normalize prompt too for consistency
    # Use the normalized ideal response as the 'correct answer' for template filling and verification for now.
    # TODO: Revisit if 'correct_answer' needs separate handling or configuration.
    norm_correct_answer_text = norm_ideal_response_text
    # Attempt to extract the final answer part from the model's response
    extracted_answer = extract_final_answer(norm_model_response_text)
    logger.debug("  - Normalized Prompt Length: %d", len(norm_prompt_content or ""))
    logger.debug(
        "  - Normalized Response Length: %d", len(norm_model_response_text or "")
    )
    logger.debug(
        "  - Normalized Ideal Response Length: %d", len(norm_ideal_response_text or "")
    )
    logger.debug("  - Extracted Final Answer: %s", extracted_answer)

    # --- 3. Initialize LLM Client ---
    active_llm_client: BaseLLMClient
    if llm_client is None:
        logger.debug(
            "Step 3: Initializing LLM client (provider: %s)...", judge_llm_provider
        )
        # Convert provider name to lowercase for case-insensitive comparison
        provider_lower = judge_llm_provider.lower()
        if provider_lower == "openai":
            try:
                # Assumes API key is handled by the client (e.g., via env vars)
                active_llm_client = OpenAIClient()
            except ImportError as e:
                msg = f"Failed to initialize OpenAI client: OpenAI library not installed? ({e})"
                logger.error(msg, exc_info=True)
                workflow_result["message"] = msg
                return workflow_result
            except Exception as e:  # Catch potential API key errors or other issues
                msg = f"Failed to initialize OpenAI client: {e}"
                logger.error(msg, exc_info=True)
                workflow_result["message"] = msg
                return workflow_result
        elif provider_lower == "anthropic":
            try:
                # Example: Dynamically import Anthropic client if needed
                from .llm_clients.anthropic_client import (
                    AnthropicClient,  # Placeholder import
                )

                # TODO: Implement AnthropicClient in llm_clients/anthropic_client.py
                # TODO: Pass necessary config (e.g., API key from llm_config)
                active_llm_client = AnthropicClient()  # Placeholder initialization
                logger.info("Initialized Anthropic LLM client.")
            except ImportError:
                msg = f"Failed to initialize Anthropic client: Client implementation or 'anthropic' library likely missing."
                logger.error(msg, exc_info=True)
                workflow_result["message"] = msg
                return workflow_result
            except Exception as e:
                msg = f"Failed to initialize Anthropic client: {e}"
                logger.error(msg, exc_info=True)
                workflow_result["message"] = msg
                return workflow_result
        elif provider_lower == "google":
            try:
                # Example: Dynamically import Google client if needed
                from .llm_clients.google_client import (
                    GoogleClient,  # Placeholder import
                )

                # TODO: Implement GoogleClient in llm_clients/google_client.py
                # TODO: Pass necessary config (e.g., API key/credentials from llm_config)
                active_llm_client = GoogleClient()  # Placeholder initialization
                logger.info("Initialized Google LLM client.")
            except ImportError:
                msg = f"Failed to initialize Google client: Client implementation or Google library likely missing."
                logger.error(msg, exc_info=True)
                workflow_result["message"] = msg
                return workflow_result
            except Exception as e:
                msg = f"Failed to initialize Google client: {e}"
                logger.error(msg, exc_info=True)
                workflow_result["message"] = msg
                return workflow_result
        # Add more elif blocks here for other supported providers (e.g., 'huggingface', 'local_llm')
        else:
            # Handle unsupported provider specified in the config
            msg = f"Unsupported LLM provider specified in configuration: '{judge_llm_provider}'"
            logger.error(msg)
            workflow_result["message"] = msg
            return workflow_result
    else:
        # Use the pre-initialized client passed as an argument
        logger.debug("Step 3: Using pre-initialized LLM client.")
        active_llm_client = llm_client

    # --- 4. Load and Format Judge Prompt ---
    logger.debug("Step 4a: Loading prompt template from '%s'...", prompt_template_path)
    prompt_template = load_prompt_template(prompt_template_path)
    if prompt_template is None:
        # Error logged in load_prompt_template
        # Error already logged by load_prompt_template
        msg = f"Failed to load prompt template from '{prompt_template_path}'."
        workflow_result["message"] = msg
        # No need to log again here, already logged in helper function
        return workflow_result

    logger.debug("Step 4b: Formatting judge prompt...")
    try:
        # Fill the template with the preprocessed content.
        # Ensure all required placeholders are present in the template file.
        # Using ideal response text also as the 'correct_answer' placeholder for now.
        filled_prompt = prompt_template.format(
            prompt_content=norm_prompt_content or "",  # Use empty string if None
            model_response_text=norm_model_response_text or "",
            ideal_response_text=norm_ideal_response_text or "",
            correct_answer=norm_correct_answer_text or "",
        )
        logger.debug("  - Formatted prompt length: %d", len(filled_prompt))
    except KeyError as e:
        msg = (
            f"Error formatting prompt template '{prompt_template_path}': "
            f"Missing placeholder key {e}. Check the template file."
        )
        logger.error(msg)
        workflow_result["message"] = msg
        return workflow_result
    except Exception as e:  # Catch other potential formatting errors
        msg = (
            f"Unexpected error formatting prompt template '{prompt_template_path}': {e}"
        )
        logger.error(msg, exc_info=True)
        workflow_result["message"] = msg
        return workflow_result

    # --- 5. Invoke LLM Judge ---
    logger.debug("Step 5: Invoking Judge LLM ('%s')...", judge_llm_model)
    judge_system_prompt = "You are an expert mathematician and rigorous evaluator assessing an AI model's response."
    # TODO: Make system prompt configurable?
    llm_response: Dict[str, Any] = active_llm_client.invoke(
        prompt=filled_prompt,
        model_name=judge_llm_model,
        system_prompt=judge_system_prompt,
        temperature=0.0,  # Use low temperature for deterministic judging
        # TODO: Add other relevant parameters like max_tokens if needed/configurable
    )
    if "error" in llm_response:
        msg = f"Judge LLM invocation failed: {llm_response['error']}"
        logger.error(msg)
        workflow_result["message"] = msg
        return workflow_result
    # Assuming the client returns a dict with 'raw_content' on success
    raw_llm_output_content: str = llm_response.get("raw_content", "")
    logger.debug("  - Raw Judge LLM output length: %d", len(raw_llm_output_content))

    # --- 6. Parse Judge Response ---
    logger.debug("Step 6: Parsing judge LLM response...")
    # Pass the loaded criteria and scores to the parser
    parsed_data: Dict[str, Any] = parse_judge_response(
        raw_response_content=raw_llm_output_content,
        expected_criteria=expected_criteria,  # Now passed from config
        allowed_scores=allowed_scores,  # Now passed from config
    )

    # Log parsing errors here, but let postprocessing handle the review flagging logic
    if "error" in parsed_data:
        logger.warning("Judge response parsing failed: %s", parsed_data["error"])
        # Continue to postprocessing, which will detect the error and flag for review.

    # --- 7. Postprocessing ---
    logger.debug("Step 7: Performing postprocessing...")
    postprocessing_results = perform_postprocessing(
        parsed_judge_response=parsed_data,  # Pass the full parser output
        extracted_final_answer=extracted_answer,
        # Use the normalized ideal response text as the ground truth for now
        correct_final_answer=norm_correct_answer_text,
    )

    # Log key postprocessing outcomes
    logger.debug(
        "  - Final Answer Verification: %s (Message: %s)",
        postprocessing_results.get("final_answer_verified"),
        postprocessing_results.get("verification_message"),
    )
    logger.debug(
        "  - Aggregated Score: %s", postprocessing_results.get("aggregated_score")
    )
    needs_review: bool = postprocessing_results.get("needs_human_review", False)
    logger.debug("  - Needs Human Review: %s", needs_review)
    if needs_review:
        review_reasons: List[str] = postprocessing_results.get("review_reasons", [])
        logger.debug("  - Review Reasons: %s", "; ".join(review_reasons))

    # --- 8. Save Result ---
    logger.debug("Step 8: Saving evaluation result...")
    evaluation_id = f"eval_{uuid.uuid4()}"
    # Prepare the data structure to be saved
    # Note: parsed_data might contain an 'error' key if parsing failed.
    # postprocessing_results contains aggregated scores, verification, etc.
    save_status: Dict[str, Any] = save_evaluation_result(
        evaluation_id=evaluation_id,
        task_id=task_id,
        model_id=model_id,
        judge_llm_model=judge_llm_model,
        judge_prompt_template_path=prompt_template_path,
        # Include raw judge output for traceability
        raw_judge_output={"raw_content": raw_llm_output_content},
        # Include parsed scores if available (might be empty dict or contain error)
        parsed_rubric_scores=parsed_data.get("evaluation", {}),
        parsing_error=parsed_data.get("error"),  # Pass parsing error if present
        # Pass all fields from the postprocessing results
        final_answer_verified=postprocessing_results.get("final_answer_verified"),
        verification_message=postprocessing_results.get("verification_message"),
        aggregated_score=postprocessing_results.get("aggregated_score"),
        needs_human_review=needs_review,
        review_reasons=postprocessing_results.get("review_reasons"),
        # Specify the output file path
        output_jsonl_path=output_jsonl_path,
    )

    if save_status.get("status") == "success":
        logger.info(
            "--- Evaluation Workflow COMPLETED Successfully for Instance: %s (Eval ID: %s) ---",
            eval_instance_id,
            evaluation_id,
        )
        workflow_result = {
            "status": "success",
            "evaluation_id": evaluation_id,
            "result": save_status,  # Contains the saved data structure implicitly
        }
    else:
        msg = f"Failed to save evaluation result: {save_status.get('message', 'Unknown error')}"
        workflow_result["message"] = msg
        logger.error(msg)
        # Note: The workflow technically completed up to the saving step,
        # but the final result wasn't persisted. Status remains 'error'.

    return workflow_result


# Note: Removed __main__ block as this module is intended for import, not direct execution.
