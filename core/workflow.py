# -*- coding: utf-8 -*-
"""
CogniBench Evaluation Workflow Module.

Version: 0.4 (Phase 6 - Structured Input and Logging Enhancements)
"""

import functools
import json
import logging
import time  # Added import
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .llm_clients.base import BaseLLMClient
from .llm_clients.openai_client import OpenAIClient
from .output_writer import save_evaluation_result
from .postprocessing import perform_postprocessing
from .preprocessing import (
    convert_math_notation,
    extract_final_answer,
    normalize_text_formats,
)
from .prompt_templates import load_prompt_template
from .response_parser import parse_judge_response

logger = logging.getLogger("backend")

DEFAULT_JUDGE_LLM_PROVIDER: str = "openai"
DEFAULT_JUDGE_LLM_MODEL: str = "gpt-4o"
DEFAULT_PROMPT_TEMPLATE_PATH: str = "prompts/judging/Math-L1-judge-v1.0.txt"
DEFAULT_STRUCTURING_TEMPLATE_PATH: str = (
    "prompts/structuring/Math-L1-structuring-v1.0.txt"
)
MAX_RETRIES: int = 3


# --- Helper Function for Parsing Embedded JSON (Copied from run_batch_evaluation.py) ---
def _parse_json_string(
    json_string: Optional[str], task_id: str, field_name: str
) -> Union[Dict[str, Any], List[Any], str, None]:
    """Attempts to parse a JSON string, potentially embedded in markdown code fences.

    Args:
        json_string: The string value to parse.
        task_id: The task ID for logging context.
        field_name: The name of the field being parsed for logging context.

    Returns:
        The parsed Python object (dict or list) if successful,
        otherwise the original string. Returns None if input is None.
    """
    if json_string is None:
        return None
    if not isinstance(json_string, str):
        logger.warning(
            "Task [%s]: Field '%s' was not a string, skipping parsing. Value: %s",
            task_id,
            field_name,
            json_string,
        )
        return json_string  # Return the original non-string value

    # Attempt to remove markdown fences (```json ... ``` or ``` ... ```)
    extracted_string = json_string.strip()
    if extracted_string.startswith("```json"):
        extracted_string = extracted_string[len("```json") :]
    elif extracted_string.startswith("```"):
        extracted_string = extracted_string[len("```") :]

    if extracted_string.endswith("```"):
        extracted_string = extracted_string[: -len("```")]

    extracted_string = extracted_string.strip()

    try:
        # Attempt to parse the extracted string
        parsed_data = json.loads(extracted_string)
        return parsed_data
    except json.JSONDecodeError as e:
        logger.warning(
            "Task [%s]: Failed to parse JSON for field '%s'. Error: %s. Keeping original string: %s",
            task_id,
            field_name,
            e,
            json_string,
            exc_info=False,
        )
        return json_string  # Return the original string on failure
    except Exception as e:  # Catch unexpected errors during parsing
        logger.error(
            "Task [%s]: Unexpected error parsing JSON for field '%s'. Error: %s. Keeping original string: %s",
            task_id,
            field_name,
            e,
            json_string,
            exc_info=True,
        )
        return json_string  # Return the original string on failure


def run_evaluation_workflow(
    prompt: str,
    response: Any,
    ideal_response: Any,
    correct_answer: str,
    config: Dict[str, Any],
    task_id: Optional[str] = None,
    model_id: Optional[str] = None,
    llm_client: Optional[BaseLLMClient] = None,
    structured: bool = False,
    output_jsonl_path: Optional[Path] = None,
    structured_ideal_cache: Optional[Dict[str, str]] = None,  # Add cache parameter
) -> Dict[str, Any]:
    start_time = time.time()  # Record start time
    structuring_api_calls = 0
    judging_api_calls = 0
    eval_instance_id = f"task_{task_id or 'unknown'}_model_{model_id or 'unknown'}"
    logger.info(
        "--- Starting Evaluation Workflow for Instance: %s ---", eval_instance_id
    )

    try:
        llm_config: Dict[str, Any] = config.get("llm_client", {})
        eval_config: Dict[str, Any] = config.get("evaluation_settings", {})

        judge_llm_provider: str = llm_config.get("provider", DEFAULT_JUDGE_LLM_PROVIDER)
        judge_llm_model: str = eval_config.get(
            "judge_model", llm_config.get("model", DEFAULT_JUDGE_LLM_MODEL)
        )
        prompt_template_path: str = eval_config.get(
            "prompt_template", DEFAULT_PROMPT_TEMPLATE_PATH
        )
        expected_criteria: Optional[List[str]] = eval_config.get("expected_criteria")
        allowed_scores: Optional[List[str]] = eval_config.get("allowed_scores")

        norm_prompt_content = normalize_text_formats(prompt)

        if llm_client is None:
            active_llm_client = OpenAIClient()
        else:
            active_llm_client = llm_client
        norm_model_response_text = normalize_text_formats(response)
        norm_ideal_response_text = normalize_text_formats(ideal_response)

        # Structuring step (mandatory before judging)
        structuring_template_path: str = config.get("structuring_settings", {}).get(
            "prompt_template", DEFAULT_STRUCTURING_TEMPLATE_PATH
        )
        structuring_prompt_template = load_prompt_template(structuring_template_path)
        if structuring_prompt_template is None:
            msg = f"Failed to load structuring prompt template from '{structuring_template_path}'."
            logger.error(msg)
            return {"status": "error", "message": msg}

        structuring_system_prompt = (
            "You are an expert structurer preparing responses for rigorous evaluation."
        )
        structuring_model_name = config.get("structuring_settings", {}).get(
            "structuring_model", judge_llm_model
        )

        # --- Structure Model Response ---
        prompt_for_model_structuring = (
            f"{structuring_prompt_template}\n\n"
            f"--- Unstructured Solution to Convert ---\n"
            f"{norm_model_response_text}"
        )
        structured_model_response_obj = None  # Initialize object
        logger.info(
            "STRUCTURING_CALL (Model Response): task_id=%s, model_id=%s, structuring_model=%s",
            task_id,
            model_id,
            structuring_model_name,
        )
        for attempt in range(MAX_RETRIES):
            structuring_api_calls += 1  # Increment counter
            response_model = active_llm_client.invoke(
                prompt=prompt_for_model_structuring,
                model_name=structuring_model_name,
                system_prompt=structuring_system_prompt,
                temperature=0.0,
            )
            if "error" not in response_model:
                raw_structured_model_response = response_model.get("raw_content", "")
                # Parse the raw response
                parsed_model_response = _parse_json_string(
                    raw_structured_model_response,
                    task_id or "unknown",
                    "structured_model_response (workflow)",
                )
                # Always wrap the parsed result to ensure the correct structuring model name is included
                structured_model_response_obj = {
                    "model": structuring_model_name, # Use name from config
                    "response": parsed_model_response, # The actual parsed content (dict or string)
                }
                break # Exit loop on successful structuring
            # Log warning if attempt failed
            logger.warning(
                "Structuring (Model Response) attempt %d failed: %s",
                attempt + 1,
                response_model["error"],
            )
        else:
            msg = f"Structuring LLM invocation failed for Model Response after {MAX_RETRIES} attempts."
            logger.error(msg)
            # Decide if we should return error or try to proceed without structured model response
            # For now, let's return an error to be safe.
            return {"status": "error", "message": msg}

        # --- Structure Ideal Response (with Caching) ---
        structured_ideal_response = None
        cache_hit = False
        # structured_ideal_response will now hold the dictionary object, not just the string
        if (
            structured_ideal_cache is not None
            and task_id is not None
            and task_id in structured_ideal_cache
        ):
            # Retrieve the dictionary object from cache
            structured_ideal_response = structured_ideal_cache[task_id]
            # Basic validation that it's likely the expected dict format
            if (
                isinstance(structured_ideal_response, dict)
                and "model" in structured_ideal_response
                and "response" in structured_ideal_response
            ):
                cache_hit = True
                logger.info(
                    "CACHE_HIT (Ideal Response): Using cached structured ideal response object for task_id=%s",
                    task_id,
                )
            else:
                logger.warning(
                    "CACHE_INVALID: Found item in cache for task_id %s, but it was not the expected dictionary format. Will re-structure.",
                    task_id,
                )
                # Invalidate cache entry if format is wrong
                del structured_ideal_cache[task_id]
                structured_ideal_response = None  # Reset to ensure re-structuring

        if not cache_hit:
            logger.info(
                "STRUCTURING_CALL (Ideal Response): task_id=%s, model_id=%s, structuring_model=%s",
                task_id,
                model_id,  # model_id is less relevant here, but kept for consistency
                structuring_model_name,
            )
            prompt_for_ideal_structuring = (
                f"{structuring_prompt_template}\n\n"
                f"--- Unstructured Solution to Convert ---\n"
                f"{norm_ideal_response_text}"
            )
            # Correct indentation: for loop starts one level inside 'if not cache_hit:'
            for attempt in range(MAX_RETRIES):
                structuring_api_calls += 1  # Increment counter
                response_ideal = active_llm_client.invoke(
                    prompt=prompt_for_ideal_structuring,
                    model_name=structuring_model_name,
                    system_prompt=structuring_system_prompt,
                    temperature=0.0,
                )
                if "error" not in response_ideal:
                    raw_structured_ideal_response = response_ideal.get("raw_content", "")
                    # Parse the raw response
                    parsed_ideal_response = _parse_json_string(
                        raw_structured_ideal_response,
                        task_id or "unknown",
                        "structured_ideal_response (workflow)",
                    )
                    # Always wrap the parsed result to include the structuring model name
                    structured_ideal_response = {
                        "model": structuring_model_name, # Use name from config
                        "response": parsed_ideal_response, # The actual parsed content (dict or string)
                    }

                    # Store the final dictionary object in cache if successful and cache is provided
                    if structured_ideal_cache is not None and task_id is not None:
                        structured_ideal_cache[task_id] = structured_ideal_response # Store the final dict object
                        logger.info(
                            "CACHE_STORE (Ideal Response): Stored structured ideal response object for task_id=%s",
                            task_id,
                        )
                    break  # Break is inside the inner if
                # This warning is part of the for loop
                logger.warning(
                    "Structuring (Ideal Response) attempt %d failed: %s",
                    attempt + 1,
                    response_ideal["error"],
                )
            # This else corresponds to the for loop (if it completes without break)
            else:
                msg = f"Structuring LLM invocation failed for Ideal Response after {MAX_RETRIES} attempts."
                logger.error(msg)
                # Decide if we should return error or try to proceed without structured ideal response
                # For now, let's return an error to be safe.
                return {"status": "error", "message": msg}
        # Indent elif and all subsequent code one level to be inside the function
        elif cache_hit and structured_ideal_response is None:
            # This case should ideally not happen if cache stores valid strings, but handle defensively
            msg = f"Cache hit for task_id {task_id}, but cached value was None. Cannot proceed."
            logger.error(msg)
            return {"status": "error", "message": msg}

        # Ensure we have strings, potentially JSON strings, for the next step
        # The judging prompt expects these as string inputs.
        # norm_model_response_text = json.dumps(structured_model_response or "") # No longer needed here
        # norm_ideal_response_text = json.dumps(structured_ideal_response or "") # No longer needed here

        norm_prompt_content = normalize_text_formats(prompt)
        norm_correct_answer_text = normalize_text_formats(correct_answer)
        extracted_answer = extract_final_answer(norm_model_response_text)

        prompt_template = load_prompt_template(prompt_template_path)
        if prompt_template is None:
            msg = f"Failed to load prompt template from '{prompt_template_path}'."
            logger.error(msg)
            return {"status": "error", "message": msg}

        # --- Prepare structured content strings for the judge prompt ---
        # Extract the 'response' part from the structured object
        model_resp_content = (
            structured_model_response_obj.get("response", "")
            if structured_model_response_obj
            else ""
        )
        # If the 'response' part is a dict/list (parsed JSON), convert it back to a JSON string for the prompt
        if isinstance(model_resp_content, (dict, list)):
            model_resp_str_for_prompt = json.dumps(model_resp_content, ensure_ascii=False)
        else:
            model_resp_str_for_prompt = str(
                model_resp_content
            )  # Use the string directly if parsing failed


        # Extract the 'response' part from the structured object for the ideal response
        # Handle cases where structured_ideal_response might be the raw string if parsing failed
        if isinstance(structured_ideal_response, dict):
            ideal_resp_content = structured_ideal_response.get("response", "")
        else:
            ideal_resp_content = structured_ideal_response or ""  # Use the string directly

        # If the extracted content is still a dict/list, convert it back to a JSON string for the prompt
        if isinstance(ideal_resp_content, (dict, list)):
            ideal_resp_str_for_prompt = json.dumps(ideal_resp_content, ensure_ascii=False)
        else:
            ideal_resp_str_for_prompt = str(ideal_resp_content)
        
        # --- Log values before formatting judge prompt ---
        logger.debug(f"Task [{task_id}] Model [{model_id}]: Preparing judge prompt. Template path: {prompt_template_path}")
        # logger.debug(f"Task [{task_id}] Model [{model_id}]: Template content: {prompt_template}") # Optional: Log template if needed, can be long
        logger.debug(f"Task [{task_id}] Model [{model_id}]: norm_prompt_content type: {type(norm_prompt_content)}")
        logger.debug(f"Task [{task_id}] Model [{model_id}]: model_resp_str_for_prompt type: {type(model_resp_str_for_prompt)}, value snippet: {model_resp_str_for_prompt[:200]}...") # Log snippet
        logger.debug(f"Task [{task_id}] Model [{model_id}]: ideal_resp_str_for_prompt type: {type(ideal_resp_str_for_prompt)}, value snippet: {ideal_resp_str_for_prompt[:200]}...") # Log snippet
        logger.debug(f"Task [{task_id}] Model [{model_id}]: norm_correct_answer_text type: {type(norm_correct_answer_text)}")

        # --- Fill the judge prompt template ---
        filled_prompt = prompt_template.format(
            prompt=norm_prompt_content,
                    structured_model_response=model_resp_str_for_prompt,
                    structured_ideal_response=ideal_resp_str_for_prompt,
                    correct_answer=norm_correct_answer_text,
                )

        judge_system_prompt = "You are an expert mathematician and rigorous evaluator assessing an AI model's response."
        llm_response = None
        # Log judging call
        logger.info(
            "JUDGING_CALL: task_id=%s, model_id=%s, judge_model=%s",
            task_id,
            model_id,
            judge_llm_model,
        )
        for attempt in range(MAX_RETRIES):
            judging_api_calls += 1  # Increment counter
            llm_response = active_llm_client.invoke(
                prompt=filled_prompt,
                model_name=judge_llm_model,
                system_prompt=judge_system_prompt,
                temperature=0.0,
            )
            if "error" not in llm_response:
                break
            logger.warning(
                "Judging attempt %d failed: %s", attempt + 1, llm_response["error"]
            )
        else:
            msg = f"Judging LLM invocation failed after {MAX_RETRIES} attempts."
            logger.error(msg)
            return {"status": "error", "message": msg}

        if "error" in llm_response:
            msg = f"Judge LLM invocation failed: {llm_response['error']}"
            logger.error(msg)
            return {"status": "error", "message": msg}

        raw_llm_output_content: str = llm_response.get("raw_content", "")
        parsed_data: Dict[str, Any] = parse_judge_response(
            raw_response_content=raw_llm_output_content,
            expected_criteria=expected_criteria,
            allowed_scores=allowed_scores,
        )

        postprocessing_results = perform_postprocessing(
            parsed_judge_response=parsed_data,
            extracted_final_answer=extracted_answer,
            correct_final_answer=norm_correct_answer_text,
            config=config,
        )

        total_time = time.time() - start_time  # Calculate total time
        evaluation_id = f"eval_{uuid.uuid4()}"
        save_status: Dict[str, Any] = save_evaluation_result(
            evaluation_id=evaluation_id,
            task_id=task_id,
            model_id=model_id,
            judge_llm_model=judge_llm_model,
            judge_prompt_template_path=prompt_template_path,
            raw_judge_output={"raw_content": raw_llm_output_content},
            parsed_rubric_scores=parsed_data.get("evaluation", {}),
            parsing_error=parsed_data.get("error"),
            final_answer_verified=postprocessing_results.get("final_answer_verified"),
            verification_message=postprocessing_results.get("verification_message"),
            aggregated_score=postprocessing_results.get("aggregated_score"),
            needs_human_review=postprocessing_results.get("needs_human_review", False),
            review_reasons=postprocessing_results.get("review_reasons", []),
            # Pass the structured object, not just the string
            structured_model_response=structured_model_response_obj,
            structured_ideal_response=structured_ideal_response,  # Pass the structured dict object
            output_jsonl_path=output_jsonl_path,
            # Pass new metrics
            structuring_api_calls=structuring_api_calls,
            judging_api_calls=judging_api_calls,  # Indent this line
            total_time_seconds=total_time,  # Indent this line
        )  # Indent this line

        # Initialize workflow_result to a default error state
        # Use the evaluation_id generated earlier for tracking
        workflow_result = {
             "status": "error",
             "message": "Workflow failed before checking save status",
             "evaluation_id": evaluation_id
        }

        # Check save status and prepare final result (Correctly indented within function)
        # Add explicit check for save_status being None
        if save_status is not None and save_status.get("status") == "success":
            logger.info(
                "--- Evaluation Workflow COMPLETED Successfully for Instance: %s (Eval ID: %s) ---",
                eval_instance_id,
                evaluation_id,
            )
            # Overwrite workflow_result on success
            workflow_result = {
                "status": "success",
                "evaluation_id": evaluation_id,
                "result": save_status,
            }
        elif save_status is None:
             # Handle case where save_evaluation_result might have returned None (shouldn't happen ideally)
             msg = "Failed to get save status (save_evaluation_result returned None)"
             workflow_result = {"status": "error", "message": msg, "evaluation_id": evaluation_id}
             logger.error(msg)
        else: # Handle case where save_status indicates failure
            msg = f"Failed to save evaluation result: {save_status.get('message', 'Unknown error')}"
            # Overwrite workflow_result on save failure
            workflow_result = {"status": "error", "message": msg, "evaluation_id": evaluation_id}
            logger.error(msg)

        # Final return for the function (still inside try block)
        return workflow_result
    except Exception as e: # Aligned with the 'try' at line 124
        # Catch any unexpected exceptions during the workflow
        logger.exception(
            "Unexpected error occurred during evaluation workflow for instance %s: %s",
            eval_instance_id,
            e,
        )
        # Return a standardized error dictionary
        return {
            "status": "error",
            "message": f"Unexpected workflow error: {e}",
            "evaluation_id": f"eval_error_{uuid.uuid4()}", # Generate an ID for tracking
        }
