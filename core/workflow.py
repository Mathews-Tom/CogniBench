# -*- coding: utf-8 -*-
"""
CogniBench Evaluation Workflow Module.

Version: 0.4 (Phase 6 - Structured Input and Logging Enhancements)
"""

import asyncio  # Added for async
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Use RELATIVE imports for modules within the 'core' package
from .batch_processor import format_request_for_batch
from .llm_clients.base import BaseLLMClient
from .llm_clients.openai_client import OpenAIClient
from .output_writer import save_evaluation_result
from .postprocessing import perform_postprocessing
from .preprocessing import normalize_text_formats
from .prompt_templates import load_prompt_template
from .response_parser import parse_judge_response

from scripts.validate_responses import validate_structuring_response, validate_judging_response

logger = logging.getLogger("backend")

DEFAULT_JUDGE_LLM_PROVIDER: str = "openai"
DEFAULT_JUDGE_LLM_MODEL: str = "gpt-4o"
DEFAULT_PROMPT_TEMPLATE_PATH: str = "prompts/judging/Math-L1-judge-v1.0.txt"
DEFAULT_STRUCTURING_TEMPLATE_PATH: str = (
    "prompts/structuring/Math-L1-structuring-v1.0.txt"
)
MAX_RETRIES: int = 3


# --- Helper Function for Parsing Embedded JSON ---
def _parse_json_string(
    json_string: Optional[str], task_id: str, field_name: str
) -> Union[Dict[str, Any], List[Any], str, None]:
    """
    Attempts to parse a JSON string, potentially embedded in markdown code fences.
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
        return json_string

    extracted_string = json_string.strip()
    if extracted_string.startswith("```json"):
        extracted_string = extracted_string[len("```json") :]
    elif extracted_string.startswith("```"):
        extracted_string = extracted_string[len("```") :]
    if extracted_string.endswith("```"):
        extracted_string = extracted_string[: -len("```")]
    extracted_string = extracted_string.strip()

    try:
        return json.loads(extracted_string)
    except json.JSONDecodeError as e:
        logger.warning(
            "Task [%s]: Failed to parse JSON for field '%s'. Error: %s. Keeping original string: %s",
            task_id,
            field_name,
            e,
            json_string,
            exc_info=False,
        )
        return json_string
    except Exception as e:
        logger.error(
            "Task [%s]: Unexpected error parsing JSON for field '%s'. Error: %s. Keeping original string: %s",
            task_id,
            field_name,
            e,
            json_string,
            exc_info=True,
        )
        return json_string


# --- Refactored Post-Judging Logic ---


def process_judging_output(
    raw_judge_output_content: str,
    expected_criteria: Optional[List[str]],
    allowed_scores: Optional[List[str]],
    structured_model_response_obj: Optional[Dict],
    correct_final_answer: Optional[str],
    config: Dict,
) -> Dict[str, Any]:
    """
    Parses judge response and performs post-processing.
    Returns a dictionary containing parsed_data and postprocessing_results.
    """
    logger.debug("Processing judging output...")
    parsed_data = parse_judge_response(
        raw_response_content=raw_judge_output_content,
        expected_criteria=expected_criteria,
        allowed_scores=allowed_scores,
    )
    postprocessing_results = perform_postprocessing(
        parsed_judge_response=parsed_data,
        structured_model_response_obj=structured_model_response_obj,
        correct_final_answer=correct_final_answer,
        config=config,
    )
    logger.debug(f"Post-processing results: {postprocessing_results}")
    return {
        "parsed_data": parsed_data,
        "postprocessing_results": postprocessing_results,
    }


def finalize_and_save_evaluation(
    evaluation_id: str,
    task_id: Optional[str],
    model_id: Optional[str],
    judge_llm_model: str,
    judge_prompt_template_path: str,
    raw_judge_output_content: str,
    processed_judge_data: Dict[str, Any],
    structured_model_response_obj: Optional[Dict],
    structured_ideal_response_obj: Optional[Dict],
    output_jsonl_path: Optional[Path],
    structuring_api_calls: int = 0,
    judging_api_calls: int = 0,
    total_time_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Calls save_evaluation_result with all necessary data."""
    logger.debug(f"Finalizing and saving evaluation {evaluation_id}...")
    parsed_data = processed_judge_data.get("parsed_data", {})
    postprocessing_results = processed_judge_data.get("postprocessing_results", {})
    save_status = save_evaluation_result(
        evaluation_id=evaluation_id,
        task_id=task_id,
        model_id=model_id,
        judge_llm_model=judge_llm_model,
        judge_prompt_template_path=judge_prompt_template_path,
        raw_judge_output={"raw_content": raw_judge_output_content},
        parsed_rubric_scores=parsed_data.get("evaluation", {}),
        parsing_error=parsed_data.get("error"),
        final_answer_verified=postprocessing_results.get("final_answer_verified"),
        verification_message=postprocessing_results.get("verification_message"),
        aggregated_score=postprocessing_results.get("aggregated_score"),
        needs_human_review=postprocessing_results.get("needs_human_review", False),
        review_reasons=postprocessing_results.get("review_reasons", []),
        structured_model_response=structured_model_response_obj,
        structured_ideal_response=structured_ideal_response_obj,
        output_jsonl_path=output_jsonl_path,
        structuring_api_calls=structuring_api_calls,
        judging_api_calls=judging_api_calls,
        total_time_seconds=total_time_seconds,
    )
    return save_status


# --- Main Workflow Function ---


async def run_evaluation_workflow(  # Changed to async def
    prompt: str,
    response: Any,
    ideal_response: Any,
    correct_answer: Optional[str],
    config: Dict[str, Any],
    task_id: Optional[str] = None,
    model_id: Optional[str] = None,
    llm_client: Optional[BaseLLMClient] = None,
    structured: bool = False,
    output_jsonl_path: Optional[Path] = None,
    structured_ideal_cache: Optional[Dict[str, Any]] = None,
    aggregate_structuring: bool = False,
) -> Dict[str, Any]:
    start_time = time.time()
    structuring_api_calls = 0
    judging_api_calls = 0
    eval_instance_id = f"task_{task_id or 'unknown'}_model_{model_id or 'unknown'}"
    logger.info(
        "--- Starting Evaluation Workflow for Instance: %s ---", eval_instance_id
    )

    structured_model_response_obj = None
    structured_ideal_response_obj = None
    model_structuring_request = None
    ideal_structuring_request = None
    evaluation_id = f"eval_error_{uuid.uuid4()}"

    try:
        llm_config = config.get("llm_client", {})
        eval_config = config.get("evaluation_settings", {})
        structuring_config = config.get("structuring_settings", {})
        judge_llm_model: str = eval_config.get(
            "judge_model", llm_config.get("model", DEFAULT_JUDGE_LLM_MODEL)
        )
        prompt_template_path: str = eval_config.get(
            "prompt_template", DEFAULT_PROMPT_TEMPLATE_PATH
        )
        expected_criteria: Optional[List[str]] = eval_config.get("expected_criteria")
        allowed_scores: Optional[List[str]] = eval_config.get("allowed_scores")
        structuring_template_path: str = structuring_config.get(
            "prompt_template", DEFAULT_STRUCTURING_TEMPLATE_PATH
        )
        structuring_model_name: str = structuring_config.get(
            "structuring_model", judge_llm_model
        )

        active_llm_client = llm_client if llm_client else OpenAIClient()

        norm_prompt_content = normalize_text_formats(prompt)
        norm_model_response_text = normalize_text_formats(response)
        norm_ideal_response_text = normalize_text_formats(ideal_response)
        norm_correct_answer_text = (
            normalize_text_formats(correct_answer) if correct_answer else None
        )

        structuring_prompt_template = load_prompt_template(structuring_template_path)
        if structuring_prompt_template is None:
            msg = f"Failed to load structuring prompt template from '{structuring_template_path}'."
            logger.error(msg)
            return {"status": "error", "message": msg, "evaluation_id": evaluation_id}

        structuring_system_prompt = (
            "You are an expert structurer preparing responses for rigorous evaluation."
        )

        # --- Structure Model Response ---
        prompt_for_model_structuring = f"{structuring_prompt_template}\n\n--- Unstructured Solution to Convert ---\n{norm_model_response_text}"
        if aggregate_structuring:
            model_structuring_payload = {
                "model": structuring_model_name,
                "messages": [
                    {"role": "system", "content": structuring_system_prompt},
                    {"role": "user", "content": prompt_for_model_structuring},
                ],
                "temperature": 0.0,
            }
            model_custom_id = (
                f"structure_model_{task_id or 'unknown'}_{model_id or 'unknown'}"
            )
            model_structuring_request = format_request_for_batch(
                custom_id=model_custom_id,
                method="POST",
                url="/v1/chat/completions",
                body=model_structuring_payload,
            )
            logger.debug(
                f"Aggregated structuring request for model response: {model_custom_id}"
            )
        else:
            logger.info(
                "STRUCTURING_CALL (Model Response): task_id=%s, model_id=%s, structuring_model=%s",
                task_id,
                model_id,
                structuring_model_name,
            )
            for attempt in range(MAX_RETRIES):
                structuring_api_calls += 1
                response_model = await active_llm_client.invoke(  # Added await
                    prompt=prompt_for_model_structuring,
                    model_name=structuring_model_name,
                    system_prompt=structuring_system_prompt,
                    temperature=0.0,
                )
                if "error" not in response_model:
                    raw_structured_model_response = response_model.get(
                        "raw_content", ""
                    )
                    parsed_model_response = _parse_json_string(
                        raw_structured_model_response,
                        task_id or "unknown",
                        "structured_model_response (workflow)",
                    )
                    structured_model_response_obj = {
                        "model": structuring_model_name,
                        "response": parsed_model_response,
                    }
                    # --- Validation: Structured Model Response ---
                    validation_model = validate_structuring_response(
                        parsed_model_response, norm_prompt_content
                    )
                    model_validation_failed = (
                        any(v is False for k, v in validation_model.items() if k not in ["missing_fields", "red_flags"])
                        or validation_model.get("missing_fields")
                        or validation_model.get("red_flags")
                    )
                    if model_validation_failed:
                        logger.warning(
                            "Validation failed for structured model response: %s", validation_model
                        )
                    break
                logger.warning(
                    "Structuring (Model Response) attempt %d failed: %s",
                    attempt + 1,
                    response_model["error"],
                )
                await asyncio.sleep(1)  # Added small delay for retries
            else:
                msg = f"Structuring LLM invocation failed for Model Response after {MAX_RETRIES} attempts."
                logger.error(msg)
                return {
                    "status": "error",
                    "message": msg,
                    "evaluation_id": evaluation_id,
                }

        # --- Structure Ideal Response (with Caching) ---
        cache_hit = False
        if (
            structured_ideal_cache is not None
            and task_id is not None
            and task_id in structured_ideal_cache
        ):
            cached_obj = structured_ideal_cache[task_id]
            if (
                isinstance(cached_obj, dict)
                and "model" in cached_obj
                and "response" in cached_obj
            ):
                structured_ideal_response_obj = cached_obj
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
                del structured_ideal_cache[task_id]

        if not cache_hit:
            prompt_for_ideal_structuring = f"{structuring_prompt_template}\n\n--- Unstructured Solution to Convert ---\n{norm_ideal_response_text}"
            if aggregate_structuring:
                ideal_structuring_payload = {
                    "model": structuring_model_name,
                    "messages": [
                        {"role": "system", "content": structuring_system_prompt},
                        {"role": "user", "content": prompt_for_ideal_structuring},
                    ],
                    "temperature": 0.0,
                }
                ideal_custom_id = f"structure_ideal_{task_id or 'unknown'}"
                ideal_structuring_request = format_request_for_batch(
                    custom_id=ideal_custom_id,
                    method="POST",
                    url="/v1/chat/completions",
                    body=ideal_structuring_payload,
                )
                logger.debug(
                    f"Aggregated structuring request for ideal response: {ideal_custom_id}"
                )
            else:
                logger.info(
                    "STRUCTURING_CALL (Ideal Response): task_id=%s, model_id=%s, structuring_model=%s",
                    task_id,
                    model_id,
                    structuring_model_name,
                )
                for attempt in range(MAX_RETRIES):
                    structuring_api_calls += 1
                    response_ideal = await active_llm_client.invoke(  # Added await
                        prompt=prompt_for_ideal_structuring,
                        model_name=structuring_model_name,
                        system_prompt=structuring_system_prompt,
                        temperature=0.0,
                    )
                    if "error" not in response_ideal:
                        raw_structured_ideal_response = response_ideal.get(
                            "raw_content", ""
                        )
                        parsed_ideal_response = _parse_json_string(
                            raw_structured_ideal_response,
                            task_id or "unknown",
                            "structured_ideal_response (workflow)",
                        )
                        structured_ideal_response_obj = {
                            "model": structuring_model_name,
                            "response": parsed_ideal_response,
                        }
                        # --- Validation: Structured Ideal Response ---
                        validation_ideal = validate_structuring_response(
                            parsed_ideal_response, norm_prompt_content
                        )
                        ideal_validation_failed = (
                            any(v is False for k, v in validation_ideal.items() if k not in ["missing_fields", "red_flags"])
                            or validation_ideal.get("missing_fields")
                            or validation_ideal.get("red_flags")
                        )
                        if ideal_validation_failed:
                            logger.warning(
                                "Validation failed for structured ideal response: %s", validation_ideal
                            )
                        if structured_ideal_cache is not None and task_id is not None:
                            structured_ideal_cache[task_id] = (
                                structured_ideal_response_obj
                            )
                            logger.info(
                                "CACHE_STORE (Ideal Response): Stored structured ideal response object for task_id=%s",
                                task_id,
                            )
                        break
                    logger.warning(
                        "Structuring (Ideal Response) attempt %d failed: %s",
                        attempt + 1,
                        response_ideal["error"],
                    )
                    await asyncio.sleep(1)  # Added small delay for retries
                else:
                    msg = f"Structuring LLM invocation failed for Ideal Response after {MAX_RETRIES} attempts."
                    logger.error(msg)
                    return {
                        "status": "error",
                        "message": msg,
                        "evaluation_id": evaluation_id,
                    }

        # --- Return aggregated requests if in aggregation mode ---
        if aggregate_structuring:
            logger.info(
                f"Instance {eval_instance_id}: Returning aggregated structuring requests."
            )
            return {
                "status": "aggregated",
                "model_request": model_structuring_request,
                "ideal_request": ideal_structuring_request,
            }

        # --- Proceed with Judging (only if not aggregating) ---
        if (
            structured_model_response_obj is None
            or structured_ideal_response_obj is None
        ):
            msg = f"Cannot proceed to judging for {eval_instance_id}: Missing structured model or ideal response object."
            logger.error(msg)
            return {"status": "error", "message": msg, "evaluation_id": evaluation_id}

        prompt_template = load_prompt_template(prompt_template_path)
        if prompt_template is None:
            msg = f"Failed to load prompt template from '{prompt_template_path}'."
            logger.error(msg)
            return {"status": "error", "message": msg, "evaluation_id": evaluation_id}

        model_resp_content = structured_model_response_obj.get("response", "")
        model_resp_str_for_prompt = (
            json.dumps(model_resp_content, ensure_ascii=False)
            if isinstance(model_resp_content, (dict, list))
            else str(model_resp_content)
        )
        ideal_resp_content = structured_ideal_response_obj.get("response", "")
        ideal_resp_str_for_prompt = (
            json.dumps(ideal_resp_content, ensure_ascii=False)
            if isinstance(ideal_resp_content, (dict, list))
            else str(ideal_resp_content)
        )

        filled_prompt = prompt_template.format(
            prompt=norm_prompt_content,
            structured_model_response=model_resp_str_for_prompt,
            structured_ideal_response=ideal_resp_str_for_prompt,
            correct_answer=norm_correct_answer_text or "",
        )

        judge_system_prompt = "You are an expert mathematician and rigorous evaluator assessing an AI model's response."
        llm_response = None
        logger.info(
            "JUDGING_CALL: task_id=%s, model_id=%s, judging_model=%s",
            task_id,
            model_id,
            judge_llm_model,
        )
        for attempt in range(MAX_RETRIES):
            judging_api_calls += 1
            llm_response = await active_llm_client.invoke(  # Added await
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
            await asyncio.sleep(1)  # Added small delay for retries
        else:
            msg = f"Judging LLM invocation failed after {MAX_RETRIES} attempts."
            logger.error(msg)
            return {"status": "error", "message": msg, "evaluation_id": evaluation_id}

        if llm_response is None or "error" in llm_response:
            msg = f"Judge LLM invocation failed: {llm_response.get('error', 'Unknown error') if llm_response else 'No response'}"
            logger.error(msg)
            return {"status": "error", "message": msg, "evaluation_id": evaluation_id}

        raw_llm_output_content: str = llm_response.get("raw_content", "")
        judging_processed_data = process_judging_output(
            raw_judge_output_content=raw_llm_output_content,
            expected_criteria=expected_criteria,
            allowed_scores=allowed_scores,
            structured_model_response_obj=structured_model_response_obj,
            correct_final_answer=norm_correct_answer_text,
            config=config,
        )

        total_time = time.time() - start_time
        evaluation_id = f"eval_{uuid.uuid4()}"

        # --- Validation: Judging Model Response ---
        validation_judging = validate_judging_response(
            judging_processed_data.get("parsed_data", {}), norm_prompt_content
        )
        judging_validation_failed = (
            any(v is False for k, v in validation_judging.items() if k not in ["missing_fields", "red_flags"])
            or validation_judging.get("missing_fields")
            or validation_judging.get("red_flags")
        )
        if judging_validation_failed:
            logger.warning(
                "Validation failed for judging model response: %s", validation_judging
            )

        # Aggregate validation results
        validation_failed = {}
        if 'model_validation_failed' in locals() and model_validation_failed:
            validation_failed["structured_model_response"] = validation_model
        if 'ideal_validation_failed' in locals() and ideal_validation_failed:
            validation_failed["structured_ideal_response"] = validation_ideal
        if judging_validation_failed:
            validation_failed["judging_response"] = validation_judging

        save_status = finalize_and_save_evaluation(
            evaluation_id=evaluation_id,
            task_id=task_id,
            model_id=model_id,
            judge_llm_model=judge_llm_model,
            judge_prompt_template_path=prompt_template_path,
            raw_judge_output_content=raw_llm_output_content,
            processed_judge_data=judging_processed_data,
            structured_model_response_obj=structured_model_response_obj,
            structured_ideal_response_obj=structured_ideal_response_obj,
            output_jsonl_path=output_jsonl_path,
            structuring_api_calls=structuring_api_calls,
            judging_api_calls=judging_api_calls,
            total_time_seconds=total_time,
        )

        if save_status is not None and save_status.get("status") == "success":
            logger.info(
                "--- Evaluation Workflow COMPLETED Successfully for Instance: %s (Eval ID: %s) ---",
                eval_instance_id,
                evaluation_id,
            )
            result_dict = {
                "status": "success",
                "evaluation_id": evaluation_id,
                "result": save_status,
            }
            if validation_failed:
                result_dict["validation_failed"] = validation_failed
            return result_dict
        else:
            msg = f"Failed to save evaluation result: {save_status.get('message', 'Unknown error') if save_status else 'Save function returned None'}"
            logger.error(msg)
            return {"status": "error", "message": msg, "evaluation_id": evaluation_id}

    except Exception as e:
        logger.exception(
            "Unexpected error occurred during evaluation workflow for instance %s: %s",
            eval_instance_id,
            e,
        )
        return {
            "status": "error",
            "message": f"Unexpected workflow error: {type(e).__name__}: {e}",
            "evaluation_id": evaluation_id,
        }
