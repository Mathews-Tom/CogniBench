# -*- coding: utf-8 -*-
"""
CogniBench Evaluation Workflow Module.

Version: 0.4 (Phase 6 - Structured Input and Logging Enhancements)
"""

import functools
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

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

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_LLM_PROVIDER: str = "openai"
DEFAULT_JUDGE_LLM_MODEL: str = "gpt-4o"
DEFAULT_PROMPT_TEMPLATE_PATH: str = "prompts/judging/Math-L1-judge-v1.0.txt"
DEFAULT_STRUCTURING_TEMPLATE_PATH: str = (
    "prompts/structuring/Math-L1-structuring-v1.0.txt"
)
MAX_RETRIES: int = 3


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
) -> Dict[str, Any]:
    eval_instance_id = f"task_{task_id or 'unknown'}_model_{model_id or 'unknown'}"
    logger.info(
        "--- Starting Evaluation Workflow for Instance: %s ---", eval_instance_id
    )

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

    structuring_filled_prompt = structuring_prompt_template.format(
        prompt=norm_prompt_content,
        model_response=norm_model_response_text,
        ideal_response=norm_ideal_response_text,
    )

    structuring_system_prompt = (
        "You are an expert structurer preparing responses for rigorous evaluation."
    )
    structuring_response = None
    # Get structuring model name for logging
    structuring_model_name = config.get("structuring_settings", {}).get(
        "structuring_model", judge_llm_model
    )
    # Log structuring call
    logger.info(
        "STRUCTURING_CALL: task_id=%s, model_id=%s, structuring_model=%s",
        task_id,
        model_id,
        structuring_model_name,
    )
    for attempt in range(MAX_RETRIES):
        structuring_response = active_llm_client.invoke(
            prompt=structuring_filled_prompt,
            model_name=config.get("structuring_settings", {}).get(
                "structuring_model", judge_llm_model
            ),
            system_prompt=structuring_system_prompt,
            temperature=0.0,
        )
        if "error" not in structuring_response:
            break
        logger.warning(
            "Structuring attempt %d failed: %s",
            attempt + 1,
            structuring_response["error"],
        )
    else:
        msg = f"Structuring LLM invocation failed after {MAX_RETRIES} attempts."
        logger.error(msg)
        return {"status": "error", "message": msg}

    structured_model_response = structuring_response.get("raw_content", "")
    structured_ideal_response = (
        norm_ideal_response_text  # Assuming ideal response doesn't need structuring
    )

    norm_model_response_text = json.dumps(structured_model_response)
    norm_ideal_response_text = json.dumps(structured_ideal_response)

    norm_prompt_content = normalize_text_formats(prompt)
    norm_correct_answer_text = normalize_text_formats(correct_answer)
    extracted_answer = extract_final_answer(norm_model_response_text)


    prompt_template = load_prompt_template(prompt_template_path)
    if prompt_template is None:
        msg = f"Failed to load prompt template from '{prompt_template_path}'."
        logger.error(msg)
        return {"status": "error", "message": msg}

    filled_prompt = prompt_template.format(
        prompt=norm_prompt_content,
        structured_model_response=norm_model_response_text,
        structured_ideal_response=norm_ideal_response_text,
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
        llm_response = active_llm_client.invoke(
            prompt=filled_prompt,
            model_name=judge_llm_model,
            system_prompt=judge_system_prompt,
            temperature=0.0,
        )
        if "error" not in llm_response:
            break
        logger.warning("Judging attempt %d failed: %s", attempt + 1, llm_response["error"])
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
        structured_model_response=structured_model_response,
        structured_ideal_response=structured_ideal_response,
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
            "result": save_status,
        }
    else:
        msg = f"Failed to save evaluation result: {save_status.get('message', 'Unknown error')}"
        workflow_result = {"status": "error", "message": msg}
        logger.error(msg)

    return workflow_result
