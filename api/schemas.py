# CogniBench - API Schemas (Pydantic Models)
# Version: 0.1 (Phase 3)

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class EvaluationRequest(BaseModel):
    """
    Defines the structure for submitting a new evaluation request via the API.
    For now, we assume the raw data is provided directly.
    Later, we might accept IDs referencing pre-stored data.
    """

    prompt_id: str = Field(..., description="A unique identifier for the prompt.")
    prompt_content: str = Field(..., description="The full text of the prompt.")
    prompt_metadata: Optional[Dict[str, Any]] = Field(
        None, description="Optional metadata about the prompt (source, topic, etc.)."
    )

    model_response_id: str = Field(
        ..., description="A unique identifier for this specific model response."
    )
    model_name: str = Field(
        ..., description="Identifier of the model that generated the response."
    )
    model_response_text: str = Field(
        ..., description="The full text of the model's response."
    )

    ideal_response_id: str = Field(
        ...,
        description="A unique identifier for the ideal response used for comparison.",
    )
    ideal_response_text: str = Field(
        ..., description="The full text of the ideal/expert response."
    )
    correct_answer: str = Field(
        ..., description="The ground-truth correct final answer."
    )

    # Optional fields to override workflow defaults
    judge_llm_model: Optional[str] = Field(
        None, description="Optional: Specify a judge LLM model to override the default."
    )
    judge_prompt_version: Optional[str] = Field(
        None, description="Optional: Specify a prompt version to override the default."
    )


class EvaluationResponse(BaseModel):
    """
    Defines the structure of the response returned after submitting an evaluation.
    """

    status: str = Field(
        ..., description="Status of the request ('submitted', 'error')."
    )
    message: Optional[str] = Field(None, description="Details or error message.")
    evaluation_id: Optional[str] = Field(
        None, description="The unique ID assigned to this evaluation run if successful."
    )
    # We might add the full result later or require fetching via GET /evaluate/{id}


class EvaluationResultData(BaseModel):
    """
    Represents the detailed data structure of a single evaluation result
    as stored in evaluations.json.
    """
    evaluation_id: str
    response_id: str
    ideal_response_id: str
    judge_llm_model: Optional[str] = None
    judge_prompt_template_version: Optional[str] = None
    raw_judge_output: Optional[Dict[str, Any]] = None
    parsed_rubric_scores: Optional[Dict[str, Any]] = None
    aggregated_score: Optional[str] = None
    final_answer_verified: Optional[bool] = None
    human_review_status: Optional[str] = None
    human_reviewer_id: Optional[str] = None
    human_review_timestamp: Optional[str] = None # Keep as string for ISO format
    human_corrected_scores: Optional[Dict[str, Any]] = None
    human_review_comments: Optional[str] = None
    created_at: str # Keep as string for ISO format
