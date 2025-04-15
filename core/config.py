# -*- coding: utf-8 -*-
"""
Centralized configuration management for CogniBench.

This module defines Pydantic models for validating the application's
configuration and provides a function to load and validate the configuration
from a YAML file, handling environment variable substitution.
"""

import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import (
    BaseModel,
    Field,
    FilePath,
    ValidationError,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


# --- Helper Functions ---
def _substitute_env_vars(value: Any, field_name: str) -> Any:
    """Recursively substitutes environment variables like ${VAR_NAME}."""
    if isinstance(value, str):
        # Regex to find ${VAR_NAME} patterns
        pattern = r"\$\{(.+?)\}"

        def replace_match(match):
            var_name = match.group(1)
            env_var = os.getenv(var_name)
            if env_var is None:
                logger.warning(
                    "Environment variable '%s' referenced in config field '%s' is not set.",
                    var_name,
                    field_name,
                )
                # Return the original placeholder or an empty string, depending on desired behavior
                return match.group(0)  # Keep placeholder if not found
                # return "" # Or return empty string
            return env_var

        # Substitute all occurrences in the string
        substituted_value, num_subs = re.subn(pattern, replace_match, value)
        # If the entire string was just the variable, return the substituted value directly
        # (handles cases where the env var might be numeric, bool, etc.)
        if num_subs == 1 and substituted_value == os.getenv(
            re.match(pattern, value).group(1)
        ):
            # Try to infer type if it looks like bool/int/float
            if substituted_value.lower() == "true":
                return True
            if substituted_value.lower() == "false":
                return False
            try:
                return int(substituted_value)
            except ValueError:
                pass
            try:
                return float(substituted_value)
            except ValueError:
                pass
            return substituted_value  # Return as string otherwise
        return substituted_value  # Return potentially modified string

    elif isinstance(value, dict):
        return {
            k: _substitute_env_vars(v, f"{field_name}.{k}") for k, v in value.items()
        }
    elif isinstance(value, list):
        return [
            _substitute_env_vars(item, f"{field_name}[{i}]")
            for i, item in enumerate(value)
        ]
    return value


# --- Pydantic Models ---
class LLMClientConfig(BaseModel):
    """Configuration for the LLM client."""

    provider: str
    model: str
    api_key: Optional[str] = Field(
        None, description="API Key for the provider. Can use ${ENV_VAR}."
    )

    # Automatically substitute env vars for api_key before validation
    @field_validator("api_key", mode="before")
    @classmethod
    def substitute_api_key(cls, v):
        return _substitute_env_vars(v, "llm_client.api_key")


class InputOptions(BaseModel):
    """Configuration for input data sources."""

    # data_file: Optional[str] = Field( # Keep data_file optional for backward compatibility?
    #     None, description="Default path to a single processed/ingested data file."
    # )
    file_paths: Optional[List[str]] = Field(
        None, description="List of paths to input JSON files for batch processing."
    )

    @model_validator(mode='after')
    def check_at_least_one_input(self) -> 'InputOptions':
        # if not self.data_file and not self.file_paths:
        #     raise ValueError("Either 'data_file' or 'file_paths' must be provided in input_options")
        # Allow empty file_paths if provided dynamically by the app
        return self

class StructuringSettings(BaseModel):
    """Configuration for the structuring phase."""

    structuring_model: str
    prompt_template: str = Field(
        description="Path to the structuring prompt template file."
    )

    @field_validator("prompt_template")
    @classmethod
    def check_template_path(cls, v: str, info: Any) -> str:
        """Validate that the template path exists relative to project root."""
        # Assume paths in config are relative to project root (parent of core)
        project_root = Path(__file__).resolve().parent.parent
        full_path = project_root / v
        if not full_path.is_file():
            raise ValueError(f"Structuring prompt template not found at: {full_path}")
        # Return the original relative path as stored in config
        return v


class EvaluationSettings(BaseModel):
    """Configuration for the evaluation (judging) phase."""

    judge_model: str
    prompt_template: str = Field(
        description="Path to the judging prompt template file."
    )
    expected_criteria: List[str]
    allowed_scores: List[str]

    @field_validator("prompt_template")
    @classmethod
    def check_template_path(cls, v: str, info: Any) -> str:
        """Validate that the template path exists relative to project root."""
        project_root = Path(__file__).resolve().parent.parent
        full_path = project_root / v
        if not full_path.is_file():
            raise ValueError(f"Judging prompt template not found at: {full_path}")
        # Return the original relative path as stored in config
        return v


class HumanReviewFlagRules(BaseModel):
    """Rules for flagging evaluations for human review."""

    flag_on_partial_score: bool = True
    flag_on_trivial_justification: bool = True
    flag_on_verification_failure: bool = True


class HumanReviewSettings(BaseModel):
    """Configuration for human review flagging."""

    flag_rules: HumanReviewFlagRules


class AggregationRules(BaseModel):
    """Rules for aggregating rubric scores."""

    fail_if_any_no: bool = True
    partial_if_any_partial: bool = True
    pass_if_all_yes: bool = True


class ConsistencyChecks(BaseModel):
    """Configuration for consistency checks during aggregation."""

    enable_trivial_justification_check: bool = True
    trivial_justification_length_threshold: int = 10


class AggregationSettings(BaseModel):
    """Configuration for aggregation and consistency checks."""

    aggregation_rules: AggregationRules
    consistency_checks: ConsistencyChecks


class OutputOptions(BaseModel):
    """Configuration for output files."""

    output_dir: str = Field(
        "data", description="Default directory for saving all output files (JSONL, JSON, etc.)."
    )
    results_file_stem: str = Field(
        "evaluation_results", description="Base name for the final evaluation results JSON file (e.g., 'Batch-001_final_results'). The specific name will be constructed."
    )
    # Note: The output directory will be created if it doesn't exist.

class AppConfig(BaseModel):
    """Root model for the application configuration."""

    llm_client: LLMClientConfig
    input_options: InputOptions
    structuring_settings: StructuringSettings
    evaluation_settings: EvaluationSettings
    human_review_settings: HumanReviewSettings
    aggregation_settings: AggregationSettings
    output_options: OutputOptions

    # Apply environment variable substitution to the whole dict after initial loading
    # This handles variables in nested structures beyond just api_key
    @model_validator(mode="before")
    @classmethod
    def substitute_all_env_vars(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return _substitute_env_vars(data, "root")
        return data

    # Add root validator for cross-field checks if needed later
    # Example: Ensure judge_model and structuring_model use the same provider if required
    # @model_validator(mode='after')
    # def check_model_providers(self) -> 'AppConfig':
    #     # Example logic (adapt as needed)
    #     # if self.llm_client.provider != self.evaluation_settings.judge_model_provider:
    #     #     raise ValueError("Judge model must use the same provider as the main LLM client")
    #     return self


# --- Loading Function ---
@lru_cache(maxsize=1)  # Cache the loaded config for efficiency
def load_config(config_path: Union[str, Path, None] = None) -> AppConfig:
    """
    Loads, validates, and returns the application configuration.

    Args:
        config_path: Optional path to the YAML configuration file.
                     Defaults to 'config.yaml' in the project root.

    Returns:
        An AppConfig object containing the validated configuration.

    Raises:
        FileNotFoundError: If the configuration file cannot be found.
        yaml.YAMLError: If the configuration file is invalid YAML.
        ValidationError: If the configuration fails Pydantic validation.
        Exception: For other unexpected errors during loading.
    """
    resolved_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    logger.info("Attempting to load configuration from: %s", resolved_path)

    if not resolved_path.is_file():
        logger.error("Configuration file not found at %s", resolved_path)
        raise FileNotFoundError(f"Configuration file not found at {resolved_path}")

    try:
        with resolved_path.open("r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        if not isinstance(raw_config, dict):
            raise TypeError(
                f"Configuration file '{resolved_path}' did not load as a dictionary."
            )

        # Validate the raw dictionary using Pydantic
        # Environment variable substitution happens within the model validator
        validated_config = AppConfig(**raw_config)
        logger.info("Configuration loaded and validated successfully.")
        return validated_config

    except yaml.YAMLError as e:
        logger.error("Error parsing YAML configuration file %s: %s", resolved_path, e)
        raise
    except ValidationError as e:
        logger.error("Configuration validation failed for %s:", resolved_path)
        # Log details of validation errors
        for error in e.errors():
            logger.error(
                "  Field: %s, Error: %s",
                " -> ".join(map(str, error["loc"])),
                error["msg"],
            )
        raise
    except Exception as e:
        logger.error(
            "An unexpected error occurred while loading configuration from %s",
            resolved_path,
            exc_info=True,
        )
        raise


# --- Example Usage (for testing) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing configuration loading...")
    try:
        # Example: Set an environment variable for testing substitution
        # os.environ["OPENAI_API_KEY"] = "sk-testkeyfromenv"
        # os.environ["MY_RESULTS_FILE"] = "data/results_from_env.json" # Example for another field

        config = load_config()
        print("\n--- Loaded Configuration ---")
        print(config.model_dump_json(indent=2))

        # Access specific values
        print(f"\nLLM Provider: {config.llm_client.provider}")
        print(f"Judge Model: {config.evaluation_settings.judge_model}")
        print(f"API Key (potentially from env): {bool(config.llm_client.api_key)}")
        print(f"Results File: {config.output_options.results_file}")

        # Test loading with a specific path (if you have another test config)
        # test_config_path = Path("./test_config.yaml")
        # if test_config_path.exists():
        #     print("\n--- Loading Test Configuration ---")
        #     test_config = load_config(test_config_path)
        #     print(test_config.model_dump_json(indent=2))

    except (FileNotFoundError, yaml.YAMLError, ValidationError, Exception) as e:
        print("\n--- Configuration Loading Failed ---")
        print(f"Error: {e}")

    finally:
        # Clean up test environment variables if set
        # if "OPENAI_API_KEY" in os.environ: del os.environ["OPENAI_API_KEY"]
        # if "MY_RESULTS_FILE" in os.environ: del os.environ["MY_RESULTS_FILE"]
        pass
