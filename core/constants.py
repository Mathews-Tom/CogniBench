"""
Centralized constants for the CogniBench project.
"""

from pathlib import Path

# --- Core Paths ---
# Assumes this file is in CogniBench/core/
CORE_DIR = Path(__file__).parent
COGNIBENCH_ROOT = CORE_DIR.parent
APP_DIR = COGNIBENCH_ROOT / "cognibench_agent"  # Path to the Streamlit app directory
DATA_DIR = COGNIBENCH_ROOT / "data"
BASE_CONFIG_PATH = COGNIBENCH_ROOT / "config.yaml"
PROMPT_TEMPLATES_DIR_ABS = COGNIBENCH_ROOT / "prompts"
STRUCTURING_TEMPLATES_DIR = PROMPT_TEMPLATES_DIR_ABS / "structuring"
JUDGING_TEMPLATES_DIR = PROMPT_TEMPLATES_DIR_ABS / "judging"


# --- Global Color Map Constant (from cognibench_agent/constants.py) ---
COLOR_MAP = {
    "Pass": "#28a745",
    "Yes": "#28a745",
    "Not Required": "#28a745",
    "Fail": "#dc3545",
    "No": "#dc3545",
    "Needs Review": "#dc3545",
    "Partial": "#ffc107",
    "None": "#fd7e14",
    "null": "#fd7e14",
    "N/A": "#fd7e14",
}

# --- Model Definitions (from cognibench_agent/constants.py) ---
AVAILABLE_MODELS = {
    "OpenAI": {
        "O1": "o1",
        "GPT-4.1": "gpt-4.1",
        "GPT-4O": "gpt-4o",
        "GPT-4 Turbo": "gpt-4-turbo",
        "GPT-4": "gpt-4",
        "GPT-3.5 Turbo": "gpt-3.5-turbo",
    },
    "Anthropic": {
        "Claude 3.5 Haiku": "claude-3-5-haiku-latest",
        "Claude 3.5 Sonnet": "claude-3-5-sonnet-20240620",
        "Claude 3.7 Sonnet": "claude-3-7-sonnet-latest",
        "Claude 3 Opus": "claude-3-opus-20240229",
    },
    "Google": {
        "Gemini 1.5 Flash": "gemini-1.5-flash-latest",
        "Gemini 1.5 Pro": "gemini-1.5-pro-latest",
        "Gemini 2.0 Flash": "gemini-2.0-flash",
        "Gemini 2.5 Pro Preview": "gemini-2.5-pro-preview-03-25",
    },
}

# Default Structuring and Judging Models (from cognibench_agent/constants.py)
DEFAULT_STRUCTURING_MODEL = "GPT-4O"
DEFAULT_JUDGING_MODEL = "GPT-4O"
