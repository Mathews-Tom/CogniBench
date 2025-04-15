"""
Constants used by the CogniBench Streamlit application.
"""

# --- Global Color Map Constant ---
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

# --- Model Definitions ---
AVAILABLE_MODELS = {
    "OpenAI": {
        "GPT-4O": "gpt-4o",
        "GPT-4 Turbo": "gpt-4-turbo",
        "GPT-4": "gpt-4",
        "GPT-3.5 Turbo": "gpt-3.5-turbo",
        "O1": "o1",
        "GPT-4.1": "gpt-4.1",
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