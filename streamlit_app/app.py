import json
import logging  # Added import
import os
import queue
import re
import subprocess
import sys  # Keep sys import
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

# --- Add project root to sys.path ---
APP_DIR = Path(__file__).parent
COGNIBENCH_ROOT = APP_DIR.parent
if str(COGNIBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(COGNIBENCH_ROOT))
# --- End sys.path modification ---
from typing import Any, Dict, Optional  # Added imports

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

# Import the setup function using absolute path (now possible due to sys.path modification)
from core.log_setup import setup_logging  # Reverted to absolute import

# Setup logging for the Streamlit app
logger = logging.getLogger('streamlit')
if "logging_setup_complete" not in st.session_state:
    setup_logging()  # Call the setup function
    st.session_state.logging_setup_complete = True  # Mark as done
    # Get the specific logger for streamlit
    logger = logging.getLogger('streamlit')
    logger.info("Initial logging setup complete.")  # Log only once
    logger.info("Streamlit app started.")  # Move this here
# --- Constants ---
APP_DIR = Path(__file__).parent
COGNIBENCH_ROOT = APP_DIR.parent
BASE_CONFIG_PATH = COGNIBENCH_ROOT / "config.yaml"
RUN_BATCH_SCRIPT_PATH = COGNIBENCH_ROOT / "scripts" / "run_batch_evaluation.py"
PROMPT_TEMPLATES_DIR_ABS = COGNIBENCH_ROOT / "prompts"

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

st.set_page_config(layout="wide", page_title="CogniBench Runner")

# Initialize temporary directory for session state
if "temp_dir_path" not in st.session_state:
    logger.info("Initializing temporary directory for session state.")
    st.session_state.temp_dir = tempfile.TemporaryDirectory()
    st.session_state.temp_dir_path = Path(st.session_state.temp_dir.name)


st.title("CogniBench Evaluation Runner")
# logger.info("Streamlit app started.") # Moved to initial setup block

# --- Phase 1: Input Selection ---
st.header("1. Upload Raw RLHF JSON Data file(s)")  # Renamed header

uploaded_files = st.file_uploader(
    "Select CogniBench JSON batch file(s)",
    type=["json"],
    accept_multiple_files=True,
    help="Upload one or more JSON files containing tasks for evaluation.",
)

if uploaded_files:
    st.write(f"Uploaded {len(uploaded_files)} file(s):")
    uploaded_file_names = [f.name for f in uploaded_files]
    # Log only if the set of uploaded files has changed
    current_upload_key = tuple(sorted(uploaded_file_names))
    if st.session_state.get("last_uploaded_files_key") != current_upload_key:
        logger.info(
            f"Uploaded {len(uploaded_files)} files: {', '.join(uploaded_file_names)}"
        )
        st.session_state.last_uploaded_files_key = current_upload_key
    for uploaded_file in uploaded_files:
        st.write(f"- {uploaded_file.name}")
else:
    st.info("Please upload at least one batch file.")

# Placeholder for Folder Picker (if implemented later)
# st.write("*(Folder selection coming soon)*")

# --- Phase 1.5: Configuration Options ---
# Define available models based on the plan
AVAILABLE_MODELS = {
    "OpenAI": {
        "GPT-4O": "gpt-4o",
        "GPT-4 Turbo": "gpt-4-turbo",
        "GPT-4": "gpt-4",
        "GPT-3.5 Turbo": "gpt-3.5-turbo",
        "O1": "o1",
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

st.header("2. Configure Models and Prompts")

with st.expander("Model Configurations", expanded=True):
    col_structuring, col_judging = st.columns(2)

with col_structuring:
    st.subheader("Structuring Model Configuration")
    structuring_provider = st.selectbox(
        "Select Structuring Model Provider",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,  # Default to first provider
        key="structuring_provider_select",
    )
    structuring_model = st.selectbox(
        "Select Structuring Model",
        options=list(AVAILABLE_MODELS[structuring_provider].keys()),
        index=0,  # Default to first model
        key="structuring_model_select",
    )
    structuring_api_key = st.text_input(
        "Structuring API Key (Optional)",
        type="password",
        placeholder="Leave blank to use environment variable",
        key="structuring_api_key_input",
    )

with col_judging:
    st.subheader("Judge Model Configuration")
    judging_provider = st.selectbox(
        "Select Judge Model Provider",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,  # Default to first provider
        key="judging_provider_select",
    )
    judging_model = st.selectbox(
        "Select Judge Model",
        options=list(AVAILABLE_MODELS[judging_provider].keys()),
        index=0,  # Default to first model
        key="judging_model_select",
    )
    judging_api_key = st.text_input(
        "Judge API Key (Optional)",
        type="password",
        placeholder="Leave blank to use environment variable",
        key="judging_api_key_input",
    )

# Define available prompt templates for structuring and judging separately
STRUCTURING_TEMPLATES_DIR = PROMPT_TEMPLATES_DIR_ABS / "structuring"
JUDGING_TEMPLATES_DIR = PROMPT_TEMPLATES_DIR_ABS / "judging"


def get_templates(directory):
    try:
        return {
            f: str(directory / f) for f in os.listdir(directory) if f.endswith(".txt")
        }
    except FileNotFoundError:
        st.error(f"Prompt templates directory not found: {directory}")
        return {}


AVAILABLE_STRUCTURING_TEMPLATES = get_templates(STRUCTURING_TEMPLATES_DIR)
AVAILABLE_JUDGING_TEMPLATES = get_templates(JUDGING_TEMPLATES_DIR)

# Use requested provider names as keys
# Use requested provider names as keys


# Explicitly initialize session state variables for default selections
if "structuring_provider_select" not in st.session_state:
    st.session_state["structuring_provider_select"] = list(AVAILABLE_MODELS.keys())[0]
if "structuring_model_select" not in st.session_state:
    st.session_state["structuring_model_select"] = list(
        AVAILABLE_MODELS[st.session_state["structuring_provider_select"]].keys()
    )[0]
if "judging_provider_select" not in st.session_state:
    st.session_state["judging_provider_select"] = list(AVAILABLE_MODELS.keys())[0]
if "judging_model_select" not in st.session_state:
    st.session_state["judging_model_select"] = list(
        AVAILABLE_MODELS[st.session_state["judging_provider_select"]].keys()
    )[0]
if "structuring_template_select" not in st.session_state:
    st.session_state["structuring_template_select"] = (
        list(AVAILABLE_STRUCTURING_TEMPLATES.keys())[0]
        if AVAILABLE_STRUCTURING_TEMPLATES
        else None
    )
if "judging_template_select" not in st.session_state:
    st.session_state["judging_template_select"] = (
        list(AVAILABLE_JUDGING_TEMPLATES.keys())[0]
        if AVAILABLE_JUDGING_TEMPLATES
        else None
    )

# Initialize session state
if "selected_provider" not in st.session_state:
    st.session_state.selected_provider = list(AVAILABLE_MODELS.keys())[
        0
    ]  # Default to first provider
if "selected_model_name" not in st.session_state:
    st.session_state.selected_model_name = list(
        AVAILABLE_MODELS[st.session_state.selected_provider].keys()
    )[0]
if "selected_template_name" not in st.session_state:
    st.session_state.selected_template_name = (
        list(AVAILABLE_STRUCTURING_TEMPLATES.keys())[0]
        if AVAILABLE_STRUCTURING_TEMPLATES
        else None
    )
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "evaluation_running" not in st.session_state:
    st.session_state.evaluation_running = False
if "evaluation_results_paths" not in st.session_state:
    st.session_state.evaluation_results_paths = []
if "last_run_output" not in st.session_state:
    st.session_state.last_run_output = []
if "results_df" not in st.session_state:
    st.session_state.results_df = None  # To store the loaded DataFrame
if "eval_start_time" not in st.session_state:
    st.session_state.eval_start_time = None
if "eval_duration_str" not in st.session_state:
    st.session_state.eval_duration_str = None
if "previous_evaluation_running" not in st.session_state:
    st.session_state.previous_evaluation_running = False  # Track previous state
if "previous_evaluation_running" not in st.session_state:
    st.session_state.previous_evaluation_running = False  # Track previous state
# Ensure thread and queue are initialized if needed (might be redundant but safe)
if "worker_thread" not in st.session_state:
    st.session_state.worker_thread = None
if "output_queue" not in st.session_state:
    st.session_state.output_queue = queue.Queue()
if "stop_event" not in st.session_state:
    st.session_state.stop_event = threading.Event()


# --- Configuration Widgets (within Expander) ---
with st.expander("Prompt configurations", expanded=False):
    col_prompt1, col_prompt2 = st.columns(2)

    with col_prompt1:
        structuring_template = st.selectbox(
            "Select Structuring Prompt Template",
            options=list(AVAILABLE_STRUCTURING_TEMPLATES.keys()),
            key="structuring_template_select",
        )
        if st.button("View Structuring Prompt"):
            st.session_state.show_structuring = not st.session_state.get(
                "show_structuring", False
            )

    with col_prompt2:
        judging_template = st.selectbox(
            "Select Judge Prompt Template",
            options=list(AVAILABLE_JUDGING_TEMPLATES.keys()),
            key="judging_template_select",
        )
        if st.button("View Judging Prompt"):
            st.session_state.show_judging = not st.session_state.get(
                "show_judging", False
            )

    if st.button("View Config.yaml"):
        st.session_state.show_config = not st.session_state.get("show_config", False)

if st.session_state.get("show_structuring", False):
    with st.expander("Structuring Prompt", expanded=True):
        structuring_prompt_path = AVAILABLE_STRUCTURING_TEMPLATES[structuring_template]
        structuring_prompt_content = Path(structuring_prompt_path).read_text()
        st.code(structuring_prompt_content, language="text")

if st.session_state.get("show_judging", False):
    with st.expander("Judging Prompt", expanded=True):
        judging_prompt_path = AVAILABLE_JUDGING_TEMPLATES[judging_template]
        judging_prompt_content = Path(judging_prompt_path).read_text()
        st.code(judging_prompt_content, language="text")

if st.session_state.get("show_config", False):
    with st.expander("Config.yaml", expanded=True):
        config_content = BASE_CONFIG_PATH.read_text()
        st.code(config_content, language="yaml")
# --- Config Completeness Check (Moved After All Widgets) ---
# Validation logic for configuration completeness
config_complete = all(
    [
        st.session_state.get("structuring_provider_select"),
        st.session_state.get("structuring_model_select"),
        st.session_state.get("judging_provider_select"),
        st.session_state.get("judging_model_select"),
        st.session_state.get("structuring_template_select"),
        st.session_state.get("judging_template_select"),
    ]
)

if config_complete:
    st.success("✅ Configuration is complete.")
else:
    st.error("❌ Configuration is incomplete. Please ensure all fields are selected.")

st.markdown("---")
AVAILABLE_JUDGING_TEMPLATES = get_templates(JUDGING_TEMPLATES_DIR)

# --- Config Completeness Check (Moved After All Widgets) ---
# --- Display Current Configuration Summary (Moved Outside Expander) ---
st.subheader("Current Configuration Summary")

selected_structuring_template_path = AVAILABLE_STRUCTURING_TEMPLATES.get(
    structuring_template, "Not selected"
)
selected_judging_template_path = AVAILABLE_JUDGING_TEMPLATES.get(
    judging_template, "Not selected"
)

structuring_api_key_status = (
    "**Provided**" if structuring_api_key else "**Using Environment Variable**"
)
judging_api_key_status = (
    "**Provided**" if judging_api_key else "**Using Environment Variable**"
)

st.markdown(f"""
**Structuring Model:** `{structuring_provider}` - `{structuring_model}`
**Structuring Prompt:** `{selected_structuring_template_path}`

**Judge Model:** `{judging_provider}` - `{judging_model}`
**Judge Prompt:** `{selected_judging_template_path}`
""")


# Separator moved outside the expander, before the next section
st.markdown("---")


# --- Function to load and process results ---
@st.cache_data(show_spinner=False)  # Re-enable cache
def load_and_process_results(absolute_results_paths):
    """Loads data from _final_results.json files (given absolute paths) and processes into a DataFrame."""
    logger.info(
        f"Attempting to load and process results from: {absolute_results_paths}"
    )
    all_results_data = []
    processed_files_count = 0
    failed_files = []
    for file_path_str in absolute_results_paths:
        try:
            file_path = Path(file_path_str)  # Convert string path to Path object
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # --- Add check for nested 'results' key ---
                if "results" not in data or not isinstance(data["results"], list):
                    logger.error(
                        f"JSON file {file_path_str} does not contain a 'results' list."
                    )
                    failed_files.append(file_path_str)
                    continue  # Skip this file if structure is wrong
                # --- End check ---
                task_count = 0
                for task in data[
                    "results"
                ]:  # Iterate over the list inside the 'results' key
                    task_count += 1
                    task_id = task.get("task_id")
                    prompt = task.get("prompt")
                    ideal_response = task.get("ideal_response")
                    final_answer_gt = task.get("final_answer")
                    metadata = task.get("metadata", {})
                    subject = metadata.get("subject", "N/A")
                    complexity = metadata.get("complexity", "N/A")

                    eval_count = 0
                    for evaluation in task.get("evaluations", []):
                        eval_count += 1
                        model_id = evaluation.get("model_id")
                        model_response = evaluation.get("model_response")
                        human_eval = evaluation.get("human_evaluation", {})
                        judge_eval = evaluation.get("judge_evaluation", {})

                        flat_judge_eval = {}
                        if isinstance(judge_eval, dict):
                            for key, value in judge_eval.items():
                                if isinstance(value, dict):
                                    if key == "parsed_rubric_scores":
                                        for (
                                            rubric_name,
                                            rubric_details,
                                        ) in value.items():
                                            if isinstance(rubric_details, dict):
                                                flat_judge_eval[
                                                    f"judge_rubric_{rubric_name}_score"
                                                ] = rubric_details.get("score")
                                                flat_judge_eval[
                                                    f"judge_rubric_{rubric_name}_justification"
                                                ] = rubric_details.get("justification")
                                            else:
                                                flat_judge_eval[
                                                    f"judge_rubric_{rubric_name}"
                                                ] = rubric_details
                                    else:
                                        for sub_key, sub_value in value.items():
                                            flat_judge_eval[
                                                f"judge_{key}_{sub_key}"
                                            ] = sub_value
                                else:
                                    flat_judge_eval[f"judge_{key}"] = value
                        else:
                            flat_judge_eval["judge_evaluation_raw"] = judge_eval

                        aggregated_score = str(
                            flat_judge_eval.get("judge_aggregated_score", "N/A")
                        ).title()

                        all_results_data.append(
                            {
                                "task_id": task_id,
                                "model_id": model_id,
                                "subject": subject,
                                "complexity": complexity,
                                "aggregated_score": aggregated_score,
                                "prompt": prompt,
                                "ideal_response": ideal_response,
                                "model_response": model_response,
                                "final_answer_ground_truth": final_answer_gt,
                                **flat_judge_eval,
                                "human_preference": human_eval.get("preference"),
                                "human_rating": human_eval.get("rating"),
                            }
                        )
        except FileNotFoundError:
            st.error(f"Results file not found: {file_path_str}")
            logger.error(f"Results file not found: {file_path_str}")  # Added log
            failed_files.append(file_path_str)
            # Consider removing 'return None' if you want to process other files
            return None
        except json.JSONDecodeError:
            st.error(f"Error decoding JSON from file: {file_path_str}")
            logger.error(
                f"Error decoding JSON from file: {file_path_str}"
            )  # Added log
            failed_files.append(file_path_str)
            # Consider removing 'return None' if you want to process other files
            return None
        except Exception as e:
            st.error(f"Error processing file {file_path_str}: {e}")
            logger.error(f"Error processing file {file_path_str}: {e}")  # Added log
            failed_files.append(file_path_str)
        else:
            processed_files_count += 1  # Increment count on success

    # Log summary before returning
    if failed_files:
        logger.error(
            f"Failed to process {len(failed_files)} files: {', '.join(failed_files)}"
        )
    logger.info(
        f"Successfully processed {processed_files_count} result files out of {len(absolute_results_paths)}."
    )
    return pd.DataFrame(all_results_data)


# Removed duplicate/erroneous lines

action = st.radio(
    "Select Action",
    ["Run Evaluations", "Recreate Graphs from Existing Data"],
    horizontal=True,
)

if action == "Run Evaluations":
    st.header("Run Evaluations")
elif action == "Recreate Graphs from Existing Data":
    st.header("Recreate Graphs from Existing Data")

if action == "Run Evaluations":
    col1, col2 = st.columns(2)

    with col1:
        run_button = st.button(
            "🚀 Run Evaluations",
            type="primary",
            disabled=not uploaded_files
            or not st.session_state.selected_template_name
            or st.session_state.get("evaluation_running", False),
        )

    with col2:
        clear_cache_button = st.button("🧹 Clear LLM Cache")
        if clear_cache_button:
            logger.info("Clear LLM Cache button clicked.")

elif action == "Recreate Graphs from Existing Data":
    data_dir = COGNIBENCH_ROOT / "data"
    available_folders = sorted(
        [
            f.name
            for f in data_dir.iterdir()
            if f.is_dir()
            and (
                data_dir / f"{f.name}/{f.name.split('_')[0]}_final_results.json"
            ).exists()
        ],
        key=lambda x: (data_dir / x).stat().st_mtime,
        reverse=True,
    )
    selected_folders = st.multiselect(
        "Select folders to regenerate graphs from existing evaluation data:",
        options=available_folders,
        help="Select one or more folders containing existing evaluation data.",
    )

    if st.button("📊 Regenerate Graphs", disabled=not selected_folders):
        logger.info(
            f"Regenerate Graphs button clicked for folders: {selected_folders}"
        )
        evaluation_results_paths = []
        for folder in selected_folders:
            folder_path = data_dir / folder
            batch_name = folder.split("_")[0]
            results_file = folder_path / f"{batch_name}_final_results.json"
            if results_file.exists():
                evaluation_results_paths.append(str(results_file))
            else:
                st.warning(f"No final results file found in {folder}")

        if evaluation_results_paths:
            # Pass the absolute paths directly
            st.session_state.results_df = load_and_process_results(
                evaluation_results_paths
            )
            st.success("Graphs regenerated successfully!")
            logger.info("Graphs regenerated successfully.")
            st.rerun()
        else:
            st.error("No valid evaluation data found in selected folders.")
            logger.warning(
                "No valid evaluation data found in selected folders for graph regeneration."
            )


# --- Config Validation Helper ---
# (Similar to the one added in other scripts)
def validate_config(config: Dict[str, Any]) -> bool:
    """Performs basic validation on the loaded configuration dictionary."""
    if not config:  # Check if config is None or empty
        st.error("Config validation failed: Configuration data is empty.")
        return False

    required_sections = ["llm_client", "evaluation_settings", "output_options"]
    for section in required_sections:
        if section not in config or not isinstance(config[section], dict):
            st.error(
                f"Config validation failed: Missing or invalid section '{section}'."
            )
            return False

    eval_settings = config["evaluation_settings"]
    required_eval_keys = [
        "judge_model",
        "prompt_template",
        "expected_criteria",
        "allowed_scores",
    ]
    for key in required_eval_keys:
        if key not in eval_settings:
            st.error(
                f"Config validation failed: Missing key '{key}' in 'evaluation_settings'."
            )
            return False
        # Specific type checks
        if key in ["expected_criteria", "allowed_scores"] and not isinstance(
            eval_settings[key], list
        ):
            st.error(
                f"Config validation failed: Key '{key}' in 'evaluation_settings' must be a list."
            )
            return False
        elif key in ["judge_model", "prompt_template"] and not isinstance(
            eval_settings[key], str
        ):
            st.error(
                f"Config validation failed: Key '{key}' in 'evaluation_settings' must be a string."
            )
            return False

    # Validate output_options has results_file (used by get endpoint, though not directly here)
    output_options = config["output_options"]
    if "results_file" not in output_options or not isinstance(
        output_options["results_file"], str
    ):
        st.error(
            "Config validation failed: Missing or invalid 'results_file' key in 'output_options'."
        )
        return False

    # Could add more checks (e.g., non-empty lists/strings) if needed
    st.success(
        "Internal configuration validation successful."
    )  # Use st.success for UI feedback
    return True


# --- Function to run evaluation in a separate thread ---
def run_evaluation_script(input_file_path, config_file_path, output_queue, stop_event):
    """Runs the batch evaluation script and puts output lines into a queue."""
    logger.info(
        f"Starting evaluation script for input: {input_file_path}, config: {config_file_path}"
    )
    command = [
        sys.executable,
        str(RUN_BATCH_SCRIPT_PATH),
        str(input_file_path),
        "--config",
        str(config_file_path),
    ]
    try:
        output_queue.put(f"INFO: Running command: {' '.join(command)}")
        output_queue.put(f"INFO: Working directory: {COGNIBENCH_ROOT}")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=COGNIBENCH_ROOT,
            bufsize=1,
        )

        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                if stop_event.is_set():
                    process.terminate()
                    output_queue.put("INFO: Evaluation stopped by user request.")
                    break
                output_queue.put(line.strip())
        # stdout is automatically closed when the process exits and the loop finishes
        process.wait()  # Wait for the process to complete
        output_queue.put(f"INFO: Process finished with exit code {process.returncode}")

    except FileNotFoundError:
        output_queue.put(
            f"ERROR: Python executable or script not found. Command: {' '.join(command)}"
        )
    except Exception as e:
        output_queue.put(f"ERROR: An unexpected error occurred: {e}")
    finally:
        output_queue.put(None)
        stop_event.clear()


# --- Run Evaluation Logic ---
if (
    action == "Run Evaluations"
    and run_button
    and uploaded_files
    and st.session_state.selected_template_name
):
    st.session_state.evaluation_running = True
    st.session_state.previous_evaluation_running = False  # Reset previous state tracker
    # Clear previous results and logs when starting a new run
    st.session_state.last_run_output = []
    st.session_state.evaluation_results_paths = []
    st.session_state.results_df = None
    st.session_state.eval_duration_str = None  # Clear duration string
    st.info(
        "Starting new evaluation run... Previous results cleared."
    )  # Optional user feedback
    st.session_state.eval_start_time = time.time()
    st.session_state.eval_duration_str = None
    st.session_state.current_file_index = 0
    st.session_state.output_queue = queue.Queue()
    st.session_state.worker_thread = None
    st.session_state.temp_dir = tempfile.TemporaryDirectory()
    st.session_state.temp_dir_path = Path(st.session_state.temp_dir.name)
    st.rerun()  # Rerun to disable button and show progress area

if st.session_state.get("evaluation_running", False):
    st.header("Evaluation Progress ⏳")
    progress_area = st.container()
    stop_button = st.button("🛑 Stop Processing", type="primary")
    if stop_button:
        st.session_state.stop_requested = True
    log_expander = st.expander("Show Full Logs", expanded=False)
    log_placeholder = log_expander.empty()

    with st.spinner("Evaluations running... Please wait."):
        progress_bar = progress_area.progress(0.0)
        progress_text = progress_area.text("Starting evaluation...")
        log_placeholder.code(
            "\n".join(st.session_state.last_run_output[-1000:]), language="log"
        )

    current_index = st.session_state.get("current_file_index", 0)
    total_files = len(uploaded_files) if uploaded_files else 0

    # --- Start worker thread if not already running for the current file ---
    if st.session_state.worker_thread is None and current_index < total_files:
        uploaded_file = uploaded_files[current_index]
        progress_area.info(
            f"Processing file {current_index + 1}/{total_files}: **{uploaded_file.name}**"
        )
        try:
            # Ensure temp dir exists (might be cleaned up unexpectedly on reruns)
            if (
                "temp_dir_path" not in st.session_state
                or not st.session_state.temp_dir_path.exists()
            ):
                st.session_state.temp_dir = tempfile.TemporaryDirectory()
                st.session_state.temp_dir_path = Path(st.session_state.temp_dir.name)

            # 1. Save uploaded file temporarily
            temp_input_path = st.session_state.temp_dir_path / uploaded_file.name
            with open(temp_input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 2. Load and modify config
            with open(BASE_CONFIG_PATH, "r") as f:
                config_data = yaml.safe_load(f)

            selected_provider = st.session_state.selected_provider
            selected_model_name = st.session_state.selected_model_name
            selected_model_api_id = AVAILABLE_MODELS[selected_provider][
                selected_model_name
            ]
            api_key = st.session_state.api_key

            config_data["evaluation_settings"]["judge_model"] = selected_model_api_id
            config_data["evaluation_settings"]["structuring_prompt_template"] = (
                selected_structuring_template_path
            )
            config_data["evaluation_settings"]["judging_prompt_template"] = (
                selected_judging_template_path
            )
            config_data["llm_client"]["provider"] = selected_provider
            config_data["llm_client"]["model"] = selected_model_api_id
            if api_key:
                config_data["llm_client"]["api_key"] = api_key
            else:
                provider_lower_for_env = selected_provider.lower()
                if provider_lower_for_env == "openai":
                    config_data["llm_client"]["api_key"] = "${OPENAI_API_KEY}"
                elif provider_lower_for_env == "anthropic":
                    config_data["llm_client"]["api_key"] = "${ANTHROPIC_API_KEY}"
                elif provider_lower_for_env == "google":
                    config_data["llm_client"]["api_key"] = "${GOOGLE_API_KEY}"
                # Add other providers if needed

            # 3. Validate the constructed config BEFORE saving and running
            if not validate_config(config_data):
                st.error("Generated configuration is invalid. Cannot start evaluation.")
                st.session_state.evaluation_running = False
                st.rerun()  # Stop and show error

            # 4. Save temporary config (if validation passed)
            temp_config_path = (
                st.session_state.temp_dir_path / f"temp_config_{current_index}.yaml"
            )
            with open(temp_config_path, "w") as f:
                yaml.dump(config_data, f)

            # 5. Start evaluation thread (including structuring and judging)
            if st.session_state.eval_start_time is None:
                st.session_state.eval_start_time = (
                    time.time()
                )  # Start timing here to include structuring

            st.session_state.output_queue = queue.Queue()
            st.session_state.worker_thread = threading.Thread(
                target=run_evaluation_script,
                args=(
                    temp_input_path,
                    temp_config_path,
                    st.session_state.output_queue,
                    st.session_state.stop_event,
                ),
                daemon=True,
            )
            st.session_state.worker_thread.start()
            progress_area.info(
                f"Started evaluation (structuring + judging) thread for {uploaded_file.name}..."
            )
            # Don't rerun here, let the loop below handle updates

        except Exception as e:
            progress_area.error(
                f"Error preparing evaluation for {uploaded_file.name}: {e}"
            )
            st.session_state.evaluation_running = False  # Stop evaluation on error
            st.session_state.worker_thread = "Error"
            st.rerun()  # Rerun to show error state

    # --- Process output queue while thread is potentially running ---
    thread_is_alive = (
        isinstance(st.session_state.worker_thread, threading.Thread)
        and st.session_state.worker_thread.is_alive()
    )
    thread_finished_this_run = False
    lines_processed_this_run = []

    while not st.session_state.output_queue.empty():
        line = st.session_state.output_queue.get()
        if line is None:  # End signal from thread
            if isinstance(st.session_state.worker_thread, threading.Thread):
                st.session_state.worker_thread.join(timeout=1.0)  # Wait briefly
            st.session_state.worker_thread = None  # Mark as finished
            thread_finished_this_run = True
            st.session_state.current_file_index += 1  # Move to next file index
            break  # Exit while loop for this run
        else:
            # Log every line received from the subprocess queue at DEBUG level
            logger.debug(f"Queue Line: {line}")
            lines_processed_this_run.append(line)
            st.session_state.last_run_output.append(line)

            # Progress update based on backend message
            progress_match = re.match(r"PROGRESS: Task (\d+)/(\d+)", line)
            if progress_match:
                current_task = int(progress_match.group(1))
                total_tasks_backend = int(progress_match.group(2))
                if total_tasks_backend > 0:
                    progress_percentage = float(current_task) / total_tasks_backend
                    progress_bar.progress(progress_percentage)
                    progress_text.text(
                        f"Evaluating Task {current_task}/{total_tasks_backend}..."
                    )
                else:
                    progress_text.text(
                        f"Evaluating Task {current_task}/Unknown Total..."
                    )
            elif "Evaluating task" in line:
                progress_text.text(line)

            # Parse for the explicit results file path marker
            stripped_line = line.strip()
            prefix = "FINAL_RESULTS_PATH: "
            if stripped_line.startswith(prefix):
                # Extract the absolute path after the prefix
                raw_path = stripped_line[len(prefix) :].strip()
                logger.info(
                    f"Found potential results file line: '{raw_path}'"
                )  # Log raw path
                # Proceed with path processing using raw_path
                results_path = raw_path  # Initialize with raw path
                try:
                    abs_path = Path(raw_path)
                    # Check if it's already absolute and within the project root for consistency
                    if abs_path.is_absolute():
                        logger.info(f"Path '{raw_path}' is absolute.")
                        if COGNIBENCH_ROOT in abs_path.parents:
                            # Convert to relative if inside project root
                            results_path = str(abs_path.relative_to(COGNIBENCH_ROOT))
                            logger.info(
                                f"Converted absolute path to relative: '{results_path}'"
                            )
                        else:
                            # Keep absolute if outside project root (less likely but handle)
                            results_path = str(abs_path)
                            logger.info(
                                f"Keeping absolute path (outside project root): '{results_path}'"
                            )
                    else:
                        # If it's relative, assume it's relative to COGNIBENCH_ROOT
                        results_path = str(Path(raw_path))  # Keep as relative string
                        logger.info(
                            f"Path '{raw_path}' is relative, keeping as: '{results_path}'"
                        )

                    # Ensure we don't add duplicates
                    if results_path not in st.session_state.evaluation_results_paths:
                        logger.info(
                            f"Appending processed path to list: '{results_path}'"
                        )  # Log before append
                        st.session_state.evaluation_results_paths.append(results_path)
                        progress_area.success(
                            f"Found and stored results file path: {results_path}"
                        )
                    else:
                        logger.info(
                            f"Path '{results_path}' already in list, skipping append."
                        )  # Log if skipped

                except Exception as path_e:
                    progress_area.warning(
                        f"Could not process results path '{raw_path}': {path_e}"
                    )
            # else: # Keep commented
            #     if "_final_results.json" in line:
            #         st.write(f"DEBUG (Log Check): Regex *failed* but string present in line: '{line}'")

    # Update log display only if new lines were processed
    if lines_processed_this_run:
        log_placeholder.code(
            "\n".join(st.session_state.last_run_output[-1000:]), language="log"
        )

    # Check if processing should continue or finish
    if thread_finished_this_run:
        if st.session_state.current_file_index >= total_files:  # All files done
            st.session_state.evaluation_running = (
                False  # Mark as finished *before* final rerun
            )
            if hasattr(
                st.session_state, "temp_dir_obj"
            ):  # Use temp_dir_obj for cleanup
                st.session_state.temp_dir_obj.cleanup()
                st.session_state.temp_dir_obj = None
                st.session_state.temp_dir_path = None
            # --- Calculations and Summary on Completion ---
            end_time = time.time()
            duration_seconds = end_time - st.session_state.eval_start_time

            # Load results first to get counts
            results_df = None
            if st.session_state.evaluation_results_paths:
                # Construct absolute paths from potentially relative paths stored during the run
                absolute_paths = [
                    str(COGNIBENCH_ROOT / p)
                    for p in st.session_state.evaluation_results_paths
                ]
                logger.info(
                    f"Attempting to load results after run from absolute paths: {absolute_paths}"
                )
                # Pass the absolute paths directly to the loading function
                results_df = load_and_process_results(
                    absolute_paths  # Use the constructed absolute paths
                )
                st.session_state.results_df = (
                    results_df  # Store loaded df in session state
                )
                # Add logging here to check if it was loaded before the rerun
                logger.info(
                    f"After evaluation run, results_df is None: {st.session_state.results_df is None}"
                )
                if st.session_state.results_df is not None:
                    logger.info(
                        f"Loaded DataFrame shape after run: {st.session_state.results_df.shape}"
                    )
            else:
                logger.warning(
                    "No evaluation_results_paths found in session state after run."
                )  # Added log
                progress_area.warning(
                    "Could not find paths to results files in the logs. Cannot calculate averages or display graphs."  # Updated warning
                )

            # Format duration human-readably
            parts = []
            hours, rem = divmod(duration_seconds, 3600)
            minutes, seconds_rem = divmod(rem, 60)
            if hours >= 1:
                parts.append(f"{int(hours)} hour{'s' if int(hours) != 1 else ''}")
            if minutes >= 1:
                parts.append(f"{int(minutes)} minute{'s' if int(minutes) != 1 else ''}")
            if (
                seconds_rem >= 1 or not parts
            ):  # Show seconds if > 0 or if it's the only unit
                parts.append(
                    f"{int(seconds_rem)} second{'s' if int(seconds_rem) != 1 else ''}"
                )
            st.session_state.eval_duration_str = ", ".join(parts)

            # Count API calls from logs
            structuring_calls = 0
            judging_calls = 0
            for log_line in st.session_state.last_run_output:
                if "STRUCTURING_CALL:" in log_line:
                    structuring_calls += 1
                elif "JUDGING_CALL:" in log_line:
                    judging_calls += 1

            # Calculate averages
            avg_time_per_task_str = "N/A"
            avg_time_per_model_eval_str = "N/A"
            num_unique_tasks = 0
            num_evaluations = 0

            if results_df is not None and not results_df.empty:
                num_unique_tasks = results_df["task_id"].nunique()
                num_evaluations = len(results_df)
                if num_unique_tasks > 0:
                    avg_time_per_task = duration_seconds / num_unique_tasks
                    avg_time_per_task_str = f"{avg_time_per_task:.2f} seconds"
                if num_evaluations > 0:
                    avg_time_per_model_eval = duration_seconds / num_evaluations
                    avg_time_per_model_eval_str = (
                        f"{avg_time_per_model_eval:.2f} seconds"
                    )

            # Construct final summary message
            summary_message = f"""
            All evaluations complete!
            - **Total Duration:** {st.session_state.eval_duration_str}
            - **Structuring Calls:** {structuring_calls}
            - **Judging Calls:** {judging_calls}
            - **Total Tasks Processed:** {num_unique_tasks}
            - **Total Model Evaluations:** {num_evaluations}
            - **Avg. Time per Task:** {avg_time_per_task_str}
            - **Avg. Time per Model Evaluation:** {avg_time_per_model_eval_str}
            """

            # Display final messages
            progress_area.success("Evaluation Run Summary:")
            progress_area.markdown(
                summary_message
            )  # Use markdown for better formatting

            if st.session_state.evaluation_results_paths:
                progress_area.write("Generated results files:")
                for path in st.session_state.evaluation_results_paths:
                    progress_area.code(path)
            # else: # Warning moved earlier

            # Clean up state AFTER calculations and display
            if "current_file_index" in st.session_state:
                del st.session_state.current_file_index
            if "output_queue" in st.session_state:
                del st.session_state.output_queue
            if "worker_thread" in st.session_state:
                del st.session_state.worker_thread
            st.rerun()  # Final rerun to trigger results loading/display
        else:
            # More files to process, trigger rerun to start next thread
            st.rerun()
    elif st.session_state.worker_thread == "Error":
        # Handle error state
        st.session_state.evaluation_running = False
        if hasattr(st.session_state, "temp_dir_obj"):  # Use temp_dir_obj for cleanup
            st.session_state.temp_dir_obj.cleanup()
            st.session_state.temp_dir_obj = None
            st.session_state.temp_dir_path = None
        # Clean up state
        if "current_file_index" in st.session_state:
            del st.session_state.current_file_index
        if "output_queue" in st.session_state:
            del st.session_state.output_queue
        if "worker_thread" in st.session_state:
            del st.session_state.worker_thread
        st.rerun()
    elif thread_is_alive:
        # Thread is still running, schedule next check without immediate rerun if no new lines
        if not lines_processed_this_run:
            time.sleep(0.5)  # Longer sleep if queue was empty
            st.rerun()
        else:
            time.sleep(0.1)  # Shorter sleep if queue had lines
            st.rerun()


# --- Display Final Logs After Completion ---
if not st.session_state.get("evaluation_running", False) and st.session_state.get(
    "last_run_output"
):
    st.subheader("Final Evaluation Logs")
    with st.expander("Show Full Logs", expanded=False):
        st.code(
            "\n".join(st.session_state.last_run_output[-1000:]), language="log"
        )  # Show last 1000 lines

# --- Clear Cache Logic ---
if action == "Run Evaluations" and clear_cache_button:
    # Define potential cache file paths relative to CogniBench root
    cache_files_to_clear = [
        COGNIBENCH_ROOT / "openai_cache.db",
        # Add other provider cache files here if known (e.g., anthropic_cache.db)
    ]
    cleared_count = 0
    errors = []
    for cache_file in cache_files_to_clear:
        try:
            if cache_file.is_file():
                cache_file.unlink()
                st.toast(f"Cleared cache file: {cache_file.name}", icon="✅")
                cleared_count += 1
            # else:
            #     st.toast(f"Cache file not found: {cache_file.name}", icon="ℹ️")
        except Exception as e:
            st.error(f"Error clearing cache file {cache_file.name}: {e}")
            errors.append(cache_file.name)

    if cleared_count > 0 and not errors:
        st.success(f"Successfully cleared {cleared_count} cache file(s).")
    elif cleared_count == 0 and not errors:
        st.info("No known cache files found to clear.")
    elif errors:
        st.warning(
            f"Cleared {cleared_count} cache file(s), but encountered errors with: {', '.join(errors)}"
        )

# --- Phase 3: Visualize Results ---
st.header("4. Results")


# Removed duplicate load_and_process_results function definition.
# The primary definition starting at line 350 will be used.

# --- Load data if results paths exist ---
# Load data if evaluation is not running, paths exist, and data isn't already loaded
if (
    not st.session_state.get("evaluation_running", False)
    and st.session_state.get("evaluation_results_paths")
    and st.session_state.get("results_df") is None  # Only load if not already loaded
):
    with st.spinner("Loading and processing results..."):
        # Use the cached function to load data
        st.session_state.results_df = load_and_process_results(
            st.session_state.evaluation_results_paths
        )

# --- Display Results if DataFrame is loaded ---
logger.info(
    f"Checking session state before graph display. results_df is None: {st.session_state.get('results_df') is None}"
)
if st.session_state.results_df is not None:
    logger.info(
        "Results DataFrame found in session state. Proceeding with visualization."
    )
    df = st.session_state.results_df
    st.success(f"Loaded {len(df)} evaluation results.")
    # Display evaluation duration if available
    if st.session_state.get("eval_duration_str"):
        st.info(
            f"Total evaluation time: {st.session_state.eval_duration_str}"
        )  # Display the new format

    # --- Enhanced Filters ---
    st.sidebar.header("Enhanced Filters")
    available_models = (
        df["model_id"].unique().tolist() if "model_id" in df.columns else []
    )
    available_tasks = df["task_id"].unique().tolist() if "task_id" in df.columns else []
    available_subjects = (
        df["subject"].unique().tolist() if "subject" in df.columns else []
    )
    available_scores = (
        df["aggregated_score"].unique().tolist()
        if "aggregated_score" in df.columns
        else []
    )

    selected_models = st.sidebar.multiselect(
        "Filter by Model:", available_models, default=available_models
    )
    selected_tasks = st.sidebar.multiselect(
        "Filter by Task ID:", available_tasks, default=available_tasks
    )
    selected_subjects = st.sidebar.multiselect(
        "Filter by Subject:", available_subjects, default=available_subjects
    )
    selected_scores = st.sidebar.multiselect(
        "Filter by Aggregated Score:", available_scores, default=available_scores
    )

    # Apply filters
    filtered_df = df.copy()
    if selected_models:
        filtered_df = filtered_df[filtered_df["model_id"].isin(selected_models)]
    if selected_tasks:
        filtered_df = filtered_df[filtered_df["task_id"].isin(selected_tasks)]
    if selected_subjects:
        filtered_df = filtered_df[filtered_df["subject"].isin(selected_subjects)]
    if selected_scores:
        filtered_df = filtered_df[filtered_df["aggregated_score"].isin(selected_scores)]

    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
        logger.warning("Filtered DataFrame is empty. No graphs will be generated.")
    else:
        logger.info(f"Filtered DataFrame shape: {filtered_df.shape}")
        # --- Overall Performance Chart (Using aggregated_score) ---
        st.subheader("Overall Performance by Model")
        logger.info("Attempting to generate 'Overall Performance by Model' chart.")
        agg_score_col = "aggregated_score"
        if agg_score_col in filtered_df.columns and "model_id" in filtered_df.columns:
            logger.info(
                f"Required columns ('{agg_score_col}', 'model_id') found for performance chart."
            )
            performance_counts = (
                filtered_df.groupby(["model_id", agg_score_col])
                .size()
                .reset_index(name="count")
            )
            logger.info(
                f"Data for performance chart:\n{performance_counts.to_string()}"
            )
            category_orders = {agg_score_col: ["Pass", "Fail", "None"]}

            logger.info("Calling px.bar for performance chart.")
            fig_perf = px.bar(
                performance_counts,
                x="model_id",
                y="count",
                color=agg_score_col,
                title="Aggregated Score Count per Model",
                labels={
                    "model_id": "Model",
                    "count": "Number of Tasks",
                    agg_score_col: "Aggregated Score",
                },
                barmode="group",
                category_orders=category_orders,
                color_discrete_map=COLOR_MAP,
            )
            st.plotly_chart(fig_perf, use_container_width=True)
            logger.info("Called st.plotly_chart for performance chart.")
        else:
            logger.warning(
                f"Skipping Overall Performance chart. Required columns ('model_id', '{agg_score_col}') not found."
            )
            st.warning(
                f"Could not generate Overall Performance chart. Required columns ('model_id', '{agg_score_col}') not found."
            )

        # --- Rubric Score Breakdown ---
        st.subheader("Rubric Score Analysis")
        logger.info("Attempting to generate 'Rubric Score Analysis' charts.")
        rubric_cols = [
            col
            for col in filtered_df.columns
            if col.startswith("judge_rubric_") and col.endswith("_score")
        ]

        if rubric_cols and "model_id" in filtered_df.columns:
            logger.info(
                f"Found rubric columns: {rubric_cols} and 'model_id'. Proceeding with rubric charts."
            )
            rubric_melted = filtered_df.melt(
                id_vars=["model_id"],
                value_vars=rubric_cols,
                var_name="rubric_criterion",
                value_name="score",
            )
            rubric_melted["rubric_criterion"] = (
                rubric_melted["rubric_criterion"]
                .str.replace("judge_rubric_", "")
                .str.replace("_score", "")
                .str.replace("_", " ")
                .str.title()
            )
            rubric_counts = (
                rubric_melted.groupby(["model_id", "rubric_criterion", "score"])
                .size()
                .reset_index(name="count")
            )

            logger.info("Calling px.bar for rubric distribution chart.")
            fig_rubric = px.bar(
                rubric_counts,
                x="rubric_criterion",
                y="count",
                color="score",
                title="Rubric Score Distribution per Criterion",
                labels={
                    "rubric_criterion": "Rubric Criterion",
                    "count": "Count",
                    "score": "Score",
                },
                barmode="group",
                facet_col="model_id",
                category_orders={"score": ["Yes", "Partial", "No", "N/A"]},
                color_discrete_map={
                    "Yes": "green",
                    "Partial": "orange",
                    "No": "red",
                    "N/A": "grey",
                },
            )
            fig_rubric.update_xaxes(tickangle=45)
            # Clean up facet titles (remove "model_id=")
            fig_rubric.for_each_annotation(
                lambda a: a.update(text=a.text.split("=")[-1])
            )
            st.plotly_chart(fig_rubric, use_container_width=True)
            logger.info("Called st.plotly_chart for rubric distribution chart.")

            # --- Rubric Score Distribution per Model (New Graph) ---
            st.subheader("Rubric Score Distribution per Model")
            logger.info(
                "Attempting to generate 'Rubric Score Distribution per Model' chart."
            )
            logger.info("Calling px.bar for rubric distribution per model chart.")
            fig_rubric_model = px.bar(
                rubric_counts,
                x="model_id",  # X-axis is now model
                y="count",
                color="score",
                title="Rubric Score Distribution per Model",
                labels={
                    "model_id": "Model",
                    "count": "Count",
                    "score": "Score",
                    "rubric_criterion": "Rubric Criterion",  # Label for facet
                },
                barmode="group",
                facet_col="rubric_criterion",  # Facet by criterion
                category_orders={"score": ["Yes", "Partial", "No", "N/A"]},
                color_discrete_map={
                    "Yes": "green",
                    "Partial": "orange",
                    "No": "red",
                    "N/A": "grey",
                },
            )
            fig_rubric_model.update_xaxes(tickangle=45)
            # Clean up facet titles (remove "rubric_criterion=")
            fig_rubric_model.for_each_annotation(
                lambda a: a.update(text=a.text.split("=")[-1])
            )
            st.plotly_chart(fig_rubric_model, use_container_width=True)
            logger.info(
                "Called st.plotly_chart for rubric distribution per model chart."
            )
            # --- Human Review Status ---
            st.subheader("Human Review Status")
            logger.info("Attempting to generate 'Human Review Status' chart.")
            review_status_col = (
                "judge_human_review_status"  # Assuming this is the column name
            )
            if (
                review_status_col in filtered_df.columns
                and "model_id"
                in filtered_df.columns  # Added check for model_id consistency
                and "model_id" in filtered_df.columns
            ):
                # --- Create Table Data First (Before fillna) ---
                # Filter for tasks needing review (case-insensitive and strip whitespace)
                # Apply on a copy to avoid modifying the original filtered_df used by other plots yet
                df_for_table_filter = filtered_df.copy()
                # Simplified filter attempt: compare lowercase directly, relying on .str to handle NaN
                review_needed_df = df_for_table_filter[
                    df_for_table_filter[review_status_col].str.lower() == "needs review"
                ]

                # --- Prepare Data for Graph (Now fillna) ---
                # Use a separate copy for graph calculation to keep original filtered_df clean if needed elsewhere
                df_for_graph = filtered_df.copy()
                df_for_graph[review_status_col] = df_for_graph[
                    review_status_col
                ].fillna("N/A")

                review_counts = (
                    df_for_graph.groupby(["model_id", review_status_col])
                    .size()
                    .reset_index(name="count")
                )

                # Define order and colors using the actual values from the data
                category_orders = {
                    review_status_col: ["Needs Review", "Not Required", "N/A"]
                }  # Use actual values

                # --- Create Graph ---
                logger.info("Calling px.bar for human review status chart.")
                fig_review = px.bar(
                    review_counts,  # Use counts derived from df_for_graph (with N/A)
                    x="model_id",
                    y="count",
                    color=review_status_col,
                    title="Human Review Status Count per Model",
                    labels={
                        "model_id": "Model",
                        "count": "Number of Tasks",
                        review_status_col: "Needs Human Review?",
                    },
                    barmode="group",
                    category_orders=category_orders,
                    color_discrete_map=COLOR_MAP,
                    text="count",  # Add count numbers on bars
                )

                fig_review.update_traces(
                    textposition="outside"
                )  # Position text labels outside bars
                st.plotly_chart(fig_review, use_container_width=True)
                logger.info("Called st.plotly_chart for human review status chart.")

                # --- Human Review Explorer ---
                st.subheader("Tasks Flagged for Human Review")
                # Display the review_needed_df created *before* fillna
                if not review_needed_df.empty:
                    logger.info(
                        f"Displaying {len(review_needed_df)} tasks flagged for human review."
                    )
                    st.write(
                        f"Found {len(review_needed_df)} evaluations flagged for human review."
                    )
                    # Select relevant columns to display
                    review_cols_to_show = {
                        "task_id": "Task ID",
                        "model_id": "Model",
                        "subject": "Subject",
                        "complexity": "Complexity",
                        "aggregated_score": "Aggregated Score",
                        review_status_col: "Review Status",
                        "judge_justification": "Judge Justification",  # Assuming this column exists
                        "prompt": "Prompt",
                        "model_response": "Model Response",
                    }
                    # Filter display_cols to only those present in the dataframe
                    actual_review_cols = [
                        col
                        for col in review_cols_to_show.keys()
                        if col in review_needed_df.columns
                    ]
                    # Display the pre-filtered dataframe
                    st.dataframe(
                        review_needed_df[actual_review_cols].rename(
                            columns=review_cols_to_show
                        )
                    )
                else:
                    logger.info(
                        "No tasks flagged for human review based on current filters."
                    )
                    st.info(
                        "No tasks flagged for human review based on current filters."
                    )

                # --- Human Review Input ---
                if not review_needed_df.empty:
                    st.subheader("Human Review Input")
                    selected_review_task = st.selectbox(
                        "Select Task ID for Review:",
                        review_needed_df["task_id"].unique(),
                    )
                    review_task_details = review_needed_df[
                        review_needed_df["task_id"] == selected_review_task
                    ].iloc[0]

                    st.write("### Review Task Details")
                    st.json(review_task_details.to_dict())

                    corrected_scores = {}
                    rubric_cols = [
                        col
                        for col in review_task_details.index
                        if col.startswith("judge_rubric_") and col.endswith("_score")
                    ]
                    for rubric in rubric_cols:
                        corrected_scores[rubric] = st.selectbox(
                            f"Corrected Score for {rubric.replace('judge_rubric_', '').replace('_score', '').replace('_', ' ').title()}:",
                            ["Yes", "Partial", "No", "N/A"],
                            index=["Yes", "Partial", "No", "N/A"].index(
                                review_task_details[rubric]
                            ),
                        )

                    review_comments = st.text_area("Review Comments:")

                    if st.button("Save Human Review"):
                        logger.info(
                            f"Saving human review for task {selected_review_task}."
                        )
                        review_task_details["human_corrected_scores"] = corrected_scores
                        review_task_details["human_review_comments"] = review_comments
                        review_task_details["human_review_status"] = "Reviewed"
                        review_task_details["human_review_timestamp"] = (
                            datetime.utcnow().isoformat() + "Z"
                        )
                        st.success("Human review saved successfully.")
                        # Here you would typically save these details back to your data storage

            else:
                logger.warning(
                    f"Skipping Human Review Status chart/explorer. Required columns ('model_id', '{review_status_col}') not found."
                )
                st.warning(
                    f"Could not generate Human Review Status chart/explorer. Required columns ('model_id', '{review_status_col}') not found."
                )
        # This else corresponds to the `if rubric_cols and "model_id" in filtered_df.columns:` check
        else:
            st.warning(
                "Skipping Rubric Score Analysis / Human Review sections. Rubric score columns not found or 'model_id' missing."
            )

        # --- Task-Level Explorer (Should be outside the rubric/review if/else, but inside the main results display else) ---
        st.subheader("Detailed Task-Level Explorer")
        logger.info("Displaying 'Detailed Task-Level Explorer'.")
        selected_task = st.selectbox(
            "Select Task ID for Detailed View:", filtered_df["task_id"].unique()
        )
        task_details = (
            filtered_df[filtered_df["task_id"] == selected_task].iloc[0].to_dict()
        )
        st.json(task_details)
        display_cols = {
            "task_id": "Task ID",
            "model_id": "Model",
            "subject": "Subject",
            "complexity": "Complexity",
            "aggregated_score": "Aggregated Score",
            "human_preference": "Human Preference",
            "human_rating": "Human Rating",
            "prompt": "Prompt",
            "ideal_response": "Ideal Response",
            "model_response": "Model Response",
        }
        cols_to_show = [
            col for col in display_cols.keys() if col in filtered_df.columns
        ]
        st.dataframe(filtered_df[cols_to_show].rename(columns=display_cols))

# Display message if results haven't loaded but paths exist (e.g., error during load)
elif (
    st.session_state.evaluation_results_paths
    and st.session_state.results_df is None
    and not st.session_state.evaluation_running
):
    st.error("Evaluation finished, but failed to load or process results.")
# Display message if evaluation is not running and no results are loaded/paths found
elif (
    not st.session_state.evaluation_running
    and not st.session_state.evaluation_results_paths
):
    st.info("Upload batch file(s) and click 'Run Evaluations' to see results.")

# Update previous running state at the very end of the script run
st.session_state.previous_evaluation_running = st.session_state.evaluation_running
