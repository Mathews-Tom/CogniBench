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
logger = logging.getLogger("streamlit")
if "logging_setup_complete" not in st.session_state:
    setup_logging()  # Call the setup function
    st.session_state.logging_setup_complete = True  # Mark as done
    # Get the specific logger for streamlit
    logger = logging.getLogger("streamlit")
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
st.header("Upload Raw RLHF JSON Data file(s)")  # Renamed header

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

st.header("Configure Models and Prompts")

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
    st.subheader("Judging Model Configuration")
    judging_provider = st.selectbox(
        "Select Judging Model Provider",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,  # Default to first provider
        key="judging_provider_select",
    )
    judging_model = st.selectbox(
        "Select Judging Model",
        options=list(AVAILABLE_MODELS[judging_provider].keys()),
        index=0,  # Default to first model
        key="judging_model_select",
    )
    judging_api_key = st.text_input(
        "Judging API Key (Optional)",
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
            "Select Judging Prompt Template",
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
**Structuring Model:** `{structuring_provider}` - `{structuring_model}` | **Structuring Prompt:** `{selected_structuring_template_path}`

**Judging Model:** `{judging_provider}` - `{judging_model}` | **Judging Prompt:** `{selected_judging_template_path}`
""")


# Separator moved outside the expander, before the next section
st.markdown("---")


# --- Function to load and process results ---
@st.cache_data(show_spinner=False)
def load_and_process_results(absolute_results_paths):
    """Loads data from _final_results.json files (given absolute paths), processes into a DataFrame, and aggregates summary statistics."""
    logger.info("--- Entering load_and_process_results ---")  # Log entry
    logger.info(
        f"Attempting to load and process results from: {absolute_results_paths}"
    )
    all_results_data = []
    aggregated_summary = {
        "total_evaluations_processed": 0,
        "total_evaluation_time_seconds": 0.0,
        "total_structuring_api_calls": 0,
        "total_judging_api_calls": 0,
        "total_tasks_processed": 0,  # Initialize missing key
        "average_time_per_model_seconds": {},  # Initialize missing key
    }
    processed_files_count = 0
    failed_files = []
    for file_path_str in absolute_results_paths:
        try:
            file_path = Path(file_path_str)  # Convert string path to Path object
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                file_summary = data.get("summary", {})

                # Aggregate summary stats safely
                # Use actual keys from JSON summary
                aggregated_summary["total_evaluations_processed"] += file_summary.get(
                    "total_evaluations_processed", 0
                )
                aggregated_summary["total_evaluation_time_seconds"] += file_summary.get(
                    "total_evaluation_time_seconds", 0.0
                )
                aggregated_summary["total_structuring_api_calls"] += file_summary.get(
                    "total_structuring_api_calls", 0
                )
                aggregated_summary["total_judging_api_calls"] += file_summary.get(
                    "total_judging_api_calls", 0
                )
                # Aggregate total_tasks_processed
                aggregated_summary["total_tasks_processed"] += file_summary.get(
                    "total_tasks_processed", 0
                )
                # Aggregate average_time_per_model_seconds (simple merge, assumes no overlapping models across files)
                # A more robust approach might average times if models appear in multiple files,
                # but for now, a simple update/merge should work if files represent distinct batches/runs.
                per_model_times = file_summary.get("average_time_per_model_seconds", {})
                if isinstance(per_model_times, dict):
                    aggregated_summary["average_time_per_model_seconds"].update(
                        per_model_times
                    )

                # --- Process 'results' list ---
                results_list = data.get("results", [])
                if not isinstance(results_list, list):
                    logger.error(
                        f"JSON file {file_path_str} does not contain a valid 'results' list."
                    )
                    failed_files.append(file_path_str)
                    # Continue processing summary even if results are bad, but don't process tasks
                else:
                    # Process tasks only if results_list is valid
                    task_count = 0
                    for task in results_list:
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
                                                    ] = rubric_details.get(
                                                        "justification"
                                                    )
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
            # Continue processing other files
        except json.JSONDecodeError:
            st.error(f"Error decoding JSON from file: {file_path_str}")
            logger.error(f"Error decoding JSON from file: {file_path_str}")  # Added log
            failed_files.append(file_path_str)
            # Continue processing other files
        except Exception as e:
            st.error(f"Error processing file {file_path_str}: {e}")
            logger.error(f"Error processing file {file_path_str}: {e}")  # Added log
            failed_files.append(file_path_str)
            # Continue processing other files
        else:
            # This else block corresponds to the try block starting at line 375
            # It executes only if no exceptions occurred during file reading/parsing
            processed_files_count += 1  # Increment count on success

    # Log summary before returning
    if failed_files:
        logger.error(
            f"Failed to process {len(failed_files)} files: {', '.join(failed_files)}"
        )
    logger.info(
        f"Successfully processed {processed_files_count} result files out of {len(absolute_results_paths)}."
    )
    logger.info(
        f"--- Exiting load_and_process_results. Returning DataFrame (shape: {pd.DataFrame(all_results_data).shape}) and Summary: {aggregated_summary} ---"
    )  # Log exit and return values
    # Return both the DataFrame and the aggregated summary
    return pd.DataFrame(all_results_data), aggregated_summary


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
        logger.info(f"Regenerate Graphs button clicked for folders: {selected_folders}")
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
            # Store the absolute paths in session state *before* loading
            st.session_state.evaluation_results_paths = evaluation_results_paths
            logger.info(
                f"Stored absolute evaluation paths in session state: {st.session_state.evaluation_results_paths}"
            )
            # Clear potentially stale raw data from previous views
            if "raw_results_data" in st.session_state:
                del st.session_state.raw_results_data
                logger.info("Cleared stale raw_results_data from session state.")

            # Load results AND summary, store both in session state using the absolute paths
            st.session_state.results_df, st.session_state.summary_data = (
                load_and_process_results(
                    st.session_state.evaluation_results_paths  # Use paths from session state
                )
            )
            st.success("Graphs regenerated successfully!")
            logger.info("Graphs regenerated successfully.")
            # st.rerun() # Removed to allow session state to persist for display
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
    progress_area = st.container()  # Removed header
    stop_button = st.button("🛑 Stop Processing", type="primary")
    if stop_button:
        st.session_state.stop_requested = True
    log_expander = st.expander("Show Full Logs", expanded=False)
    log_placeholder = log_expander.empty()

    with st.spinner("Evaluation Progress..."):  # Updated spinner text
        # Removed progress bar
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
                    # Removed progress bar update
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
                # Always store the absolute path
                try:
                    path_obj = Path(raw_path)
                    # Resolve to absolute path if it's relative (assuming relative to COGNIBENCH_ROOT)
                    if not path_obj.is_absolute():
                        abs_path_str = str((COGNIBENCH_ROOT / path_obj).resolve())
                        logger.info(
                            f"Resolved relative path '{raw_path}' to absolute: '{abs_path_str}'"
                        )
                    else:
                        abs_path_str = str(
                            path_obj.resolve()
                        )  # Ensure it's fully resolved even if absolute
                        logger.info(
                            f"Path '{raw_path}' is already absolute: '{abs_path_str}'"
                        )

                    # Ensure we don't add duplicates
                    if abs_path_str not in st.session_state.evaluation_results_paths:
                        logger.info(
                            f"Appending absolute path to list: '{abs_path_str}'"
                        )
                        st.session_state.evaluation_results_paths.append(abs_path_str)
                        # Display the absolute path in the success message
                        progress_area.success(
                            f"Found and stored results file path: {abs_path_str}"
                        )
                    else:
                        logger.info(
                            f"Path '{abs_path_str}' already in list, skipping append."
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
                # Load results AND summary
                results_df, summary_data = load_and_process_results(
                    absolute_paths  # Use the constructed absolute paths
                )
                st.session_state.results_df = results_df  # Store loaded df
                st.session_state.summary_data = summary_data  # Store summary data
                # Add logging here to check if it was loaded before the rerun
                logger.info(
                    f"After evaluation run, results_df is None: {st.session_state.get('results_df') is None}"
                )
                if st.session_state.get("results_df") is not None:
                    logger.info(
                        f"Loaded DataFrame shape after run: {st.session_state.results_df.shape}"
                    )
                logger.info(
                    f"After evaluation run, summary_data: {st.session_state.get('summary_data')}"
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

            # API calls are now retrieved from summary_data
            structuring_calls = st.session_state.get("summary_data", {}).get(
                "total_structuring_api_calls", "N/A"
            )
            judging_calls = st.session_state.get("summary_data", {}).get(
                "total_judging_api_calls", "N/A"
            )
            # Calculate averages using summary_data for consistency
            avg_time_per_task_str = "N/A"
            avg_time_per_model_eval_str = "N/A"
            summary_data = st.session_state.get("summary_data", {})
            total_evals_from_summary = summary_data.get("total_evaluations")
            total_duration_from_summary = summary_data.get("total_duration_seconds")

            # Need num_unique_tasks from the dataframe if available
            num_unique_tasks = 0
            if (
                st.session_state.get("results_df") is not None
                and not st.session_state.results_df.empty
            ):
                results_df = st.session_state.results_df
                num_unique_tasks = results_df["task_id"].nunique()

            # Calculate averages if relevant data from summary is valid
            if (
                isinstance(total_duration_from_summary, (int, float))
                and total_duration_from_summary > 0
            ):
                if num_unique_tasks > 0:
                    avg_time_per_task = total_duration_from_summary / num_unique_tasks
                    avg_time_per_task_str = (
                        f"{avg_time_per_task:.2f}s"  # Use 's' suffix
                    )
                if (
                    isinstance(total_evals_from_summary, int)
                    and total_evals_from_summary > 0
                ):
                    avg_time_per_model_eval = (
                        total_duration_from_summary / total_evals_from_summary
                    )
                    avg_time_per_model_eval_str = (
                        f"{avg_time_per_model_eval:.2f}s"  # Use 's' suffix
                    )
            else:
                # Use summary data if df is empty/None but summary exists
                num_evaluations = st.session_state.get("summary_data", {}).get(
                    "total_evaluations", 0
                )

            # --- Display New Summary Metrics ---
            progress_area.success("Evaluation Run Summary:")
            summary_data = st.session_state.get("summary_data", {})

            # Use wall-clock duration calculated earlier
            total_duration_display = st.session_state.get("eval_duration_str", "N/A")

            # Get counts from summary_data with defaults, using correct keys
            total_tasks_display = summary_data.get(
                "total_tasks_processed", "N/A"
            )  # Use correct key for tasks
            total_evals_display = summary_data.get(
                "total_evaluations_processed", "N/A"
            )  # Fetch evaluations count
            structuring_calls_display = summary_data.get(
                "total_structuring_api_calls", "N/A"
            )
            judging_calls_display = summary_data.get("total_judging_api_calls", "N/A")

            # Row 1
            col1, col2 = progress_area.columns(2)
            col1.metric(
                "Total Tasks Processed", total_tasks_display
            )  # Updated label and variable
            col2.metric("Total Evaluation Duration", total_duration_display)

            # Row 2
            col3, col4, col5 = progress_area.columns(3)
            # Keep 3 columns, but update the first metric
            col3.metric(
                "Total Evaluations Completed", total_evals_display
            )  # Use evaluations count and new label
            col4.metric("Total Structuring Calls", structuring_calls_display)
            col5.metric("Total Judging Calls", judging_calls_display)

            # --- Display Averages in Expander ---
            with progress_area.expander("Average Statistics"):
                # Prepare per-model average time string
                avg_time_per_model_dict = summary_data.get(
                    "average_time_per_model_seconds", {}
                )
                per_model_avg_lines = [
                    f"  - `{model}`: {time:.2f}s"
                    for model, time in avg_time_per_model_dict.items()
                ]
                per_model_avg_str = (
                    "\n".join(per_model_avg_lines) if per_model_avg_lines else "  - N/A"
                )

                st.markdown(f"""
                - **Avg. Time per Task:** {avg_time_per_task_str}
                - **Avg. Time per Model Evaluation (Overall):** {avg_time_per_model_eval_str}
                - **Avg. Time per Model Evaluation (Breakdown):**
{per_model_avg_str}
                """)

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
        # Load results and summary, store both
        st.session_state.results_df, st.session_state.summary_data = (
            load_and_process_results(st.session_state.evaluation_results_paths)
        )
        logger.info(f"--- After calling load_and_process_results ---")
        logger.info(
            f"st.session_state.results_df is None: {st.session_state.results_df is None}"
        )
        if st.session_state.results_df is not None:
            logger.info(
                f"st.session_state.results_df shape: {st.session_state.results_df.shape}"
            )
        logger.info(f"st.session_state.summary_data: {st.session_state.summary_data}")

# --- Display Results if DataFrame is loaded ---
logger.info(
    f"Checking session state before graph display. results_df is None: {st.session_state.get('results_df') is None}"
)
# --- Load Raw Results Data (for JSON viewers) ---
# Attempt to load raw data if paths exist and it's not already loaded
if (
    st.session_state.get("evaluation_results_paths")
    and "raw_results_data" not in st.session_state
):
    raw_data_load_success = False
    results_file_abs_path = None  # Initialize path variable

    # Check if the paths list exists and is not empty
    if st.session_state.get("evaluation_results_paths"):
        # Use the absolute path string directly from session state
        abs_path_str = st.session_state.evaluation_results_paths[
            0
        ]  # Get the stored absolute path string
        results_file_abs_path = Path(abs_path_str)  # Convert string to Path object
        logger.info(
            f"Attempting to load raw results data from: {results_file_abs_path}"
        )
    else:
        logger.warning(
            "evaluation_results_paths is empty or missing in session state, cannot load raw data."
        )

    # Proceed only if we have a valid path
    if results_file_abs_path:
        try:  # Correctly indented try block
            if results_file_abs_path.is_file():
                with open(results_file_abs_path, "r", encoding="utf-8") as f:
                    # Store the entire parsed JSON, including summary and results list
                    st.session_state.raw_results_data = json.load(f)
                    # Basic validation
                    if isinstance(
                        st.session_state.raw_results_data.get("results"), list
                    ):
                        logger.info(
                            f"Successfully loaded raw results data from {results_file_abs_path.name}"
                        )
                        raw_data_load_success = True
                    else:
                        logger.error(
                            f"'results' key not found or not a list in {results_file_abs_path}"
                        )
                        st.warning(
                            f"Could not find a valid 'results' list in {results_file_abs_path.name}. Detailed JSON views may be affected."
                        )
                        # Keep potentially valid summary data if results list is bad
                        if "results" in st.session_state.raw_results_data:
                            del st.session_state.raw_results_data[
                                "results"
                            ]  # Remove invalid results
            else:
                logger.error(f"Raw results file not found at {results_file_abs_path}")
                st.warning(
                    f"Could not find the results file ({results_file_abs_path.name}) needed for detailed JSON views."
                )

        except json.JSONDecodeError:
            logger.exception(
                f"JSONDecodeError reading raw results from {results_file_abs_path}"
            )
            st.error(
                f"Error decoding JSON from {results_file_abs_path.name}. Detailed JSON views may be unavailable."
            )
        except Exception as e:
            logger.exception(
                f"Error loading raw results data from {results_file_abs_path}: {e}"
            )
            st.error(
                f"An error occurred loading raw results data: {e}. Detailed JSON views may be unavailable."
            )

        # Clear the session state variable if loading failed completely
        if not raw_data_load_success and "raw_results_data" in st.session_state:
            del st.session_state.raw_results_data
    # Removed duplicated block from previous failed diff

if st.session_state.get("results_df") is not None:
    logger.info(
        "Results DataFrame found in session state. Proceeding with visualization."
    )
    df = st.session_state.results_df
    st.success(f"Loaded {len(df)} evaluation results.")
    # --- Display Summary Metrics (Loaded Data) ---
    summary_data = st.session_state.get("summary_data", {})
    logger.info(f"Summary data from session state: {summary_data}")  # Add logging
    # Use correct key from JSON
    total_duration_s = summary_data.get("total_evaluation_time_seconds", 0.0)
    total_duration_display = f"{total_duration_s:.1f}s" if total_duration_s else "N/A"
    # Use correct key from JSON
    total_tasks_display = summary_data.get(
        "total_tasks_processed", "N/A"
    )  # Use correct key for tasks
    total_evals_display = summary_data.get(
        "total_evaluations_processed", "N/A"
    )  # Fetch evaluations count
    structuring_calls_display = summary_data.get("total_structuring_api_calls", "N/A")
    judging_calls_display = summary_data.get("total_judging_api_calls", "N/A")

    st.subheader("Evaluation Summary")
    # Row 1
    col1, col2 = st.columns(2)
    col1.metric(
        "Total Tasks Processed", total_tasks_display
    )  # Updated label and variable
    col2.metric("Total Evaluation Duration", total_duration_display)

    # Row 2
    col3, col4, col5 = st.columns(3)
    col3.metric(
        "Total Evaluations Processed", total_evals_display
    )  # Use evaluations count and new label
    col4.metric("Total Structuring Calls", structuring_calls_display)
    col5.metric("Total Judging Calls", judging_calls_display)

    # --- Display Averages in Expander (Loaded Data) ---
    # Calculate averages using summary_data and df
    avg_time_per_task_str = "N/A"
    avg_time_per_model_eval_str = "N/A"
    # Get total evaluations from summary data
    total_evals_from_summary = summary_data.get(
        "total_evaluations_processed"
    )  # Use correct key

    # Get unique tasks count from the dataframe
    num_unique_tasks = 0
    if not df.empty:
        num_unique_tasks = df["task_id"].nunique()

    # Calculate averages if relevant data is valid
    if isinstance(total_duration_s, (int, float)) and total_duration_s > 0:
        if num_unique_tasks > 0:
            avg_time_per_task = total_duration_s / num_unique_tasks
            avg_time_per_task_str = f"{avg_time_per_task:.2f}s"  # Use 's' suffix
        # Use total_evals_from_summary for per-model-eval average
        if isinstance(total_evals_from_summary, int) and total_evals_from_summary > 0:
            avg_time_per_model_eval = total_duration_s / total_evals_from_summary
            avg_time_per_model_eval_str = (
                f"{avg_time_per_model_eval:.2f}s"  # Use 's' suffix
            )

    with st.expander("Average Statistics"):
        # Prepare per-model average time string
        avg_time_per_model_dict = summary_data.get("average_time_per_model_seconds", {})
        logger.info(f"Per-model avg dict: {avg_time_per_model_dict}")  # Add logging
        per_model_avg_lines = [
            f"  - `{model}`: {time:.2f}s"
            for model, time in avg_time_per_model_dict.items()
        ]
        per_model_avg_str = (
            "\n".join(per_model_avg_lines) if per_model_avg_lines else "  - N/A"
        )

        st.markdown(
            f"""
- **Avg. Time per Task:** {avg_time_per_task_str}
- **Avg. Time per Model Evaluation (Overall):** {avg_time_per_model_eval_str}
- **Avg. Time per Model Evaluation (Breakdown):**
{per_model_avg_str}
"""
        )

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
                    # Keep using review_needed_df to populate the selectbox of tasks needing review
                    selected_review_task_id = st.selectbox(
                        "Select Task ID for Review:",
                        options=review_needed_df["task_id"].unique(),
                        key="human_review_task_select",  # Add key
                    )

                    # Get the model_id associated with this flagged review from the filtered df
                    review_model_id = review_needed_df.loc[
                        review_needed_df["task_id"] == selected_review_task_id,
                        "model_id",
                    ].iloc[0]

                    st.write(
                        f"Reviewing Task **{selected_review_task_id}** for Model **{review_model_id}**"
                    )

                    # --- Load and Display Raw JSON for the selected task/model ---
                    raw_task_dict = None
                    target_evaluation = None
                    parsed_rubric_scores = {}  # Initialize

                    if "raw_results_data" in st.session_state and isinstance(
                        st.session_state.raw_results_data.get("results"), list
                    ):
                        # Find the raw task dictionary
                        raw_task_dict = next(
                            (
                                task
                                for task in st.session_state.raw_results_data["results"]
                                if task.get("task_id", task.get("id"))
                                == selected_review_task_id
                            ),
                            None,
                        )

                        if raw_task_dict:
                            # Find the specific evaluation within this task matching the model_id
                            target_evaluation = next(
                                (
                                    ev
                                    for ev in raw_task_dict.get("evaluations", [])
                                    if ev.get("model_id") == review_model_id
                                ),
                                None,
                            )

                            if target_evaluation:
                                st.write(
                                    "#### Review Task Details (Raw JSON)"
                                )  # Changed heading size
                                st.json(
                                    target_evaluation, expanded=False
                                )  # Display the specific evaluation JSON

                                # Extract rubric scores from the target evaluation's judge_evaluation
                                judge_eval = target_evaluation.get(
                                    "judge_evaluation", {}
                                )
                                if isinstance(judge_eval, dict):
                                    parsed_rubric_scores = judge_eval.get(
                                        "parsed_rubric_scores", {}
                                    )
                                    if not isinstance(parsed_rubric_scores, dict):
                                        st.warning(
                                            "Parsed rubric scores format is unexpected."
                                        )
                                        logger.warning(
                                            f"Unexpected format for parsed_rubric_scores in task {selected_review_task_id}, model {review_model_id}"
                                        )
                                        parsed_rubric_scores = {}  # Reset to empty dict
                            else:
                                st.error(
                                    f"Could not find evaluation details for model '{review_model_id}' within task '{selected_review_task_id}' in the raw data."
                                )
                                logger.error(
                                    f"Evaluation for model {review_model_id} not found in raw data for task {selected_review_task_id}"
                                )
                        else:
                            st.error(
                                f"Could not find raw task details for task ID: {selected_review_task_id}"
                            )
                            logger.error(
                                f"Raw task details not found for task ID: {selected_review_task_id}"
                            )
                    else:
                        st.warning(
                            "Raw results data not available. Cannot display detailed JSON or populate rubric scores accurately."
                        )
                        logger.warning(
                            "st.session_state.raw_results_data not found or invalid for human review section."
                        )

                    # --- Rubric Score Correction Inputs ---
                    # Use the extracted parsed_rubric_scores dictionary
                    st.write("#### Correct Rubric Scores")
                    corrected_scores = {}
                    if parsed_rubric_scores:  # Check if we have scores to display
                        for rubric_name, score_details in parsed_rubric_scores.items():
                            options = ["Yes", "Partial", "No", "N/A"]
                            current_value = score_details.get(
                                "score", "N/A"
                            )  # Default to N/A if score key missing

                            # Handle potential NaN or unexpected values robustly
                            if pd.notna(current_value) and current_value in options:
                                try:
                                    default_index = options.index(current_value)
                                except ValueError:
                                    logger.warning(
                                        f"Rubric score '{current_value}' for '{rubric_name}' not in options {options}. Defaulting to N/A."
                                    )
                                    default_index = options.index("N/A")
                            else:
                                default_index = options.index(
                                    "N/A"
                                )  # Default to N/A index if NaN or invalid

                            # Use rubric_name for the key and label
                            corrected_scores[rubric_name] = st.selectbox(
                                f"Corrected Score for {rubric_name.replace('_', ' ').title()}:",
                                options,
                                index=default_index,
                                key=f"review_rubric_{selected_review_task_id}_{review_model_id}_{rubric_name}",  # Unique key
                            )
                    else:
                        st.info(
                            "No rubric scores found in the judge evaluation for this task/model."
                        )

                    review_comments = st.text_area(
                        "Review Comments:",
                        key=f"review_comments_{selected_review_task_id}_{review_model_id}",
                    )

                    if st.button(
                        "Save Human Review",
                        key=f"save_review_{selected_review_task_id}_{review_model_id}",
                    ):
                        logger.info(
                            f"Attempting to save human review for task {selected_review_task_id}, model {review_model_id}."
                        )
                        # --- Logic to update the original data structure (or save elsewhere) ---
                        # This part is complex as it requires modifying the potentially large raw_results_data
                        # or saving the review separately. For now, just log and show success.
                        # TODO: Implement actual saving mechanism (e.g., update file, save to DB)
                        st.success(
                            f"Human review recorded for Task {selected_review_task_id}, Model {review_model_id} (Saving mechanism TBD)."
                        )
                        logger.info(
                            f"Human review data captured: Scores={corrected_scores}, Comments='{review_comments}'"
                        )
                        # Optionally, update the status in the main DataFrame (df) to reflect review completion
                        # This requires finding the correct row index in 'df'
                        try:
                            row_index = df[
                                (df["task_id"] == selected_review_task_id)
                                & (df["model_id"] == review_model_id)
                            ].index
                            if not row_index.empty:
                                # Update the status column used for filtering/display (e.g., 'judge_human_review_status')
                                # Make sure the column name matches exactly what's used earlier
                                status_col_name = "judge_human_review_status"  # Or derive dynamically if needed
                                if status_col_name in df.columns:
                                    df.loc[row_index, status_col_name] = (
                                        "Reviewed"  # Or another appropriate status
                                    )
                                    logger.info(
                                        f"Updated status in DataFrame for task {selected_review_task_id}, model {review_model_id}"
                                    )
                                    # Maybe trigger a rerun to refresh the 'Tasks Flagged for Human Review' table
                                    # st.rerun() # Consider implications of rerun here
                                else:
                                    logger.warning(
                                        f"Status column '{status_col_name}' not found in DataFrame, cannot update."
                                    )

                        except Exception as update_e:
                            logger.error(
                                f"Error updating DataFrame status after review: {update_e}"
                            )

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

        # --- Task-Level Explorer (Loads raw JSON for selected task) ---
        st.subheader("Detailed Task-Level Explorer")
        logger.info(
            "Attempting to display 'Detailed Task-Level Explorer' with raw JSON."
        )

        raw_task_data = None
        task_ids = []
        selected_task_details = None

        # Check if results paths are available in session state
        if st.session_state.get("evaluation_results_paths"):
            # For now, assume the first file contains the relevant data (as per instruction #5)
            # Paths stored might be relative, so construct absolute path
            results_file_rel_path_str = st.session_state.evaluation_results_paths[0]
            results_file_abs_path = COGNIBENCH_ROOT / results_file_rel_path_str
            logger.info(
                f"Attempting to load raw task data from: {results_file_abs_path}"
            )

            try:
                if results_file_abs_path.is_file():
                    with open(results_file_abs_path, "r", encoding="utf-8") as f:
                        raw_data = json.load(f)
                    # Expecting a list under the 'results' key
                    if isinstance(raw_data.get("results"), list):
                        raw_task_data = raw_data["results"]
                        # Extract task IDs (handle potential 'id' key as fallback, filter None)
                        task_ids = sorted(
                            list(
                                filter(
                                    None,
                                    [
                                        task.get("task_id", task.get("id"))
                                        for task in raw_task_data
                                    ],
                                )
                            )
                        )
                        if not task_ids:
                            st.warning(
                                f"No tasks with 'task_id' or 'id' found in the 'results' list of {results_file_abs_path.name}."
                            )
                            logger.warning(
                                f"No task IDs found in {results_file_abs_path}"
                            )
                        else:
                            logger.info(
                                f"Successfully loaded {len(task_ids)} task IDs from {results_file_abs_path.name}."
                            )
                    else:
                        st.error(
                            f"Could not find a 'results' list in {results_file_abs_path.name}."
                        )
                        logger.error(
                            f"'results' key not found or not a list in {results_file_abs_path}"
                        )
                else:
                    st.error(f"Results file not found: {results_file_abs_path}")
                    logger.error(f"Results file not found at {results_file_abs_path}")

            except json.JSONDecodeError:
                st.error(f"Error decoding JSON from file: {results_file_abs_path.name}")
                logger.exception(f"JSONDecodeError reading {results_file_abs_path}")
            except Exception as e:
                st.error(f"An error occurred while loading task details: {e}")
                logger.exception(
                    f"Error loading raw task details from {results_file_abs_path}: {e}"
                )

        else:
            # This case might occur if results_df exists but paths were cleared somehow, or if regenerating graphs failed.
            if st.session_state.get("results_df") is not None:
                st.warning(
                    "Results data frame is loaded, but the original results file path is missing. Cannot display detailed task JSON."
                )
                logger.warning(
                    "evaluation_results_paths is missing in session state, cannot load raw JSON."
                )
            # If results_df is also None, the main conditional block handles the message.

        # Display selectbox and JSON only if task IDs were successfully loaded
        if task_ids and raw_task_data:
            selected_task_id = st.selectbox(
                "Select Task ID for Detailed View:",
                options=task_ids,
                key="detailed_task_id_select",  # Add a key for stability
            )

            if selected_task_id:
                # Find the full dictionary for the selected task_id
                selected_task_details = next(
                    (
                        task
                        for task in raw_task_data
                        if task.get("task_id", task.get("id")) == selected_task_id
                    ),
                    None,  # Default to None if not found
                )

                if selected_task_details:
                    st.write(f"**Full JSON for Task:** `{selected_task_id}`")
                    st.json(selected_task_details, expanded=False)  # Start collapsed
                    logger.info(
                        f"Displayed JSON for selected task ID: {selected_task_id}"
                    )
                else:
                    # This case should ideally not happen if selected_task_id comes from task_ids
                    st.error(
                        f"Could not find details for selected task ID: {selected_task_id}"
                    )
                    logger.error(
                        f"Task ID {selected_task_id} selected, but details not found in raw_task_data."
                    )
        elif st.session_state.get("results_df") is not None:
            # Only show this info if we have loaded results but couldn't get raw data
            st.info(
                "Detailed JSON view requires the original results file to be accessible."
            )

        # --- Filtered Results Table (Keep this below the JSON viewer) ---
        st.subheader("Filtered Results Table")  # Add subheader for clarity
        display_cols = {
            "task_id": "Task ID",
            "model_id": "Model",
            "subject": "Subject",
            "complexity": "Complexity",
            "aggregated_score": "Aggregated Score",
            "human_preference": "Human Preference",
            "human_rating": "Human Rating",
            # Remove prompt/response from table view as they are large and shown in JSON
            # "prompt": "Prompt",
            # "ideal_response": "Ideal Response",
            # "model_response": "Model Response",
        }
        # Add rubric scores and justification if they exist in the filtered_df
        if not filtered_df.empty:  # Check if filtered_df exists and is not empty
            # Dynamically add judge rubric scores and justification if present
            judge_cols = [
                col for col in filtered_df.columns if col.startswith("judge_")
            ]
            for col in judge_cols:
                # Simple title case formatting for display name
                display_name = col.replace("judge_", "").replace("_", " ").title()
                # Add prefix to avoid potential name collisions with original columns
                display_cols[col] = f"Judge: {display_name}"

            cols_to_show = [
                col for col in display_cols.keys() if col in filtered_df.columns
            ]
            st.dataframe(filtered_df[cols_to_show].rename(columns=display_cols))
            logger.info("Displayed filtered results DataFrame.")
        else:
            # This handles the case where filtered_df might be empty due to filters
            # The main conditional `if st.session_state.results_df is not None:` handles if df wasn't loaded at all.
            if (
                st.session_state.get("results_df") is not None
            ):  # Check if the base df exists
                logger.info("Filtered DataFrame is empty, skipping table display.")
                # No need for a specific message here as the filter warning above covers it.

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
