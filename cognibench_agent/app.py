"""
CogniBench Streamlit Application.

Provides a web-based user interface for running CogniBench evaluations,
configuring models and prompts, uploading data, viewing results, and
managing the evaluation process.
"""

import json
import logging
import os  # Added for directory sorting
import queue
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Add project root to sys.path ---
# COGNIBENCH_ROOT is now defined in core.constants, but needed for sys.path here
_APP_DIR_TEMP = Path(__file__).parent
_COGNIBENCH_ROOT_TEMP = _APP_DIR_TEMP.parent
if str(_COGNIBENCH_ROOT_TEMP) not in sys.path:
    sys.path.insert(0, str(_COGNIBENCH_ROOT_TEMP))
# --- End sys.path modification ---

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml
from core.config import AppConfig

# Import constants from the new core location
from core.constants import COGNIBENCH_ROOT  # Import the official one
from core.constants import (  # Import if needed, though defined locally above for sys.path; Keep if used directly, otherwise remove
    APP_DIR,
    AVAILABLE_MODELS,
    BASE_CONFIG_PATH,
    COLOR_MAP,
    DATA_DIR,
    DEFAULT_JUDGING_MODEL,
    DEFAULT_STRUCTURING_MODEL,
    JUDGING_TEMPLATES_DIR,
    PROMPT_TEMPLATES_DIR_ABS,
    STRUCTURING_TEMPLATES_DIR,
)
from core.evaluation_runner import run_batch_evaluation_core
from core.llm_clients.openai_client import clear_openai_cache
from core.log_setup import setup_logging

# --- Constants ---
# Definitions moved to core/constants.py

# --- Logging Setup ---
logger = logging.getLogger("frontend")  # Changed from "streamlit"
if "logging_setup_complete" not in st.session_state:
    # Ensure logging is set to DEBUG level to capture detailed logs
    setup_logging(log_level=logging.DEBUG)
    st.session_state.logging_setup_complete = True
    logger = logging.getLogger(
        "frontend"
    )  # Re-get logger after setup, changed from "streamlit"
    # Force level just in case setup didn't stick or was overridden
    logger.setLevel(logging.DEBUG)
    logger.info("Logger 'frontend' level forced to DEBUG.")  # Add confirmation log
    logger.info("Initial logging setup complete.")
    logger.info("Streamlit app started.")

# --- Page Config ---
st.set_page_config(layout="wide", page_title="CogniBench Runner")

# --- Helper Functions ---


def get_templates(directory: Path) -> Dict[str, str]:
    """Scans a directory for .txt files and returns a dictionary of name: path."""
    try:
        return {
            f.stem: str(f)  # Use stem for cleaner names
            for f in directory.iterdir()
            if f.is_file() and f.suffix == ".txt"
        }
    except FileNotFoundError:
        st.error(f"Prompt templates directory not found: {directory}")
        logger.error(f"Prompt templates directory not found: {directory}")
        return {}


AVAILABLE_STRUCTURING_TEMPLATES = get_templates(STRUCTURING_TEMPLATES_DIR)
AVAILABLE_JUDGING_TEMPLATES = get_templates(JUDGING_TEMPLATES_DIR)


def initialize_session_state():
    """Initializes Streamlit session state variables if they don't exist."""
    defaults = {
        "temp_dir": None,
        "temp_dir_path": None,
        "uploaded_files_info": [],  # Store name and path
        "last_uploaded_files_key": None,
        "structuring_provider_select": list(AVAILABLE_MODELS.keys())[0],
        # "structuring_model_select": list( # Default set below
        #     AVAILABLE_MODELS[list(AVAILABLE_MODELS.keys())[0]].keys()
        # )[0],
        "structuring_api_key_input": "",
        "judging_provider_select": list(AVAILABLE_MODELS.keys())[0],
        # "judging_model_select": list( # Default set below
        #     AVAILABLE_MODELS[list(AVAILABLE_MODELS.keys())[0]].keys()
        # )[0],
        "judging_api_key_input": "",
        "structuring_template_select": list(AVAILABLE_STRUCTURING_TEMPLATES.keys())[0]
        if AVAILABLE_STRUCTURING_TEMPLATES
        else None,
        "judging_template_select": list(AVAILABLE_JUDGING_TEMPLATES.keys())[0]
        if AVAILABLE_JUDGING_TEMPLATES
        else None,
        "show_structuring": False,
        "show_judging": False,
        "show_config": False,
        "evaluation_running": False,
        "evaluation_results_paths": [],  # Store paths to _final_results.json
        "last_run_output": [],  # Store console output/logs from runner
        "results_df": None,
        "aggregated_summary": None,
        "eval_start_time": None,
        "eval_duration_str": None,
        "worker_thread": None,
        "output_queue": queue.Queue(),
        "stop_event": threading.Event(),
        "selected_results_folders": [],
        "config_complete": False,
        "evaluation_error": None,
        # "action_mode": "Run Evaluation", # Removed - Tabs control the view now
        "newly_completed_run_folder": None,  # Track the folder from the last completed run
        # "switch_to_view_results": False, # Removed - No longer needed with tabs
        # Add state for selected task/model in view results
        "view_selected_task_id": None,
        "view_selected_model_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Set default models if not already set
    if "structuring_model_select" not in st.session_state:
        st.session_state["structuring_model_select"] = DEFAULT_STRUCTURING_MODEL
        logger.info(
            f"Initialized structuring_model_select to {DEFAULT_STRUCTURING_MODEL}"
        )
    if "judging_model_select" not in st.session_state:
        st.session_state["judging_model_select"] = DEFAULT_JUDGING_MODEL
        logger.info(f"Initialized judging_model_select to {DEFAULT_JUDGING_MODEL}")

    # Special handling for temp dir
    if st.session_state.temp_dir is None:
        logger.info("Initializing temporary directory for session state.")
        st.session_state.temp_dir = tempfile.TemporaryDirectory()
        st.session_state.temp_dir_path = Path(st.session_state.temp_dir.name)


# --- UI Rendering Functions ---


def render_file_uploader():
    """Renders the file uploader and saves uploaded files to temp dir."""
    st.header("Upload Raw RLHF JSON Data file(s)")
    uploaded_files = st.file_uploader(
        "Select CogniBench JSON batch file(s)",
        type=["json"],
        accept_multiple_files=True,
        help="Upload one or more JSON files containing tasks for evaluation.",
    )

    if uploaded_files:
        uploaded_file_names = [f.name for f in uploaded_files]
        current_upload_key = tuple(sorted(uploaded_file_names))

        # Check if files changed or session state needs update
        if (
            st.session_state.last_uploaded_files_key != current_upload_key
            or not st.session_state.uploaded_files_info
        ):
            logger.info(f"Processing {len(uploaded_files)} uploaded files...")
            st.session_state.uploaded_files_info = []
            temp_dir = st.session_state.temp_dir_path
            for uploaded_file in uploaded_files:
                try:
                    dest_path = temp_dir / uploaded_file.name
                    with open(dest_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    st.session_state.uploaded_files_info.append(
                        {"name": uploaded_file.name, "path": str(dest_path)}
                    )
                    logger.info(f"Saved uploaded file to temporary path: {dest_path}")
                except Exception as e:
                    st.error(f"Error saving file {uploaded_file.name}: {e}")
                    logger.error(f"Error saving file {uploaded_file.name}: {e}")

            st.session_state.last_uploaded_files_key = current_upload_key
            logger.info(
                f"Finished processing uploads. {len(st.session_state.uploaded_files_info)} files ready."
            )
            # Rerun to update UI immediately after processing uploads
            st.rerun()

        st.write(f"Using {len(st.session_state.uploaded_files_info)} uploaded file(s):")
        for file_info in st.session_state.uploaded_files_info:
            st.write(f"- {file_info['name']}")

    else:
        st.info("Please upload at least one batch file.")
        # Clear saved file info if no files are uploaded
        if st.session_state.uploaded_files_info:
            st.session_state.uploaded_files_info = []
            st.session_state.last_uploaded_files_key = None


def render_config_ui():
    """Renders the configuration widgets for models and prompts."""
    st.header("Configure Models and Prompts")

    with st.expander("Model Configurations", expanded=False):
        col_structuring, col_judging = st.columns(2)

        with col_structuring:
            st.subheader("Structuring Model")
            st.selectbox(
                "Provider",
                options=list(AVAILABLE_MODELS.keys()),
                key="structuring_provider_select",
            )
            st.selectbox(
                "Model",
                options=list(
                    AVAILABLE_MODELS[
                        st.session_state.structuring_provider_select
                    ].keys()
                ),
                # index=5, # Removed: Default set via session state
                key="structuring_model_select",
            )
            st.text_input(
                "API Key (Optional)",
                type="password",
                placeholder="Leave blank to use environment variable",
                key="structuring_api_key_input",
            )

        with col_judging:
            st.subheader("Judging Model")
            st.selectbox(
                "Provider",
                options=list(AVAILABLE_MODELS.keys()),
                key="judging_provider_select",
            )
            st.selectbox(
                "Model",
                options=list(
                    AVAILABLE_MODELS[st.session_state.judging_provider_select].keys()
                ),
                # index=5, # Removed: Default set via session state
                key="judging_model_select",
            )
            st.text_input(
                "API Key (Optional)",
                type="password",
                placeholder="Leave blank to use environment variable",
                key="judging_api_key_input",
            )

    with st.expander("Prompt Configurations", expanded=False):
        col_prompt1, col_prompt2 = st.columns(2)

        with col_prompt1:
            st.selectbox(
                "Structuring Prompt Template",
                options=list(AVAILABLE_STRUCTURING_TEMPLATES.keys()),
                key="structuring_template_select",
                help="Select the template file for structuring.",
            )
            if st.button("View Structuring Prompt"):
                st.session_state.show_structuring = (
                    not st.session_state.show_structuring
                )

        with col_prompt2:
            st.selectbox(
                "Judging Prompt Template",
                options=list(AVAILABLE_JUDGING_TEMPLATES.keys()),
                key="judging_template_select",
                help="Select the template file for judging.",
            )
            if st.button("View Judging Prompt"):
                st.session_state.show_judging = not st.session_state.show_judging

        if st.button("View Base Config.yaml"):
            st.session_state.show_config = not st.session_state.show_config

    # --- Display Prompt/Config Content ---
    if (
        st.session_state.show_structuring
        and st.session_state.structuring_template_select
    ):
        with st.expander("Structuring Prompt Content", expanded=True):
            try:
                path = AVAILABLE_STRUCTURING_TEMPLATES[
                    st.session_state.structuring_template_select
                ]
                content = Path(path).read_text()
                st.code(content, language="text")
            except Exception as e:
                st.error(f"Error reading structuring prompt: {e}")

    if st.session_state.show_judging and st.session_state.judging_template_select:
        with st.expander("Judging Prompt Content", expanded=True):
            try:
                path = AVAILABLE_JUDGING_TEMPLATES[
                    st.session_state.judging_template_select
                ]
                content = Path(path).read_text()
                st.code(content, language="text")
            except Exception as e:
                st.error(f"Error reading judging prompt: {e}")

    if st.session_state.show_config:
        with st.expander("Base Config.yaml Content", expanded=True):
            try:
                content = BASE_CONFIG_PATH.read_text()
                st.code(content, language="yaml")
            except Exception as e:
                st.error(f"Error reading config.yaml: {e}")

    # --- Config Completeness Check ---
    st.session_state.config_complete = all(
        [
            st.session_state.structuring_provider_select,
            st.session_state.structuring_model_select,
            st.session_state.judging_provider_select,
            st.session_state.judging_model_select,
            st.session_state.structuring_template_select,
            st.session_state.judging_template_select,
        ]
    )

    if st.session_state.config_complete:
        st.success("✅ Configuration is complete.")
    else:
        st.warning(
            "Configuration is incomplete. Please ensure all model and prompt fields are selected."
        )


def render_config_summary():
    """Displays a summary of the selected configuration."""
    st.subheader("Current Configuration Summary")
    struct_provider = st.session_state.structuring_provider_select
    struct_model_name = st.session_state.structuring_model_select
    struct_model_id = AVAILABLE_MODELS.get(struct_provider, {}).get(
        struct_model_name, "N/A"
    )
    struct_template_name = st.session_state.structuring_template_select

    judge_provider = st.session_state.judging_provider_select
    judge_model_name = st.session_state.judging_model_select
    judge_model_id = AVAILABLE_MODELS.get(judge_provider, {}).get(
        judge_model_name, "N/A"
    )
    judge_template_name = st.session_state.judging_template_select
    # judge_template_path = AVAILABLE_JUDGING_TEMPLATES.get( # Unused variable
    #     judge_template_name, "Not Selected"
    # )

    st.markdown(f"""
- **Structuring Model:** `{struct_provider}` - `{struct_model_name}` (`{struct_model_id}`)     |     Prompt: `{struct_template_name}`
- **Judging Model:**     `{judge_provider}` - `{judge_model_name}` (`{judge_model_id}`)     |     Prompt: `{judge_template_name}`
    """)
    # API Key status can be added here if needed


# --- Core Logic Functions ---


def generate_run_config() -> Optional[AppConfig]:
    """Generates the AppConfig object based on UI selections."""
    if not st.session_state.config_complete or not st.session_state.uploaded_files_info:
        st.error(
            "Cannot generate config. Ensure files are uploaded and configuration is complete."
        )
        return None

    try:
        # Load base config
        with open(BASE_CONFIG_PATH, "r") as f:
            base_config_data = yaml.safe_load(f)

        # Get selected model IDs
        struct_provider = st.session_state.structuring_provider_select
        struct_model_name = st.session_state.structuring_model_select
        struct_model_id = AVAILABLE_MODELS[struct_provider][struct_model_name]

        judge_provider = st.session_state.judging_provider_select
        judge_model_name = st.session_state.judging_model_select
        judge_model_id = AVAILABLE_MODELS[judge_provider][judge_model_name]

        # Get selected template paths
        struct_template_name = st.session_state.structuring_template_select
        struct_template_path = AVAILABLE_STRUCTURING_TEMPLATES[struct_template_name]

        judge_template_name = st.session_state.judging_template_select
        judge_template_path = AVAILABLE_JUDGING_TEMPLATES[judge_template_name]

        # Get API keys (use None if blank, so core logic uses env vars)
        struct_api_key = st.session_state.structuring_api_key_input or None
        judge_api_key = st.session_state.judging_api_key_input or None

        # Get input file paths
        input_file_paths = [f["path"] for f in st.session_state.uploaded_files_info]

        # --- Generate Run-Specific Output Directory ---
        # Combine stems of input files (limit length if needed)
        input_stems = sorted([Path(p).stem for p in input_file_paths])
        # Clean common suffixes like '_ingested'
        cleaned_stems = []
        common_suffixes = ["_ingested", "_tasks"]
        for stem in input_stems:
            for suffix in common_suffixes:
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break  # Only remove one suffix if multiple match
            cleaned_stems.append(stem)

        # Limit total length and join stems
        max_len_per_stem = 20  # Limit length contribution of each stem
        shortened_stems = [s[:max_len_per_stem] for s in cleaned_stems]
        combined_stem = "_".join(shortened_stems)
        max_total_stem_len = 60  # Limit overall stem length
        if len(combined_stem) > max_total_stem_len:
            combined_stem = combined_stem[:max_total_stem_len] + "_etc"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_specific_dir_name = f"{combined_stem}_{timestamp}"
        run_output_dir = DATA_DIR / run_specific_dir_name
        logger.info(f"Generated run-specific output directory: {run_output_dir}")
        # --- End Output Directory Generation ---

        # Create the config dictionary to override base config
        override_config = {
            "input_options": {"file_paths": input_file_paths},
            "structuring_settings": {
                "llm_client": {
                    "provider": struct_provider.lower(),
                    "model": struct_model_id,
                    "api_key": struct_api_key,
                },
                "prompt_template_path": struct_template_path,
                "structuring_model": struct_model_id,  # Also set the older key
            },
            "evaluation_settings": {
                "llm_client": {
                    "provider": judge_provider.lower(),
                    "model": judge_model_id,
                    "api_key": judge_api_key,
                },
                "prompt_template_path": judge_template_path,
                "judge_model": judge_model_id,  # Also set the older key
            },
            # Output options: Set output_dir to the generated run-specific path
            "output_options": {
                "output_dir": str(run_output_dir),  # Pass the full specific path
                # Ensure necessary save flags are True if not set in base config
                "save_evaluations_jsonl": True,
                "save_evaluations_json": True,
                "save_final_results_json": True,
            },
        }

        # Deep merge override_config into base_config_data (simple update for now)
        # A proper deep merge function would be more robust
        def deep_update(
            source: Dict[str, Any], overrides: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Recursively update a dictionary."""
            for key, value in overrides.items():
                if (
                    isinstance(value, dict)
                    and key in source
                    and isinstance(source[key], dict)
                ):
                    deep_update(source[key], value)  # Recursive call
                else:
                    source[key] = value  # Update or add key
            return source

        final_config_data = deep_update(base_config_data, override_config)

        # Validate and create AppConfig object
        app_config = AppConfig(**final_config_data)
        logger.info("Successfully generated AppConfig for evaluation run.")
        # Optionally log the generated config (be careful with API keys)
        # logger.debug(f"Generated AppConfig: {app_config.model_dump_json(indent=2)}")
        return app_config

    except Exception as e:
        st.error(f"Error generating configuration: {e}")
        logger.error(f"Error generating AppConfig: {traceback.format_exc()}")
        return None


# --- Queue Handler for Logging ---
class QueueLogHandler(logging.Handler):
    """Sends log records to a queue."""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord):
        """Puts formatted log message into the queue."""
        log_entry = self.format(record)
        self.log_queue.put(f"LOG:{log_entry}")


# --- Evaluation Worker Thread ---
def evaluation_worker(
    config: AppConfig,
    output_queue: queue.Queue,
    stop_event: threading.Event,
    log_queue: queue.Queue,
):
    """Runs the core evaluation logic in a separate thread."""
    # Add queue handler to root logger for this thread
    root_logger = logging.getLogger()
    queue_handler = QueueLogHandler(log_queue)
    # Use a simple formatter for queue logs
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    queue_handler.setFormatter(formatter)
    root_logger.addHandler(queue_handler)
    # Set level for this handler if needed (root logger level should control it)
    # queue_handler.setLevel(logging.DEBUG)

    try:
        output_queue.put("STATUS: Starting evaluation...")
        logger.info("Evaluation worker thread started.")
        logger.debug(f"Worker using config: {config.model_dump_json(indent=2)}")

        # Check stop event periodically if possible within run_batch_evaluation_core
        # If not possible, the stop event is mainly useful *before* starting the core logic
        if stop_event.is_set():
            output_queue.put("STATUS: Evaluation stopped before starting.")
            logger.info("Evaluation stopped by event before core logic execution.")
            return

        # --- Run the core evaluation ---
        # Pass the stop_event to the core function if it supports cancellation
        # Modify run_batch_evaluation_core to accept and check stop_event
        output_queue.put("STATUS: Running core evaluation...")
        # *** MODIFIED CALL ***
        results_paths = run_batch_evaluation_core(
            config=config,
            output_dir=config.output_options.output_dir,  # Pass explicitly
            use_structured=True,  # Pass explicitly, default to True
            stop_event=stop_event,  # Pass the event
            # Add a callback for progress updates if implemented
            # progress_callback=lambda p: output_queue.put(f"PROGRESS:{p}")
        )
        # --- Core evaluation finished ---

        if stop_event.is_set():
            output_queue.put("STATUS: Evaluation stopped during execution.")
            logger.info("Evaluation stopped by event during core logic execution.")
        elif results_paths:
            output_queue.put("STATUS: Evaluation completed successfully.")
            output_queue.put(f"RESULTS:{json.dumps(results_paths)}")
            logger.info(f"Evaluation completed. Result paths: {results_paths}")
        else:
            # This case might indicate an internal issue or early exit without error
            output_queue.put("STATUS: Evaluation finished with no results paths.")
            logger.warning("Evaluation finished, but no results paths were returned.")

    except Exception as e:
        error_msg = f"Error during evaluation: {e}"
        output_queue.put(f"ERROR:{error_msg}")
        logger.error(f"Error in evaluation worker: {traceback.format_exc()}")
    finally:
        output_queue.put("END")  # Signal completion
        # Remove the handler specific to this thread
        root_logger.removeHandler(queue_handler)
        logger.info("Evaluation worker thread finished.")


# --- Evaluation Control Functions ---
def start_core_evaluation():
    """Starts the evaluation process in a background thread."""
    logger.info("Attempting to start evaluation...")
    config = generate_run_config()
    if config:
        st.session_state.evaluation_running = True
        st.session_state.evaluation_error = None
        st.session_state.evaluation_results_paths = []
        st.session_state.last_run_output = []
        st.session_state.eval_start_time = time.time()
        st.session_state.eval_duration_str = "0:00:00"
        st.session_state.stop_event.clear()  # Reset stop event
        st.session_state.newly_completed_run_folder = None  # Reset completed folder

        # Clear OpenAI cache if applicable
        if (
            st.session_state.structuring_provider_select == "OpenAI"
            or st.session_state.judging_provider_select == "OpenAI"
        ):
            clear_openai_cache()
            logger.info("Cleared OpenAI cache.")

        # Start the worker thread
        st.session_state.worker_thread = threading.Thread(
            target=evaluation_worker,
            args=(
                config,
                st.session_state.output_queue,
                st.session_state.stop_event,
                st.session_state.output_queue,  # Pass the same queue for logs
            ),
            daemon=True,
        )
        st.session_state.worker_thread.start()
        logger.info("Evaluation worker thread started.")
        st.rerun()  # Update UI to show running state
    else:
        st.error("Failed to generate configuration. Cannot start evaluation.")
        logger.error("Failed to start evaluation due to config generation error.")


def stop_evaluation():
    """Signals the evaluation worker thread to stop."""
    if st.session_state.worker_thread and st.session_state.worker_thread.is_alive():
        st.session_state.stop_event.set()
        logger.info("Stop event set for evaluation worker thread.")
        # Optionally wait for a short period, but don't block indefinitely
        # st.session_state.worker_thread.join(timeout=2)
        st.session_state.evaluation_running = False  # Assume stop will work
        # Calculate duration up to stop point
        if st.session_state.eval_start_time:
            elapsed = time.time() - st.session_state.eval_start_time
            st.session_state.eval_duration_str = str(timedelta(seconds=int(elapsed)))
        st.warning("Stop signal sent. Evaluation may take a moment to halt.")
        st.rerun()
    else:
        st.info("No evaluation is currently running.")


# --- Results Processing and Display ---


# Define format_score function once at a higher scope if needed in multiple places
# Or keep it nested if only used within display_results_table
def format_score(score_value):
    """Formats score values with color coding for Streamlit display."""
    score_str = str(score_value).strip().lower()
    if score_str in ["pass", "yes"]:
        return f":green[{str(score_value).title()}]"
    elif score_str in ["fail", "no"]:
        return f":red[{str(score_value).title()}]"
    elif score_str == "partial":
        return f":orange[{str(score_value).title()}]"
    elif score_str == "n/a":
        return f":grey[{str(score_value).upper()}]"
    else:
        # Attempt to format numbers, otherwise return as string
        try:
            return f"{float(score_value):.2f}"
        except (ValueError, TypeError):
            return str(score_value)  # Default formatting


@st.cache_data(show_spinner="Loading results data...")  # Re-enabled caching
def load_and_process_results(
    selected_folders: List[str],
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Loads data from _final_results.json files, processes into a DataFrame, and aggregates summary statistics."""
    all_results_data = []
    processed_files_count = 0
    logger.info(f"Loading results from selected folders: {selected_folders}")

    if not selected_folders:
        logger.warning("No result folders selected for loading.")
        return None, None

    for folder_path_str in selected_folders:
        folder_path = Path(folder_path_str)
        # Find the _final_results.json file within the folder
        results_files = list(folder_path.glob("*_final_results.json"))
        if not results_files:
            logger.warning(f"No '*_final_results.json' file found in {folder_path}")
            continue

        # Assume only one such file per folder for simplicity
        results_file = results_files[0]
        try:
            with open(results_file, "r") as f:
                data = json.load(f)
                # Add folder name (run identifier) to each record
                run_identifier = folder_path.name
                # Check if 'results' key exists and is a list
                if isinstance(data.get("results"), list):
                    for record in data["results"]:
                        # Ensure record is a dictionary before assignment
                        if isinstance(record, dict):
                            # --- Flatten the structure: Loop through evaluations ---
                            task_info = {
                                "run_identifier": run_identifier,
                                "task_id": record.get("task_id"),
                                "prompt": record.get("prompt"),
                                "ideal_response": record.get("ideal_response"),
                                "final_answer_ideal": record.get(
                                    "final_answer"
                                ),  # Rename to avoid clash
                                "metadata": record.get("metadata"),
                                "structured_ideal_response": record.get(
                                    "structured_ideal_response"
                                ),
                            }
                            try:
                                timestamp_str = run_identifier.split("_")[-2:]
                                task_info["run_timestamp"] = datetime.strptime(
                                    "_".join(timestamp_str), "%Y%m%d_%H%M"
                                )
                            except (IndexError, ValueError):
                                task_info["run_timestamp"] = None

                            evaluations = record.get("evaluations", [])
                            if not evaluations:
                                logger.warning(
                                    f"Task {task_info.get('task_id')} in {results_file} has no evaluations."
                                )
                                # Optionally append a record with task info only if needed
                                # flat_record = task_info.copy()
                                # flat_record['model_id'] = 'N/A' # Indicate missing eval
                                # all_results_data.append(flat_record)
                                # file_processed_successfully = True # Mark file as processed even if no evals
                                continue  # Skip task if no evaluations

                            for evaluation_record in evaluations:
                                if not isinstance(evaluation_record, dict):
                                    logger.warning(
                                        f"Skipping non-dict evaluation in task {task_info.get('task_id')} in {results_file}"
                                    )
                                    continue

                                flat_record = task_info.copy()  # Start with task info
                                flat_record.update(
                                    evaluation_record
                                )  # Add evaluation info

                                # --- Data Cleaning/Normalization (Applied to evaluation) ---
                                judge_eval = flat_record.get("judge_evaluation", {})
                                if not isinstance(
                                    judge_eval, dict
                                ):  # Ensure judge_eval is a dict
                                    logger.warning(
                                        f"judge_evaluation is not a dict for model {flat_record.get('model_id')} in task {task_info.get('task_id')}. Skipping cleaning."
                                    )
                                    judge_eval = {}  # Use empty dict to avoid errors below

                                # Use aggregated_score directly from judge_evaluation if present
                                if "aggregated_score" not in judge_eval:
                                    # Fallback logic if aggregated_score is missing
                                    if judge_eval.get("judge_pass_fail") is not None:
                                        judge_eval["aggregated_score"] = judge_eval[
                                            "judge_pass_fail"
                                        ]
                                    elif (
                                        judge_eval.get("judge_overall_score")
                                        is not None
                                    ):
                                        judge_eval["aggregated_score"] = str(
                                            judge_eval["judge_overall_score"]
                                        )
                                    else:
                                        judge_eval["aggregated_score"] = (
                                            "N/A"  # Default if missing
                                        )

                                # Normalize 'Needs Review' (case-insensitive)
                                agg_score = judge_eval.get(
                                    "aggregated_score"
                                )  # Get score safely
                                if (
                                    isinstance(agg_score, str)
                                    and agg_score.lower() == "needs review"
                                ):
                                    judge_eval["aggregated_score"] = (
                                        "Needs Review"  # Standardize case
                                    )

                                # Normalize boolean-like rubric scores within judge_evaluation.parsed_rubric_scores
                                rubric_scores = judge_eval.get(
                                    "parsed_rubric_scores", {}
                                )
                                if isinstance(rubric_scores, dict):
                                    for (
                                        rubric_key,
                                        rubric_data,
                                    ) in rubric_scores.items():
                                        if (
                                            isinstance(rubric_data, dict)
                                            and "score" in rubric_data
                                        ):
                                            score_value = rubric_data["score"]
                                            if isinstance(score_value, str):
                                                val_lower = score_value.lower()
                                                if val_lower in ["yes", "pass"]:
                                                    rubric_data["score"] = (
                                                        "Yes"  # Standardize
                                                    )
                                                elif val_lower in ["no", "fail"]:
                                                    rubric_data["score"] = (
                                                        "No"  # Standardize
                                                    )
                                else:
                                    logger.warning(
                                        f"parsed_rubric_scores is not a dict for model {flat_record.get('model_id')} in task {task_info.get('task_id')}."
                                    )

                                # Update flat_record with potentially modified judge_eval
                                flat_record["judge_evaluation"] = judge_eval

                                # Add cleaned/derived fields directly to flat_record for easier access later
                                flat_record["aggregated_score"] = judge_eval.get(
                                    "aggregated_score", "N/A"
                                )
                                # Add individual rubric scores as top-level columns
                                if isinstance(rubric_scores, dict):
                                    for (
                                        rubric_key,
                                        rubric_data,
                                    ) in rubric_scores.items():
                                        if isinstance(rubric_data, dict):
                                            # Create column name like judge_rubric_Problem_Understanding_score
                                            col_name = (
                                                f"judge_rubric_{rubric_key}_score"
                                            )
                                            flat_record[col_name] = rubric_data.get(
                                                "score"
                                            )

                                all_results_data.append(flat_record)
                                file_processed_successfully = (
                                    True  # Mark success for this file
                                )

                            # --- End of evaluation loop ---
                        else:  # Corresponds to 'if isinstance(record, dict)'
                            logger.warning(
                                f"Skipping non-dict record in {results_file}: {record}"
                            )
                            continue  # Skip to the next record

                    # Increment count only if the file had valid records processed
                    if file_processed_successfully:
                        processed_files_count += 1

                else:  # Corresponds to 'if isinstance(data.get("results"), list)'
                    logger.warning(
                        f"'results' key not found or not a list in {results_file}. Skipping file."
                    )
                    continue  # Skip this file

        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {results_file}. Skipping file.")
            continue  # Skip this file if JSON is invalid
        except Exception as e:
            logger.error(
                f"Unexpected error processing {results_file}: {e}", exc_info=True
            )
            continue  # Skip this file on other unexpected errors

    # --- After processing all folders ---
    if not all_results_data:
        logger.warning("No valid records found in any processed files.")
        return None, None  # Return None if no data was successfully appended

    # Convert collected data to DataFrame
    try:
        results_df = pd.DataFrame(all_results_data)
        logger.info(
            f"Successfully loaded {len(results_df)} records from {processed_files_count} files."
        )

        # --- Aggregate Statistics ---
        summary_stats = {}
        if not results_df.empty:
            # Example: Calculate average score if 'judge_overall_score' is numeric
            numeric_scores = pd.to_numeric(
                results_df.get("judge_overall_score"), errors="coerce"
            )
            mean_score = numeric_scores.mean()  # Calculate mean (skips NA in Series)
            if pd.notna(mean_score):  # Check if the resulting mean is a valid number
                summary_stats["average_judge_score"] = mean_score

            # Count occurrences of aggregated scores
            if "aggregated_score" in results_df.columns:
                summary_stats["score_counts"] = (
                    results_df["aggregated_score"].value_counts().to_dict()
                )

            # Count tasks needing review
            if "aggregated_score" in results_df.columns:
                summary_stats["needs_review_count"] = results_df[
                    results_df["aggregated_score"] == "Needs Review"
                ].shape[0]

            # Add more stats as needed...
            logger.info(f"Calculated summary statistics: {summary_stats}")

        return results_df, summary_stats

    except Exception as e:
        logger.error(
            f"Error creating DataFrame or calculating stats: {e}", exc_info=True
        )
        return None, None  # Return None if DataFrame creation fails

    # --- Aggregated Summary Calculation ---
    total_evaluations = len(results_df)
    needs_review_count = (
        len(results_df[results_df["aggregated_score"] == "Needs Review"])
        if "aggregated_score" in results_df.columns
        else 0
    )
    pass_fail_counts = (
        results_df["aggregated_score"].value_counts().to_dict()
        if "aggregated_score" in results_df.columns
        else {}
    )

    # Calculate average scores for numeric rubrics if they exist
    avg_scores = {}
    numeric_rubric_cols = [
        col
        for col in results_df.columns
        if col.startswith("judge_rubric_") and col.endswith("_score")
    ]
    for col in numeric_rubric_cols:
        # Attempt to convert to numeric, coercing errors to NaN
        numeric_series = pd.to_numeric(results_df[col], errors="coerce")
        if not numeric_series.isnull().all():  # Check if there are any valid numbers
            avg_scores[col] = numeric_series.mean()

    aggregated_summary = {
        "total_evaluations": total_evaluations,
        "needs_review_count": needs_review_count,
        "pass_fail_counts": pass_fail_counts,
        "average_numeric_rubric_scores": avg_scores,
        "processed_files": processed_files_count,
        "selected_folders": selected_folders,
    }
    logger.info(f"Aggregated summary calculated: {aggregated_summary}")

    return results_df, aggregated_summary


def render_results_selector():
    """Allows the user to select which evaluation result folders to display."""
    # *** MODIFIED: Removed sidebar reference ***
    st.subheader("Select Evaluation Runs")  # Changed from sidebar.header
    try:
        # List subdirectories in DATA_DIR
        all_folders = [
            d for d in DATA_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]
        # Sort folders by name (often includes timestamp)
        all_folders.sort(key=lambda x: x.name, reverse=True)  # Show newest first
        folder_options = {d.name: str(d) for d in all_folders}

        if not folder_options:
            # *** MODIFIED: Removed sidebar reference ***
            st.warning("No evaluation result folders found in data directory.")
            st.session_state.selected_results_folders = []  # Clear selection
            return

        # Determine default selection: newly completed run or previously selected
        default_selection = []
        if st.session_state.newly_completed_run_folder:
            # Ensure the newly completed folder actually exists before selecting
            new_folder_path = Path(st.session_state.newly_completed_run_folder)
            if new_folder_path.exists() and new_folder_path.name in folder_options:
                default_selection = [st.session_state.newly_completed_run_folder]
                logger.info(
                    f"Defaulting selection to newly completed run: {new_folder_path.name}"
                )
            else:
                logger.warning(
                    f"Newly completed folder {st.session_state.newly_completed_run_folder} not found or invalid, clearing."
                )
                st.session_state.newly_completed_run_folder = None  # Clear if invalid
                # Fall back to previous selection if available
                if st.session_state.selected_results_folders:
                    # Filter previous selection to ensure they still exist
                    valid_previous = [
                        f
                        for f in st.session_state.selected_results_folders
                        if Path(f).exists() and Path(f).name in folder_options
                    ]
                    default_selection = valid_previous
                    logger.info(
                        f"Falling back to valid previous selection: {default_selection}"
                    )

        elif st.session_state.selected_results_folders:
            # Filter previous selection to ensure they still exist
            valid_previous = [
                f
                for f in st.session_state.selected_results_folders
                if Path(f).exists() and Path(f).name in folder_options
            ]
            default_selection = valid_previous
            logger.info(
                f"Using valid previous selection as default: {default_selection}"
            )

        # *** MODIFIED: Removed sidebar reference ***
        selected_folder_names = st.multiselect(  # Changed from sidebar.multiselect
            "Choose result folders:",
            options=list(folder_options.keys()),
            # Select the folder from the most recent run by default, if available
            default=[Path(f).name for f in default_selection],
            key="results_folder_multiselect",
        )

        # Update session state with full paths
        st.session_state.selected_results_folders = [
            folder_options[name] for name in selected_folder_names
        ]
        # Clear the 'newly completed' flag after it's been used for default selection
        # st.session_state.newly_completed_run_folder = None # Keep it until next run?

    except FileNotFoundError:
        # *** MODIFIED: Removed sidebar reference ***
        st.error(f"Data directory not found: {DATA_DIR}")
        logger.error(f"Data directory not found: {DATA_DIR}")
        st.session_state.selected_results_folders = []  # Clear selection
    except Exception as e:
        # *** MODIFIED: Removed sidebar reference ***
        st.error(f"Error listing result folders: {e}")
        logger.error(f"Error listing result folders: {traceback.format_exc()}")
        st.session_state.selected_results_folders = []  # Clear selection


def display_summary_stats(summary_data: Dict[str, Any]):
    """Displays the aggregated summary statistics."""
    st.subheader("📈 Evaluation Summary")
    if not summary_data:
        st.warning("No summary data available.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Evaluations Processed", summary_data.get("total_evaluations", 0))
    col2.metric("Tasks Flagged for Review", summary_data.get("needs_review_count", 0))
    col3.metric("Result Folders Analyzed", summary_data.get("processed_files", 0))

    st.write("**Score Distribution:**")
    pass_fail_counts = summary_data.get("pass_fail_counts", {})
    if pass_fail_counts:
        # Convert counts to DataFrame for easier display
        score_df = pd.DataFrame(
            pass_fail_counts.items(), columns=["Aggregated Score", "Count"]
        ).sort_values(by="Aggregated Score")
        st.dataframe(score_df, hide_index=True)

        # Optional: Plot score distribution
        try:
            fig = px.pie(
                score_df,
                values="Count",
                names="Aggregated Score",
                title="Aggregated Score Distribution",
                color="Aggregated Score",
                color_discrete_map=COLOR_MAP,  # Use defined color map
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate score distribution plot: {e}")
            logger.warning(f"Plotly error for score distribution: {e}")

    else:
        st.info("No aggregated score data available for distribution.")

    # Display average numeric rubric scores if available
    avg_scores = summary_data.get("average_numeric_rubric_scores", {})
    if avg_scores:
        st.write("**Average Numeric Rubric Scores:**")
        avg_df = pd.DataFrame(avg_scores.items(), columns=["Rubric", "Average Score"])
        # Format rubric names
        avg_df["Rubric"] = (
            avg_df["Rubric"]
            .str.replace("judge_rubric_", "")
            .str.replace("_score", "")
            .str.replace("_", " ")
            .str.title()
        )
        st.dataframe(avg_df.round(2), hide_index=True)  # Round for display


def display_performance_plots(df: pd.DataFrame):
    """Displays performance plots based on aggregated scores."""
    st.subheader("📊 Performance Overview")
    if df is None or df.empty or "aggregated_score" not in df.columns:
        st.warning("No data available for performance plots.")
        return

    # --- Overall Performance Distribution ---
    st.write("**Overall Aggregated Score Distribution:**")
    try:
        # Ensure consistent category order and colors
        all_possible_scores = ["Pass", "Partial", "Fail"] + [
            s
            for s in df["aggregated_score"].unique()
            if s not in ["Pass", "Partial", "Fail"]
        ]
        performance_counts = (
            df["aggregated_score"]
            .value_counts()
            .reindex(all_possible_scores, fill_value=0)
            .reset_index()
        )
        performance_counts.columns = ["Aggregated Score", "Count"]

        color_discrete_map = COLOR_MAP  # Use the constant

        fig_perf = px.bar(
            performance_counts,
            x="Aggregated Score",
            y="Count",
            title="Distribution of Aggregated Scores",
            color="Aggregated Score",
            color_discrete_map=color_discrete_map,
            category_orders={"Aggregated Score": all_possible_scores},  # Enforce order
            labels={"Count": "Number of Evaluations"},
        )
        fig_perf.update_layout(xaxis_title="Aggregated Score", yaxis_title="Count")
        st.plotly_chart(fig_perf, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate overall performance plot: {e}")
        logger.warning(f"Plotly error for overall performance: {e}")

    # --- Performance by Model ---
    if "model_id" in df.columns:
        st.write("**Aggregated Scores by Model:**")
        try:
            model_perf_counts = (
                df.groupby("model_id")["aggregated_score"]
                .value_counts()
                .unstack(fill_value=0)
                .reset_index()
            )
            # Ensure all possible score columns exist
            for score in all_possible_scores:
                if score not in model_perf_counts.columns:
                    model_perf_counts[score] = 0
            # Reorder columns for consistent plotting
            model_perf_counts = model_perf_counts[["model_id"] + all_possible_scores]

            # Melt for Plotly Express
            model_perf_melted = model_perf_counts.melt(
                id_vars="model_id", var_name="Aggregated Score", value_name="Count"
            )

            fig_model_perf = px.bar(
                model_perf_melted,
                x="model_id",
                y="Count",
                color="Aggregated Score",
                title="Aggregated Scores per Model",
                barmode="group",  # Or 'stack'
                color_discrete_map=color_discrete_map,
                category_orders={
                    "Aggregated Score": all_possible_scores
                },  # Enforce order
                labels={"model_id": "Model ID", "Count": "Number of Evaluations"},
            )
            fig_model_perf.update_layout(xaxis_title="Model ID", yaxis_title="Count")
            st.plotly_chart(fig_model_perf, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate performance by model plot: {e}")
            logger.warning(f"Plotly error for performance by model: {e}")


def display_rubric_plots(df: pd.DataFrame):
    """Displays plots for rubric scores."""
    st.subheader("📐 Rubric Score Analysis")
    if df is None or df.empty:
        st.warning("No data available for rubric analysis.")
        return

    # Identify rubric score columns dynamically
    rubric_cols = sorted(
        [
            col
            for col in df.columns
            if col.startswith("judge_rubric_") and col.endswith("_score")
        ]
    )

    if not rubric_cols:
        st.info(
            "No rubric score columns found in the data (expected format: 'judge_rubric_*_score')."
        )
        return

    # --- Overall Rubric Score Distribution ---
    st.write("**Distribution Across All Rubrics:**")
    try:
        # Melt the DataFrame to have rubric criteria and scores in separate columns
        id_vars = [
            "task_id",
            "model_id",
            "run_identifier",
        ]  # Identify unique evaluations
        # Ensure id_vars exist
        id_vars = [v for v in id_vars if v in df.columns]
        if not id_vars:
            st.warning(
                "Cannot create unique ID for melting rubric data (missing task_id, model_id, or run_identifier)."
            )
            return  # Cannot proceed without unique IDs

        rubric_melted = df.melt(
            id_vars=id_vars,
            value_vars=rubric_cols,
            var_name="Rubric Criterion",
            value_name="Score",
        )
        # Clean up rubric names for display
        rubric_melted["Rubric Criterion"] = (
            rubric_melted["Rubric Criterion"]
            .str.replace("judge_rubric_", "")
            .str.replace("_score", "")
            .str.replace("_", " ")
            .str.title()
        )

        # Define expected categories and colors (handle numeric scores if present)
        # Convert scores to string for consistent categorization first
        rubric_melted["Score_Str"] = rubric_melted["Score"].astype(str)

        # Determine unique scores present
        unique_scores = rubric_melted["Score_Str"].dropna().unique()
        # Define standard categories first
        rubric_score_categories = ["Yes", "No", "Partial", "N/A"]
        # Add any other scores found (could be numbers or other strings)
        other_scores = sorted(
            [s for s in unique_scores if s not in rubric_score_categories]
        )
        all_categories = rubric_score_categories + other_scores

        # Count occurrences for each rubric and score
        rubric_counts = (
            rubric_melted.groupby(["Rubric Criterion", "Score_Str"])
            .size()
            .unstack(fill_value=0)
            .reindex(
                columns=all_categories, fill_value=0
            )  # Ensure all categories exist
            .reset_index()
        )
        rubric_counts_melted = rubric_counts.melt(
            id_vars="Rubric Criterion", var_name="Score", value_name="Count"
        )

        # Define colors - extend base map if needed
        rubric_color_map = COLOR_MAP.copy()
        # Add colors for any non-standard scores if necessary
        # Example: rubric_color_map.update({"1.0": "lightblue", "0.5": "lightgreen"})

        fig_rubric = px.bar(
            rubric_counts_melted,
            x="Rubric Criterion",
            y="Count",
            color="Score",
            title="Score Distribution per Rubric Criterion",
            barmode="group",
            color_discrete_map=rubric_color_map,
            category_orders={"Score": all_categories},  # Enforce order
            labels={"Count": "Number of Evaluations"},
        )
        fig_rubric.update_layout(xaxis_title="Rubric Criterion", yaxis_title="Count")
        st.plotly_chart(fig_rubric, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not generate overall rubric distribution plot: {e}")
        logger.warning(
            f"Plotly error for overall rubric distribution: {traceback.format_exc()}"
        )

    # --- Rubric Scores by Model ---
    if "model_id" in df.columns:
        st.write("**Rubric Scores by Model:**")
        try:
            # Use the previously melted data (rubric_melted)
            rubric_model_counts = (
                rubric_melted.groupby(["model_id", "Rubric Criterion", "Score_Str"])
                .size()
                .unstack(fill_value=0)  # Unstack by Score_Str
                .reindex(
                    columns=all_categories, fill_value=0
                )  # Ensure all score columns
                .stack()  # Stack back to long format
                .reset_index(name="Count")
            )

            # Allow user to select a specific rubric criterion to view by model
            criterion_options = sorted(rubric_melted["Rubric Criterion"].unique())
            selected_criterion = st.selectbox(
                "Select Rubric Criterion to view by Model:",
                options=criterion_options,
                key="rubric_model_filter",
            )

            if selected_criterion:
                criterion_data = rubric_model_counts[
                    rubric_model_counts["Rubric Criterion"] == selected_criterion
                ]

                if not criterion_data.empty:
                    fig_rubric_model = px.bar(
                        criterion_data,
                        x="model_id",
                        y="Count",
                        color="Score_Str",  # Color by score string
                        title=f"'{selected_criterion}' Scores per Model",
                        barmode="group",
                        color_discrete_map=rubric_color_map,
                        category_orders={
                            "Score_Str": all_categories
                        },  # Use score string categories
                        labels={
                            "model_id": "Model ID",
                            "Count": "Number of Evaluations",
                            "Score_Str": "Score",
                        },
                    )
                    fig_rubric_model.update_layout(
                        xaxis_title="Model ID", yaxis_title="Count"
                    )
                    st.plotly_chart(fig_rubric_model, use_container_width=True)
                else:
                    st.info(f"No data available for criterion '{selected_criterion}'.")
            else:
                st.info("Select a rubric criterion to see the breakdown by model.")

        except Exception as e:
            st.warning(f"Could not generate rubric scores by model plot: {e}")
            logger.warning(
                f"Plotly error for rubric scores by model: {traceback.format_exc()}"
            )


def display_results_table(df: pd.DataFrame):
    """Displays the detailed results in a filterable table and allows drilling down."""
    st.subheader("📄 Detailed Results")
    if df is None or df.empty:
        st.warning("No results data available to display.")
        return

    # --- Filtering Options ---
    st.write("**Filter Results:**")
    col_filter1, col_filter2, col_filter3 = st.columns(3)

    # Filter by Model ID
    model_options = ["All"] + sorted(df["model_id"].unique())
    selected_model = col_filter1.selectbox("Model ID:", model_options)

    # Filter by Aggregated Score
    score_options = ["All"] + sorted(df["aggregated_score"].unique())
    selected_scores = col_filter2.multiselect(
        "Aggregated Score(s):", score_options, default=["All"]
    )

    # Filter by Subject (if exists)
    subject_options = ["All"]
    if "subject" in df.columns:
        subject_options.extend(sorted(df["subject"].dropna().unique()))
    selected_subject = col_filter3.selectbox("Subject:", subject_options)

    # Filter by Complexity (if exists)
    complexity_options = ["All"]
    if "complexity" in df.columns:
        complexity_options.extend(sorted(df["complexity"].dropna().unique()))
    selected_complexity = col_filter3.selectbox("Complexity:", complexity_options)

    # Filter by Needs Review (if exists) - Moved from separate section
    review_options = ["All", "Needs Review Only", "Exclude Needs Review"]
    if "aggregated_score" in df.columns:
        selected_review_status = st.radio(
            "Human Review Status Filter:",  # Renamed label slightly
            review_options,
            index=0,  # Default to 'All'
            horizontal=True,
            key="review_filter_radio",
        )
    else:
        selected_review_status = "All"

    # Apply filters
    filtered_df = df.copy()
    if selected_model != "All":
        filtered_df = filtered_df[filtered_df["model_id"] == selected_model]
    if "All" not in selected_scores:
        filtered_df = filtered_df[filtered_df["aggregated_score"].isin(selected_scores)]
    if selected_subject != "All" and "subject" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["subject"] == selected_subject]
    if selected_complexity != "All" and "complexity" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["complexity"] == selected_complexity]

    # Apply review status filter
    if selected_review_status == "Needs Review Only":
        filtered_df = filtered_df[filtered_df["aggregated_score"] == "Needs Review"]
    elif selected_review_status == "Exclude Needs Review":
        filtered_df = filtered_df[filtered_df["aggregated_score"] != "Needs Review"]

    # --- Display Table ---
    st.write(f"**Displaying {len(filtered_df)} / {len(df)} Evaluations**")

    # Define columns to show in the main table (can be customized)
    # Start with basic identifiers and scores
    cols_to_show = [
        "task_id",
        "model_id",
        "aggregated_score",
        "subject",
        "complexity",
    ]
    # Dynamically add rubric score columns
    rubric_score_cols_main = sorted(
        [
            col
            for col in filtered_df.columns
            if col.startswith("judge_rubric_") and col.endswith("_score")
        ]
    )
    cols_to_show.extend(rubric_score_cols_main)
    # Add other potentially interesting judge fields
    cols_to_show.extend(["judge_overall_score", "judge_pass_fail", "judge_comments"])

    # Filter out columns that don't exist in the DataFrame
    cols_to_show = [col for col in cols_to_show if col in filtered_df.columns]

    if not filtered_df.empty:
        # Create display DF with formatted column names
        display_table_df = filtered_df[cols_to_show].copy()
        column_rename_map_table = {
            "task_id": "Task ID",
            "model_id": "Model ID",
            "aggregated_score": "Agg Score",  # Abbreviate for table
            "subject": "Subject",
            "complexity": "Complexity",
            "judge_overall_score": "Judge Score",
            "judge_pass_fail": "Judge P/F",
            "judge_comments": "Judge Comments",
        }
        for col in rubric_score_cols_main:
            formatted_name = (
                col.replace("judge_rubric_", "")
                .replace("_score", "")
                .replace("_", " ")
                .title()
            )
            column_rename_map_table[col] = formatted_name  # Use formatted name

        display_table_df.rename(columns=column_rename_map_table, inplace=True)
        # Get the final list of renamed columns in the desired order
        final_table_cols = [
            column_rename_map_table.get(col, col) for col in cols_to_show
        ]

        st.dataframe(
            display_table_df[final_table_cols],  # Use renamed columns
            use_container_width=True,
            hide_index=True,
        )

        # --- Drill-down Functionality (MODIFIED) ---
        st.write("**Select Evaluation for Details:**")
        col_select1, col_select2 = st.columns(2)

        # Dropdown for Task ID
        task_id_options = ["-"] + sorted(
            list(filtered_df["task_id"].astype(str).unique())
        )
        # Use session state to store/retrieve selection to persist across interactions
        selected_task_id_str = col_select1.selectbox(
            "Select Task ID:",
            options=task_id_options,
            index=task_id_options.index(st.session_state.view_selected_task_id)
            if st.session_state.view_selected_task_id in task_id_options
            else 0,
            key="drilldown_task_select_stateful",  # Use a key that persists
            on_change=lambda: st.session_state.update(
                view_selected_task_id=st.session_state.drilldown_task_select_stateful,
                view_selected_model_id=None,
            ),  # Reset model on task change
        )
        # Update session state if changed via selectbox
        st.session_state.view_selected_task_id = selected_task_id_str

        # Dropdown for Model ID (filtered by selected task)
        if selected_task_id_str != "-":
            try:
                selected_task_id_orig_type = df["task_id"].dtype.type(
                    selected_task_id_str
                )
            except:
                selected_task_id_orig_type = selected_task_id_str
            available_models_for_task = sorted(
                list(
                    filtered_df[filtered_df["task_id"] == selected_task_id_orig_type][
                        "model_id"
                    ].unique()
                )
            )
            model_id_options = ["-"] + available_models_for_task
        else:
            model_id_options = ["-"]  # Only show '-' if no task is selected

        selected_model_id = col_select2.selectbox(
            "Select Model ID:",
            options=model_id_options,
            index=model_id_options.index(st.session_state.view_selected_model_id)
            if st.session_state.view_selected_model_id in model_id_options
            else 0,
            key="drilldown_model_select_stateful",  # Use a key that persists
            on_change=lambda: st.session_state.update(
                view_selected_model_id=st.session_state.drilldown_model_select_stateful
            ),
        )
        # Update session state if changed via selectbox
        st.session_state.view_selected_model_id = selected_model_id

        # Display details only if BOTH Task and Model are selected
        if selected_task_id_str != "-" and selected_model_id != "-":
            # Find the specific row using the selected IDs
            try:
                selected_task_id_orig_type = df["task_id"].dtype.type(
                    selected_task_id_str
                )
            except:
                selected_task_id_orig_type = selected_task_id_str

            selected_rows = filtered_df[
                (filtered_df["task_id"] == selected_task_id_orig_type)
                & (filtered_df["model_id"] == selected_model_id)
            ]

            if not selected_rows.empty:
                selected_row = selected_rows.iloc[0]

                st.subheader(
                    f"Details for Task {selected_task_id_str} | Model {selected_model_id}"
                )

                # Display Prompt, Model Response, Ideal Response (as before)
                with st.expander("Prompt", expanded=False):
                    st.text_area(
                        "Prompt",
                        selected_row.get("prompt", "N/A"),
                        height=150,
                        disabled=True,
                    )
                with st.expander("Model Response", expanded=True):
                    st.text_area(
                        "Model Response",
                        selected_row.get("model_response", "N/A"),
                        height=200,
                        disabled=True,
                    )
                with st.expander("Ideal Response / Ground Truth", expanded=False):
                    st.text_area(
                        "Ideal Response",
                        selected_row.get("ideal_response", "N/A"),
                        height=200,
                        disabled=True,
                    )
                    st.markdown("**Final Answer (Ground Truth):**")
                    st.text_area(
                        "Final Answer (Ground Truth)",
                        selected_row.get("final_answer_ground_truth", "N/A"),
                        height=100,
                        disabled=True,
                    )

                # Display Judge Evaluation Details (Collapsed JSON - MODIFIED)
                st.write("**Judge Evaluation:**")
                with st.expander("View Judge Evaluation Details", expanded=False):
                    # Define the internal keys corresponding to the required display keys
                    # Use .get() for safety, provide default 'N/A'
                    judge_data_to_display = {
                        "Llm Model": selected_row.get(
                            "judge_model_id", "N/A"
                        ),  # Assuming judge_model_id stores the judging model used
                        "Prompt Template Path": selected_row.get(
                            "judge_prompt_template_path", "N/A"
                        ),  # Assuming this is stored
                        "Aggregated Score": selected_row.get("aggregated_score", "N/A"),
                        "Verification Message": selected_row.get(
                            "judge_comments", "N/A"
                        ),  # Assuming judge_comments holds this
                        "Needs Human Review": selected_row.get("aggregated_score")
                        == "Needs Review",  # Derived
                        "Review Reasons": selected_row.get(
                            "judge_review_reasons", "N/A"
                        ),  # Assuming this column exists or is added
                        "Human Review Status": selected_row.get(
                            "human_review_status", "Not Reviewed"
                        ),  # Assuming this column exists or is added
                        "Created At": str(
                            selected_row.get("run_timestamp", "N/A")
                        ),  # Assuming timestamp is available per row, convert to string for JSON
                    }
                    # Ensure the order matches the requirement
                    ordered_keys = [
                        "Llm Model",
                        "Prompt Template Path",
                        "Aggregated Score",
                        "Verification Message",
                        "Needs Human Review",
                        "Review Reasons",
                        "Human Review Status",
                        "Created At",
                    ]
                    final_judge_data = {
                        k: judge_data_to_display.get(k) for k in ordered_keys
                    }

                    st.json(final_judge_data)

                # Display Rubric Scores and Justifications (Optional - Keep if needed)
                judge_cols = [
                    col for col in selected_row.index if col.startswith("judge_")
                ]
                rubric_score_cols = sorted(
                    [  # Sort for consistency
                        col
                        for col in judge_cols
                        if col.startswith("judge_rubric_") and col.endswith("_score")
                    ]
                )
                if rubric_score_cols:
                    with st.expander(
                        "View Rubric Scores & Justifications", expanded=False
                    ):
                        # Use the globally defined format_score function
                        for score_col in rubric_score_cols:
                            criterion_raw = score_col.replace(
                                "judge_rubric_", ""
                            ).replace("_score", "")
                            criterion_formatted = criterion_raw.replace(
                                "_", " "
                            ).title()
                            score = selected_row.get(score_col, "N/A")
                            just_col = f"judge_rubric_{criterion_raw}_justification"
                            justification = selected_row.get(
                                just_col, "*No justification provided.*"
                            )

                            formatted_score = format_score(score)  # Use helper

                            st.markdown(f"**{criterion_formatted}:** {formatted_score}")
                            st.markdown(f"> _{justification}_")
                            st.markdown("---")  # Separator

            else:
                # This case should be less likely now with filtered dropdowns
                st.warning(
                    f"No data found for the selected Task ID '{selected_task_id_str}' and Model ID '{selected_model_id}' combination in the current filtered view."
                )
        # else: # Optional: Message if selection is incomplete
        # st.info("Select both a Task ID and a Model ID to view details.")

    else:
        st.info("No evaluations match the current filter criteria.")


# --- New Function for Human Review ---
def display_human_review_tasks(df: pd.DataFrame):
    """Displays tasks flagged for human review in a table with specified columns."""  # Docstring updated
    st.subheader("🚩 Tasks Flagged for Human Review")
    if df is None or df.empty or "aggregated_score" not in df.columns:
        st.warning("No results data available to check for review tasks.")
        return

    review_df = df[df["aggregated_score"] == "Needs Review"].copy()
    review_count = len(review_df)

    st.metric("Tasks Flagged for Review", review_count)

    if review_count > 0:
        with st.expander(
            f"View {review_count} Tasks Needing Review", expanded=True
        ):  # Expand by default if tasks exist
            # Dynamically find rubric score columns
            rubric_score_cols = sorted(
                [  # Sort for consistent order
                    col
                    for col in review_df.columns
                    if col.startswith("judge_rubric_") and col.endswith("_score")
                ]
            )

            # Define base columns in desired order
            base_cols = ["task_id", "model_id"]  # Removed aggregated_score initially

            # Combine and ensure they exist in the review_df
            cols_to_show_review = (
                base_cols + rubric_score_cols + ["aggregated_score"]
            )  # Add aggregated score at the end
            cols_to_show_review = [
                col for col in cols_to_show_review if col in review_df.columns
            ]

            # Create a copy for display formatting
            display_df = review_df[cols_to_show_review].copy()

            # Format column names for display
            column_rename_map = {
                "task_id": "Task ID",
                "model_id": "Model ID",
                "aggregated_score": "Aggregated Score",  # Keep this mapping
            }
            # Add rubric columns to the rename map
            for col in rubric_score_cols:
                # Example: judge_rubric_accuracy_score -> Accuracy Score
                formatted_name = (
                    col.replace("judge_rubric_", "")
                    .replace("_score", "")
                    .replace("_", " ")
                    .title()
                )
                column_rename_map[col] = formatted_name

            display_df.rename(columns=column_rename_map, inplace=True)

            # Reorder columns for final display after renaming according to requirements
            final_column_order = (
                ["Task ID", "Model ID"]
                + [column_rename_map[col] for col in rubric_score_cols]
                + ["Aggregated Score"]
            )
            # Filter out any columns that might not exist after renaming (safety check)
            final_column_order = [
                col for col in final_column_order if col in display_df.columns
            ]

            # Display the formatted table
            st.dataframe(
                display_df[final_column_order],  # Use the reordered list
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.success("No tasks are currently flagged for human review.")


# --- Main App Layout and Logic ---


def render_evaluation_progress(
    output_placeholder, log_placeholder, progress_placeholder
):
    """Displays the progress/output of the evaluation run."""
    st.subheader("🚀 Evaluation Progress")

    # --- Run/Stop Buttons ---
    col_run, col_stop = st.columns(2)
    run_button_disabled = (
        st.session_state.evaluation_running
        or not st.session_state.config_complete
        or not st.session_state.uploaded_files_info
    )
    run_button_label = (
        "Running..." if st.session_state.evaluation_running else "Start Evaluation"
    )
    if col_run.button(run_button_label, disabled=run_button_disabled, type="primary"):
        start_core_evaluation()

    stop_button_disabled = not st.session_state.evaluation_running
    if col_stop.button("Stop Evaluation", disabled=stop_button_disabled):
        stop_evaluation()

    # --- Status, Progress Bar, and Elapsed Time (MODIFIED LAYOUT) ---

    # Calculate duration
    if st.session_state.eval_start_time and st.session_state.evaluation_running:
        elapsed = time.time() - st.session_state.eval_start_time
        st.session_state.eval_duration_str = str(timedelta(seconds=int(elapsed)))
    elif not st.session_state.evaluation_running and st.session_state.eval_start_time:
        # Keep showing final duration if run finished or was stopped
        pass  # Duration string is already set by worker or stop function
    else:
        st.session_state.eval_duration_str = "Not Started"  # Reset if never started

    # Determine status and progress value
    status_text = "Idle"
    progress_value = 0
    if st.session_state.evaluation_running:
        status_text = "Running..."
        # Simple indeterminate progress for now
        # TODO: Parse progress updates from the queue if available
        progress_value = 50  # Placeholder for running state
    elif st.session_state.evaluation_error:
        status_text = f"Error: {st.session_state.evaluation_error}"
        progress_value = 0
    elif (
        st.session_state.evaluation_results_paths
        and not st.session_state.evaluation_running
        and not st.session_state.evaluation_error
    ):
        # Check for error flag before declaring success
        status_text = "Finished Successfully"
        progress_value = 100
        # Display success message prompting user to check results tab
        st.success("Evaluation complete. Please check the 'View Results' tab.")
    elif (
        st.session_state.stop_event.is_set() and not st.session_state.evaluation_running
    ):  # Check if stopped and not running
        status_text = "Stopped by User"
        progress_value = 0
    elif (
        st.session_state.eval_start_time and not st.session_state.evaluation_running
    ):  # Started but not running and no results/error/stop yet
        status_text = "Finished (Unknown Status)"  # Clarified status
        progress_value = 0

    # Display Status ABOVE the progress bar and time
    # Use the dedicated placeholder if it's meant ONLY for status, otherwise use st.info/st.error etc.
    # Assuming output_placeholder is primarily for status messages
    if st.session_state.evaluation_error:
        output_placeholder.error(f"Status: {status_text}")
    else:
        output_placeholder.info(f"Status: {status_text}")

    # Use columns for Progress Bar and Elapsed Time side-by-side BELOW status
    col_progress, col_time = st.columns([3, 1])

    with col_progress:
        # Use the dedicated placeholder if it's meant ONLY for progress bar
        progress_placeholder.progress(
            progress_value, text="Evaluation Progress"
        )  # Add text label

    with col_time:
        # Use st.metric directly in the column
        st.metric("Elapsed Time", st.session_state.eval_duration_str)

    # --- Live Log Output (Collapsible) ---
    # Expander is now controlled by evaluation_running state
    log_expander = log_placeholder.expander(
        "Live Evaluation Output", expanded=st.session_state.evaluation_running
    )
    log_container = log_expander.container(height=400)  # Scrollable container

    # Process queue messages
    while not st.session_state.output_queue.empty():
        try:
            message = st.session_state.output_queue.get_nowait()
            if message == "END":
                st.session_state.evaluation_running = False
                if st.session_state.eval_start_time:  # Calculate final duration
                    final_elapsed = time.time() - st.session_state.eval_start_time
                    st.session_state.eval_duration_str = str(
                        timedelta(seconds=int(final_elapsed))
                    )
                # Check if stop event was set before declaring success/failure based on results/error
                # *** MODIFIED: Removed switch_to_view_results logic ***
                # if (
                #     not st.session_state.stop_event.is_set()
                #     and not st.session_state.evaluation_error
                #     and st.session_state.evaluation_results_paths
                # ):
                #     st.session_state.switch_to_view_results = (
                #         True  # Flag to switch tabs
                #     )
                st.rerun()  # Rerun to update UI state after completion/stop
                break  # Exit loop
            elif message.startswith("STATUS:"):
                status_text_update = message.split(":", 1)[1].strip()
                # Update status area (use appropriate placeholder/widget)
                if st.session_state.evaluation_error:  # Don't overwrite error status
                    pass
                else:
                    output_placeholder.info(f"Status: {status_text_update}")
            elif message.startswith("ERROR:"):
                error_msg = message.split(":", 1)[1].strip()
                st.session_state.evaluation_error = error_msg
                output_placeholder.error(
                    f"Status: Error - {error_msg}"  # Show error in status area
                )
                st.session_state.evaluation_running = False  # Stop on error
                st.rerun()
                break
            elif message.startswith("RESULTS:"):
                try:
                    paths = json.loads(message.split(":", 1)[1])
                    st.session_state.evaluation_results_paths = paths
                    logger.info(f"Received results paths: {paths}")
                    # Store the parent folder of the first result file
                    if paths:
                        try:
                            first_result_path = Path(paths[0])
                            st.session_state.newly_completed_run_folder = str(
                                first_result_path.parent
                            )
                            logger.info(
                                f"Stored newly completed run folder: {st.session_state.newly_completed_run_folder}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Error extracting parent folder from results path {paths[0]}: {e}"
                            )
                except json.JSONDecodeError:
                    logger.error("Failed to decode results paths from queue.")
                    st.session_state.evaluation_error = "Failed to parse results paths."
                    output_placeholder.error(
                        "Status: Error - Failed to parse results paths."
                    )
                    st.session_state.evaluation_running = False
                    st.rerun()
                    break
            elif message.startswith("LOG:"):
                log_entry = message.split(":", 1)[1].strip()
                st.session_state.last_run_output.append(log_entry)
            else:  # Assume it's a log message if no prefix
                st.session_state.last_run_output.append(message)

        except queue.Empty:
            break  # No more messages for now

    # Display accumulated logs
    log_container.code("\n".join(st.session_state.last_run_output), language="log")

    # If running, schedule a rerun to check the queue again
    if st.session_state.evaluation_running:
        time.sleep(0.5)  # Small delay to prevent excessive reruns
        st.rerun()


def main() -> None:
    """Main function to run the Streamlit app."""
    st.title("🧠 CogniBench Evaluation Runner")

    initialize_session_state()  # Ensure state is initialized

    # --- Main Content Area using Tabs ---
    # *** MODIFIED: Replaced sidebar radio with tabs ***
    tab_run, tab_view = st.tabs(["Run Evaluation", "View Results"])

    with tab_run:
        st.header("⚙️ Run New Evaluation")
        render_file_uploader()
        render_config_ui()

        if st.session_state.config_complete and st.session_state.uploaded_files_info:
            render_config_summary()
            # Placeholders for dynamic updates
            status_placeholder = st.empty()
            progress_placeholder = st.empty()
            log_placeholder = st.empty()
            render_evaluation_progress(
                status_placeholder, log_placeholder, progress_placeholder
            )
        elif not st.session_state.uploaded_files_info:
            st.warning("Upload data files to proceed.")
        else:  # Config incomplete
            st.warning("Complete the model and prompt configuration to proceed.")

    with tab_view:
        st.header("📊 View Evaluation Results")
        # Render folder selector directly within this tab
        render_results_selector()

        if st.session_state.selected_results_folders:
            # Load data based on selection
            # Use a separate cache key or disable caching if selection changes often
            # For simplicity, keeping the cache as is for now.
            results_df, summary_data = load_and_process_results(
                st.session_state.selected_results_folders
            )
            st.session_state.results_df = results_df  # Store in session state
            st.session_state.aggregated_summary = summary_data

            if results_df is not None:
                # Display sections using inner tabs within the "View Results" tab
                tab_summary, tab_plots, tab_details, tab_review = st.tabs(
                    [
                        "Summary Stats",
                        "Performance Plots",
                        "Detailed Results",
                        "Human Review",
                    ]
                )

                with tab_summary:
                    display_summary_stats(summary_data)

                with tab_plots:
                    display_performance_plots(results_df)
                    display_rubric_plots(results_df)  # Display rubric plots here too

                with tab_details:
                    display_results_table(
                        results_df
                    )  # This now contains the drill-down

                with tab_review:
                    display_human_review_tasks(results_df)

            else:
                st.warning(
                    "No data loaded from the selected folder(s). Ensure they contain valid '_final_results.json' files."
                )
        else:
            st.info(
                "Select one or more result folders above to view results."
            )  # Updated info message

    # --- Footer or other common elements ---
    st.markdown("---")
    st.caption("CogniBench Streamlit Interface")

    # --- Cleanup (Optional, kept commented) ---
    # Cleanup temp dir on script exit (might not always trigger reliably in Streamlit)
    # Consider alternative cleanup strategies if needed
    # def cleanup():
    #     if "temp_dir" in st.session_state and st.session_state.temp_dir:
    #         try:
    #             st.session_state.temp_dir.cleanup()
    #             logger.info("Temporary directory cleaned up.")
    #         except Exception as e:
    #             logger.error(f"Error cleaning up temp directory: {e}")
    # import atexit
    # atexit.register(cleanup)


if __name__ == "__main__":
    main()
