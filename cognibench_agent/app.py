"""
CogniBench Streamlit Application.

Provides a web-based user interface for running CogniBench evaluations,
configuring models and prompts, uploading data, viewing results, and
managing the evaluation process.
"""

import json
import logging
import queue
import subprocess  # Added import
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta  # Added datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Add project root to sys.path ---
APP_DIR = Path(__file__).parent
COGNIBENCH_ROOT = APP_DIR.parent
if str(COGNIBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(COGNIBENCH_ROOT))
# --- End sys.path modification ---

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml
from cognibench_agent.constants import AVAILABLE_MODELS, COLOR_MAP
from core.config import AppConfig
from core.evaluation_runner import run_batch_evaluation_core
from core.llm_clients.openai_client import clear_openai_cache
from core.log_setup import setup_logging

# --- Constants ---
BASE_CONFIG_PATH = COGNIBENCH_ROOT / "config.yaml"
PROMPT_TEMPLATES_DIR_ABS = COGNIBENCH_ROOT / "prompts"
STRUCTURING_TEMPLATES_DIR = PROMPT_TEMPLATES_DIR_ABS / "structuring"
JUDGING_TEMPLATES_DIR = PROMPT_TEMPLATES_DIR_ABS / "judging"
DATA_DIR = COGNIBENCH_ROOT / "data"  # Default data directory

# Constants moved to constants.py

# --- Logging Setup ---
logger = logging.getLogger("streamlit")
if "logging_setup_complete" not in st.session_state:
    setup_logging()
    st.session_state.logging_setup_complete = True
    logger = logging.getLogger("streamlit")  # Re-get logger after setup
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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Set default models if not already set
    if "structuring_model_select" not in st.session_state:
        st.session_state["structuring_model_select"] = "GPT-4.1"
        logger.info("Initialized structuring_model_select to GPT-4.1")
    if "judging_model_select" not in st.session_state:
        st.session_state["judging_model_select"] = "GPT-4.1"
        logger.info("Initialized judging_model_select to GPT-4.1")

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
    # struct_template_path = AVAILABLE_STRUCTURING_TEMPLATES.get( # Unused variable
    #     struct_template_name, "Not Selected"
    # )

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
- **Structuring Model:** `{struct_provider}` - `{struct_model_name}` (`{struct_model_id}`) | Prompt: `{struct_template_name}`
- **Judging Model:**     `{judge_provider}` - `{judge_model_name}` (`{judge_model_id}`) | Prompt: `{judge_template_name}`
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

        # --- Define persistent output directory (Moved Before override_config) ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")  # User requested format
        if st.session_state.uploaded_files_info:
            # Use the stem of the first uploaded file as the base name
            first_file_name = st.session_state.uploaded_files_info[0]["name"]
            base_name = Path(first_file_name).stem
            # Clean up common suffixes if needed
            for suffix in [".json", "_ingested", "_tasks"]:
                if base_name.endswith(suffix):
                    base_name = base_name[: -len(suffix)]
            folder_name = f"{base_name}_{timestamp}"
        else:
            # Fallback if somehow config is generated without files
            folder_name = f"StreamlitRun_{timestamp}"

        persistent_output_dir = DATA_DIR / folder_name
        logger.info(f"Setting persistent output directory: {persistent_output_dir}")
        # --- End Output Directory Definition ---

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
            # Output options using the persistent directory
            "output_options": {
                "output_dir": str(
                    persistent_output_dir
                ),  # Use calculated persistent path
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
        logger.exception("Error generating AppConfig:")
        return None


# Custom Log Handler to redirect logs to the Streamlit queue
class QueueLogHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord):
        log_entry = self.format(record)
        self.log_queue.put(
            f"LOG: {log_entry}"
        )  # Prefix to distinguish from status messages


def evaluation_worker(
    config: AppConfig, output_queue: queue.Queue, stop_event: threading.Event
):
    """Worker function to run the core evaluation in a separate thread."""
    # Setup queue logging
    log_handler = QueueLogHandler(output_queue)
    # Define format for logs shown in Streamlit UI
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    log_handler.setFormatter(formatter)

    # Get loggers to capture from (adjust names as needed based on module structure)
    core_logger = logging.getLogger(
        "core"
    )  # Captures core.evaluation_runner, core.workflow etc.
    backend_logger = logging.getLogger(
        "backend"
    )  # Capture backend logs if seen in errors
    # Add other loggers if necessary

    loggers_to_capture = [core_logger, backend_logger]

    # Add handler to loggers
    for lg in loggers_to_capture:
        lg.addHandler(log_handler)
        # Optionally set level if needed, e.g., lg.setLevel(logging.INFO)

    try:
        logger.info("Evaluation worker thread started (Queue logging enabled).")
        output_queue.put("INFO: Starting evaluation process...")

        # Redirect stdout/stderr? The core runner should use logging.
        # For now, assume core runner logs appropriately.

        # --- Step 1: Ingestion (Replicating script logic) ---
        output_dir = Path(
            config.output_options.output_dir
        )  # Ensure output_dir is defined
        output_queue.put("INFO: Starting ingestion step...")
        logger.info("Starting ingestion step for Streamlit run...")
        raw_file_paths_str = config.input_options.file_paths
        ingested_file_paths = []
        ingestion_success = True

        for raw_path_str in raw_file_paths_str:
            if stop_event.is_set():
                ingestion_success = False
                logger.warning("Stop event detected during ingestion loop.")
                break

            raw_path = Path(raw_path_str)
            ingestion_script_path = COGNIBENCH_ROOT / "scripts/ingest_rlhf_data.py"
            ingestion_command = [
                sys.executable,
                str(ingestion_script_path),
                str(raw_path),
                "--output-dir",  # Pass the specific batch output directory
                str(output_dir),
            ]
            output_queue.put(f"INFO: Running ingestion for {raw_path.name}...")
            logger.debug("Running ingestion command: %s", " ".join(ingestion_command))
            try:
                process = subprocess.run(
                    ingestion_command,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )
                ingested_path_str = process.stdout.strip()
                if not ingested_path_str:
                    raise ValueError("Ingestion script did not output a file path.")
                ingested_path = Path(ingested_path_str)
                if not ingested_path.is_file():
                    raise FileNotFoundError(
                        f"Ingested file path reported but not found: {ingested_path}"
                    )
                ingested_file_paths.append(str(ingested_path))
                output_queue.put(
                    f"INFO: Ingestion successful for {raw_path.name} -> {ingested_path.name}"
                )
                logger.info(
                    f"Ingestion successful for {raw_path.name} -> {ingested_path}"
                )
            except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
                error_msg = f"ERROR: Ingestion failed for {raw_path.name}: {e}"
                output_queue.put(error_msg)
                logger.error(error_msg, exc_info=True)
                if isinstance(e, subprocess.CalledProcessError):
                    if e.stdout:
                        logger.error("Ingestion stdout:\n%s", e.stdout)
                    if e.stderr:
                        logger.error("Ingestion stderr:\n%s", e.stderr)
                ingestion_success = False
                break  # Stop processing further files if one fails
            except Exception as e:
                error_msg = (
                    f"ERROR: Unexpected error during ingestion for {raw_path.name}: {e}"
                )
                output_queue.put(error_msg)
                logger.error(error_msg, exc_info=True)
                ingestion_success = False
                break

        if not ingestion_success:
            output_queue.put("ERROR: Evaluation aborted due to ingestion failure.")
            logger.error("Evaluation aborted due to ingestion failure.")
            # No need to call core runner if ingestion failed
            results_paths = None  # Ensure results_paths is None
        elif stop_event.is_set():
            output_queue.put("INFO: Evaluation cancelled after ingestion.")
            logger.info("Evaluation cancelled after ingestion.")
            results_paths = None  # Ensure results_paths is None
        else:
            # --- Step 2: Core Evaluation (using ingested paths) ---
            output_queue.put("INFO: Starting core evaluation step...")
            logger.info("Starting core evaluation step with ingested files...")
            # Update config with ingested paths BEFORE passing to core runner
            config.input_options.file_paths = ingested_file_paths
            logger.info(
                f"Updated config file paths for core runner: {ingested_file_paths}"
            )

            # Determine if structured evaluation should be used.
            use_structured = bool(config.structuring_settings)

            results_paths = run_batch_evaluation_core(
                config=config,  # Pass the UPDATED config
                output_dir=output_dir,
                use_structured=use_structured,
                stop_event=stop_event,
            )

        if stop_event.is_set():
            output_queue.put("INFO: Evaluation run cancelled by user.")
            logger.info("Evaluation run cancelled by user.")
        elif results_paths:
            output_queue.put(
                f"SUCCESS: Evaluation complete. Result files generated: {results_paths}"
            )
            output_queue.put(
                {"type": "results", "paths": results_paths}
            )  # Signal completion with paths
            logger.info(f"Evaluation successful. Results: {results_paths}")
        else:
            output_queue.put(
                "ERROR: Evaluation finished but no result paths were returned."
            )
            logger.error("Evaluation finished but no result paths were returned.")
            output_queue.put({"type": "error", "message": "No result paths returned."})

    except Exception as e:
        error_msg = f"ERROR: An error occurred during evaluation: {e}"
        output_queue.put(error_msg)
        output_queue.put({"type": "error", "message": str(e)})
        logger.exception("Exception in evaluation worker thread:")
    finally:
        # Remove handler from loggers
        for lg in loggers_to_capture:
            lg.removeHandler(log_handler)
        output_queue.put(None)  # Signal thread completion
        logger.info("Evaluation worker thread finished (Queue logging disabled).")


def start_core_evaluation():
    """Generates config and starts the evaluation worker thread."""
    st.session_state.evaluation_running = True
    st.session_state.eval_start_time = time.time()
    st.session_state.eval_duration_str = None
    st.session_state.last_run_output = []
    st.session_state.evaluation_results_paths = []
    st.session_state.results_df = None
    st.session_state.aggregated_summary = None
    st.session_state.evaluation_error = None
    st.session_state.stop_event.clear()  # Reset stop event

    logger.info("Attempting to start core evaluation...")
    app_config = generate_run_config()

    if app_config:
        st.session_state.output_queue = queue.Queue()  # Ensure fresh queue
        st.session_state.worker_thread = threading.Thread(
            target=evaluation_worker,
            args=(
                app_config,
                st.session_state.output_queue,
                st.session_state.stop_event,
            ),
            daemon=True,
        )
        st.session_state.worker_thread.start()
        logger.info("Evaluation worker thread started.")
    else:
        st.error("Failed to start evaluation due to configuration errors.")
        logger.error("Failed to start evaluation due to configuration errors.")
        st.session_state.evaluation_running = False
        st.session_state.eval_start_time = None


def stop_evaluation():
    """Signals the evaluation worker thread to stop."""
    if st.session_state.worker_thread and st.session_state.worker_thread.is_alive():
        logger.info("Attempting to stop evaluation worker thread...")
        st.session_state.stop_event.set()
        # Optionally add a timeout join here if needed
        # st.session_state.worker_thread.join(timeout=5)
        # if st.session_state.worker_thread.is_alive():
        #     logger.warning("Worker thread did not stop within timeout.")
        # else:
        #     logger.info("Worker thread stopped successfully.")
        # st.session_state.evaluation_running = False # Let the queue handler do this


# --- Results Processing and Display Functions ---


@st.cache_data(show_spinner="Loading results data...")
def load_and_process_results(
    absolute_results_paths: List[str],
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Loads data from _final_results.json files, processes into a DataFrame, and aggregates summary statistics."""
    logger.info(
        f"Attempting to load and process results from: {absolute_results_paths}"
    )
    all_results_data = []
    aggregated_summary = {
        "total_evaluations_processed": 0,
        "total_evaluation_time_seconds": 0.0,
        "total_structuring_api_calls": 0,
        "total_judging_api_calls": 0,
        "total_tasks_processed": 0,
        "average_time_per_model_seconds": {},
        "processed_files_count": 0,
        "failed_files": [],
    }

    for file_path_str in absolute_results_paths:
        try:
            file_path = Path(file_path_str)
            if not file_path.exists():
                logger.warning(f"Result file not found: {file_path_str}")
                aggregated_summary["failed_files"].append(
                    f"{file_path_str} (Not Found)"
                )
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                file_summary = data.get("summary", {})

                # Aggregate summary stats safely
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
                aggregated_summary["total_tasks_processed"] += file_summary.get(
                    "total_tasks_processed", 0
                )

                per_model_times = file_summary.get("average_time_per_model_seconds", {})
                if isinstance(per_model_times, dict):
                    # Simple merge/update - assumes distinct batches or accept latest time
                    aggregated_summary["average_time_per_model_seconds"].update(
                        per_model_times
                    )

                # Process 'results' list
                results_list = data.get("results", [])
                if not isinstance(results_list, list):
                    logger.error(
                        f"JSON file {file_path_str} has invalid 'results' list."
                    )
                    aggregated_summary["failed_files"].append(
                        f"{file_path_str} (Invalid Format)"
                    )
                    continue  # Skip processing tasks for this file

                for task in results_list:
                    task_id = task.get("task_id")
                    prompt = task.get("prompt")
                    ideal_response = task.get("ideal_response")
                    final_answer_gt = task.get("final_answer")
                    metadata = task.get("metadata", {})
                    subject = metadata.get("subject", "N/A")
                    complexity = metadata.get("complexity", "N/A")

                    for evaluation in task.get("evaluations", []):
                        model_id = evaluation.get("model_id")
                        model_response = evaluation.get("model_response")
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
                                            else:  # Handle cases where score might not be nested
                                                flat_judge_eval[
                                                    f"judge_rubric_{rubric_name}"
                                                ] = rubric_details
                                    else:  # Flatten other nested dicts like final_answer_structured
                                        for sub_key, sub_value in value.items():
                                            flat_judge_eval[
                                                f"judge_{key}_{sub_key}"
                                            ] = sub_value
                                else:
                                    flat_judge_eval[f"judge_{key}"] = value
                        else:
                            flat_judge_eval["judge_evaluation_raw"] = (
                                judge_eval  # Store raw if not dict
                            )

                        # Ensure aggregated score is string and title cased
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
                                # Add human eval fields if needed later
                            }
                        )
                aggregated_summary["processed_files_count"] += 1
                logger.info(f"Successfully processed results from: {file_path_str}")

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {file_path_str}: {e}")
            aggregated_summary["failed_files"].append(
                f"{file_path_str} (JSON Decode Error)"
            )
        except Exception as e:
            logger.exception(f"Unexpected error processing file {file_path_str}:")
            aggregated_summary["failed_files"].append(
                f"{file_path_str} (Processing Error: {e})"
            )

    if not all_results_data:
        logger.warning("No valid results data loaded.")
        return None, aggregated_summary  # Return summary even if no data

    try:
        df = pd.DataFrame(all_results_data)
        logger.info(f"Created DataFrame with shape: {df.shape}")
        # Basic type conversions or cleaning can happen here if needed
        # Example: df['complexity'] = df['complexity'].astype(str)
        return df, aggregated_summary
    except Exception:
        logger.exception("Error creating DataFrame from results data:")
        return None, aggregated_summary


def render_results_selector():
    """Renders UI to select existing result folders."""
    st.header("Load Existing Results (Optional)")
    st.info(
        "Select previously generated batch result folders (containing `_final_results.json`) to view."
    )

    # --- Folder Selection Logic ---
    try:
        available_folders = sorted(
            [d.name for d in DATA_DIR.iterdir() if d.is_dir() and "Batch-" in d.name],
            reverse=True,  # Show newest first
        )
    except FileNotFoundError:
        st.error(f"Data directory not found: {DATA_DIR}")
        available_folders = []

    if not available_folders:
        st.warning("No batch result folders found in the 'data' directory to load.")
        # Don't return early, still show the clear cache button
    else:
        # Only show selection if folders exist
        # The result is stored in st.session_state.selected_results_folders via the key
        st.multiselect(
            "Select result folder(s):",
            options=available_folders,
            key="selected_results_folders",
            help="Choose one or more folders to load results from.",
        )

        if st.button("Load Selected Results"):
            st.session_state.evaluation_results_paths = []
            st.session_state.results_df = None
            st.session_state.aggregated_summary = None
            st.session_state.evaluation_error = None
            st.session_state.last_run_output = [
                "INFO: Loading selected results..."
            ]  # Clear previous run output

            absolute_paths_to_load = []
            found_any = False
            for folder_name in st.session_state.selected_results_folders:
                folder_path = DATA_DIR / folder_name
                results_file = next(
                    folder_path.glob(f"{folder_name}_final_results.json"), None
                )
                if results_file and results_file.exists():
                    absolute_paths_to_load.append(str(results_file))
                    found_any = True
                    logger.info(f"Found results file to load: {results_file}")
                else:
                    st.warning(
                        f"Could not find '{folder_name}_final_results.json' in {folder_path}"
                    )
                    logger.warning(
                        f"Could not find '{folder_name}_final_results.json' in {folder_path}"
                    )

            if found_any:
                st.session_state.evaluation_results_paths = absolute_paths_to_load
                st.session_state.results_df, st.session_state.aggregated_summary = (
                    load_and_process_results(absolute_paths_to_load)
                )
                if st.session_state.results_df is None:
                    st.error("Failed to load or process data from selected folders.")
                    logger.error("load_and_process_results returned None DataFrame.")
                else:
                    st.success(
                        f"Successfully loaded data from {len(absolute_paths_to_load)} result file(s)."
                    )
                    logger.info(
                        f"Successfully loaded data. DataFrame shape: {st.session_state.results_df.shape}"
                    )
                st.rerun()
            elif st.session_state.selected_results_folders:
                st.error(
                    "No valid `_final_results.json` files found in the selected folder(s)."
                )
            else:
                st.info("No folders selected to load.")

    # --- Clear Cache Button (Always Visible) ---
    # Placed outside the folder loading logic

    # Add Clear Caches button
    if st.button("Clear Caches (Results & LLM)", key="clear_cache_button"):
        st.cache_data.clear()  # Clear Streamlit data cache
        clear_openai_cache()  # Clear OpenAI LLM cache
        # Clear related session state variables to force reload appearance
        st.session_state.results_df = None
        st.session_state.aggregated_summary = None
        st.session_state.evaluation_results_paths = []
        # st.session_state.selected_results_folders = [] # DO NOT directly modify widget state here
        st.success("Streamlit results cache and OpenAI LLM cache cleared.")
        time.sleep(1)  # Brief pause to show message
        st.rerun()


def display_summary_stats(summary_data: Dict[str, Any]):
    """Displays aggregated summary statistics."""
    st.subheader("📊 Overall Summary")
    if not summary_data:
        st.warning("No summary data available.")
        return

    total_evals = summary_data.get("total_evaluations_processed", 0)
    total_tasks = summary_data.get("total_tasks_processed", 0)
    total_time_sec = summary_data.get("total_evaluation_time_seconds", 0.0)
    struct_calls = summary_data.get("total_structuring_api_calls", 0)
    judge_calls = summary_data.get("total_judging_api_calls", 0)
    processed_files = summary_data.get("processed_files_count", 0)
    failed_files = summary_data.get("failed_files", [])

    total_time_fmt = (
        str(timedelta(seconds=int(total_time_sec))) if total_time_sec else "N/A"
    )

    st.metric("Processed Result Files", processed_files)
    if failed_files:
        with st.expander(
            f"⚠️ {len(failed_files)} File(s) Failed to Load/Process", expanded=False
        ):
            for f in failed_files:
                st.write(f"- `{f}`")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tasks Processed", f"{total_tasks:,}")
    col2.metric("Total Evaluations Processed", f"{total_evals:,}")
    col3.metric("Total Evaluation Time", total_time_fmt)

    col4, col5 = st.columns(2)
    col4.metric("Total Structuring API Calls", f"{struct_calls:,}")
    col5.metric("Total Judging API Calls", f"{judge_calls:,}")

    avg_times = summary_data.get("average_time_per_model_seconds", {})
    if avg_times:
        st.write("**Average Time per Model (seconds):**")
        avg_time_df = pd.DataFrame(
            list(avg_times.items()), columns=["Model", "Avg Time (s)"]
        )
        avg_time_df["Avg Time (s)"] = avg_time_df["Avg Time (s)"].round(2)
        st.dataframe(avg_time_df, use_container_width=True, hide_index=True)


def display_performance_plots(df: pd.DataFrame):
    """Displays plots for aggregated performance scores."""
    st.subheader("📈 Performance Overview")
    if df is None or df.empty or "aggregated_score" not in df.columns:
        st.warning("No data available for performance plots.")
        return

    # --- Overall Performance Distribution ---
    st.write("**Overall Aggregated Score Distribution**")
    available_scores = sorted(df["aggregated_score"].unique())
    # Ensure standard scores are present for consistent coloring, even if count is 0
    all_possible_scores = ["Pass", "Partial", "Fail", "Needs Review", "N/A"] + [
        s
        for s in available_scores
        if s not in ["Pass", "Partial", "Fail", "Needs Review", "N/A"]
    ]

    performance_counts = (
        df["aggregated_score"].value_counts().reindex(all_possible_scores, fill_value=0)
    )
    performance_counts_df = performance_counts.reset_index()
    performance_counts_df.columns = ["Aggregated Score", "Count"]

    # Use the global COLOR_MAP
    color_discrete_map = {
        k: v
        for k, v in COLOR_MAP.items()
        if k in performance_counts_df["Aggregated Score"].tolist()
    }

    try:
        fig_perf = px.bar(
            performance_counts_df,
            x="Aggregated Score",
            y="Count",
            title="Distribution of Aggregated Scores",
            color="Aggregated Score",
            color_discrete_map=color_discrete_map,
            text_auto=True,
        )
        fig_perf.update_layout(xaxis_title=None, yaxis_title="Number of Evaluations")
        st.plotly_chart(fig_perf, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating performance plot: {e}")
        logger.error(f"Error generating performance plot: {e}")

    # --- Performance by Model ---
    st.write("**Aggregated Scores by Model**")
    try:
        model_perf_counts = (
            df.groupby(["model_id", "aggregated_score"]).size().unstack(fill_value=0)
        )
        # Ensure all standard scores are columns for consistent stacking/coloring
        for score in all_possible_scores:
            if score not in model_perf_counts.columns:
                model_perf_counts[score] = 0
        # Order columns consistently
        model_perf_counts = model_perf_counts[all_possible_scores]

        fig_model_perf = px.bar(
            model_perf_counts,
            title="Aggregated Scores per Model",
            barmode="group",  # Changed from "stack" to "group"
            color_discrete_map=color_discrete_map,
            text_auto=True,
        )
        fig_model_perf.update_layout(
            xaxis_title="Model ID",
            yaxis_title="Number of Evaluations",
            legend_title="Aggregated Score",
        )
        st.plotly_chart(fig_model_perf, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating performance by model plot: {e}")
        logger.error(f"Error generating performance by model plot: {e}")


def display_rubric_plots(df: pd.DataFrame):
    """Displays plots for rubric scores."""
    st.subheader("📐 Rubric Score Analysis")
    rubric_cols = [
        col
        for col in df.columns
        if col.startswith("judge_rubric_") and col.endswith("_score")
    ]

    if df is None or df.empty or not rubric_cols:
        st.warning("No rubric score data available for plotting.")
        return

    logger.info(f"Found rubric score columns: {rubric_cols}")

    # --- Overall Rubric Score Distribution ---
    st.write("**Overall Rubric Score Distribution**")  # Title updated below
    try:
        # Melt the DataFrame (as before)
        rubric_melted = df.melt(
            id_vars=["task_id", "model_id"],
            value_vars=rubric_cols,
            var_name="Rubric Criterion",
            value_name="Score",
        )
        rubric_melted["Rubric Criterion"] = (
            rubric_melted["Rubric Criterion"]
            .str.replace("judge_rubric_", "")
            .str.replace("_score", "")
            .str.replace("_", " ")
            .str.title()
        )
        rubric_melted["Score"] = (
            rubric_melted["Score"].fillna("N/A").astype(str).str.title()
        )

        # --- Add Model Filter Dropdown ---
        all_models_option = "All Models"
        available_models = sorted(rubric_melted["model_id"].unique())
        model_options = [all_models_option] + available_models
        selected_model_filter = st.selectbox(
            "Filter by Model:",
            options=model_options,
            index=0,  # Default to "All Models"
        )

        # Filter data if a specific model is selected
        filtered_rubric_data = rubric_melted
        plot_title = "Overall Rubric Score Distribution (All Models)"
        if selected_model_filter != all_models_option:
            filtered_rubric_data = rubric_melted[
                rubric_melted["model_id"] == selected_model_filter
            ]
            plot_title = (
                f"Overall Rubric Score Distribution (Model: {selected_model_filter})"
            )

        if filtered_rubric_data.empty:
            st.warning(f"No rubric data found for model: {selected_model_filter}")
        else:
            # Define expected categories and colors (based on potentially filtered data)
            rubric_score_categories = ["Yes", "No", "Partial", "N/A"] + sorted(
                [
                    s
                    for s in filtered_rubric_data["Score"].unique()
                    if s not in ["Yes", "No", "Partial", "N/A"]
                ]
            )
            rubric_color_map = {
                k: v for k, v in COLOR_MAP.items() if k in rubric_score_categories
            }

            # Calculate counts for the plot based on filtered data
            rubric_counts = (
                filtered_rubric_data.groupby(["Rubric Criterion", "Score"])
                .size()
                .unstack(fill_value=0)
            )
            # Ensure all categories are present as columns
            for cat in rubric_score_categories:
                if cat not in rubric_counts.columns:
                    rubric_counts[cat] = 0
            rubric_counts = rubric_counts[rubric_score_categories]  # Order columns

            fig_rubric = px.bar(
                rubric_counts,
                title=plot_title,  # Use dynamic title
                barmode="group",
                color_discrete_map=rubric_color_map,
                text_auto=True,
            )
            fig_rubric.update_layout(
                xaxis_title="Rubric Criterion",
                yaxis_title="Number of Evaluations",
                legend_title="Score",
            )
            st.plotly_chart(fig_rubric, use_container_width=True)

    except Exception as e:
        st.error(f"Error generating overall rubric plot: {e}")
        logger.exception("Error generating overall rubric plot:")

    # --- Rubric Scores by Model ---
    st.write("**Rubric Scores by Model**")
    try:
        # Use the already melted data
        rubric_model_counts = (
            rubric_melted.groupby(["model_id", "Rubric Criterion", "Score"])
            .size()
            .unstack(fill_value=0)
        )

        # Ensure all score categories are present as columns
        for cat in rubric_score_categories:
            if cat not in rubric_model_counts.columns:
                rubric_model_counts[cat] = 0
        rubric_model_counts = rubric_model_counts[
            rubric_score_categories
        ]  # Order columns

        # --- Rubric Scores by Model (with "All Criteria" option) ---
        all_criteria_option = "All Criteria"
        available_criteria = [all_criteria_option] + sorted(
            rubric_melted["Rubric Criterion"].unique()
        )
        selected_criterion = st.selectbox(
            "Select Rubric Criterion to view by Model:",
            options=available_criteria,
            index=0,  # Default to "All Criteria"
        )

        if selected_criterion:
            if selected_criterion == all_criteria_option:
                # Aggregate counts across all criteria for each model and score
                all_criteria_counts = (
                    rubric_melted.groupby(["model_id", "Score"])
                    .size()
                    .unstack(fill_value=0)
                )
                # Ensure all score categories are present
                for cat in rubric_score_categories:
                    if cat not in all_criteria_counts.columns:
                        all_criteria_counts[cat] = 0
                all_criteria_counts = all_criteria_counts[
                    rubric_score_categories
                ]  # Order columns

                fig_rubric_model = px.bar(
                    all_criteria_counts,
                    title="Overall Rubric Scores by Model (All Criteria)",
                    barmode="group",
                    color_discrete_map=rubric_color_map,
                    text_auto=True,
                )
                fig_rubric_model.update_layout(
                    xaxis_title="Model ID",
                    yaxis_title="Total Count Across All Criteria",
                    legend_title="Score",
                )
            else:
                # Filter for the specific selected criterion (existing logic)
                criterion_data = rubric_model_counts.loc[
                    rubric_model_counts.index.get_level_values("Rubric Criterion")
                    == selected_criterion
                ]
                criterion_data = criterion_data.reset_index(
                    level="Rubric Criterion", drop=True
                )

                fig_rubric_model = px.bar(
                    criterion_data,
                    title=f"Scores for '{selected_criterion}' by Model",
                    barmode="group",
                    color_discrete_map=rubric_color_map,
                    text_auto=True,
                )
                fig_rubric_model.update_layout(
                    xaxis_title="Model ID",
                    yaxis_title="Number of Evaluations",
                    legend_title="Score",
                )

            st.plotly_chart(fig_rubric_model, use_container_width=True)

    except Exception as e:
        st.error(f"Error generating rubric scores by model plot: {e}")
        logger.exception("Error generating rubric scores by model plot:")


def display_results_table(df: pd.DataFrame):
    """Displays the detailed results in a filterable table and allows drilling down."""
    st.subheader("🔍 Detailed Results Explorer")
    if df is None or df.empty:
        st.warning("No results data to display.")
        return

    # --- Filtering ---
    st.write("**Filter Data**")
    col_filter1, col_filter2, col_filter3 = st.columns(3)

    # Filter by Model
    available_models = sorted(df["model_id"].unique())
    selected_models = col_filter1.multiselect(
        "Filter by Model:", options=available_models, default=available_models
    )

    # Filter by Aggregated Score
    available_scores = sorted(df["aggregated_score"].unique())
    selected_scores = col_filter2.multiselect(
        "Filter by Aggregated Score:",
        options=available_scores,
        default=available_scores,
    )

    # Filter by Subject (if available)
    if "subject" in df.columns:
        available_subjects = sorted(df["subject"].unique())
        selected_subjects = col_filter3.multiselect(
            "Filter by Subject:", options=available_subjects, default=available_subjects
        )
    else:
        selected_subjects = []  # No subject column

    # Filter by Needs Review (if judge_needs_review exists)
    needs_review_col = "judge_needs_review"  # Adjust if column name differs
    review_options = ["All", "Yes", "No"]
    selected_review_status = "All"
    if needs_review_col in df.columns:
        selected_review_status = st.radio(
            "Filter by 'Needs Review' Flag:",
            options=review_options,
            index=0,
            horizontal=True,
        )

    # Apply filters
    filtered_df = df[df["model_id"].isin(selected_models)]
    filtered_df = filtered_df[filtered_df["aggregated_score"].isin(selected_scores)]
    if selected_subjects and "subject" in df.columns:
        filtered_df = filtered_df[filtered_df["subject"].isin(selected_subjects)]
    if selected_review_status != "All" and needs_review_col in filtered_df.columns:
        # Handle potential boolean/string representations
        review_bool = True if selected_review_status == "Yes" else False
        # Be robust to different types in the column
        try:
            filtered_df = filtered_df[
                filtered_df[needs_review_col].astype(bool) == review_bool
            ]
        except Exception:  # Fallback for string comparison if casting fails
            filtered_df = filtered_df[
                filtered_df[needs_review_col].astype(str).str.lower()
                == selected_review_status.lower()
            ]

    st.write(f"Displaying **{len(filtered_df)}** of **{len(df)}** evaluations.")

    if filtered_df.empty:
        st.warning("No data matches the current filters.")
        return

    # --- Display Table ---
    st.write("**Filtered Evaluations Table**")
    # Select and order columns for display
    cols_to_show = [
        "task_id",
        "model_id",
        "aggregated_score",
        "subject",
        "complexity",
        # Add key rubric scores or flags if desired
        "judge_needs_review",
        "judge_is_complete",
        "judge_final_answer_structured_matches_ground_truth",
    ]
    # Filter out columns that don't exist in the dataframe
    cols_to_show = [col for col in cols_to_show if col in filtered_df.columns]

    st.dataframe(filtered_df[cols_to_show], use_container_width=True, hide_index=True)

    # --- Drill Down ---
    st.write("**Drill Down into Specific Task/Model**")
    available_tasks = sorted(filtered_df["task_id"].unique())
    if not available_tasks:
        st.info("Select filters above to enable drill-down.")
        return

    selected_task_id_detail = st.selectbox("Select Task ID:", options=available_tasks)

    if selected_task_id_detail:
        task_df = filtered_df[filtered_df["task_id"] == selected_task_id_detail]
        models_for_task = sorted(task_df["model_id"].unique())
        selected_model_id_detail = st.selectbox(
            "Select Model ID:", options=models_for_task
        )

        if selected_model_id_detail:
            detail_row = task_df[task_df["model_id"] == selected_model_id_detail].iloc[
                0
            ]

            st.markdown("---")
            st.subheader(
                f"Details for Task: `{selected_task_id_detail}` | Model: `{selected_model_id_detail}`"
            )

            # Display key info
            st.markdown(
                f"**Aggregated Score:** {detail_row.get('aggregated_score', 'N/A')}"
            )
            # Add more fields as needed: subject, complexity, etc.

            # Display Prompt & Responses in columns
            col_p, col_i, col_m = st.columns(3)
            with col_p:
                st.write("**Prompt**")
                st.text_area(
                    "Prompt Content",
                    value=detail_row.get("prompt", ""),
                    height=300,
                    key=f"prompt_{selected_task_id_detail}_{selected_model_id_detail}",
                    disabled=True,
                )
            with col_i:
                st.write("**Ideal Response**")
                st.text_area(
                    "Ideal Response Content",
                    value=detail_row.get("ideal_response", ""),
                    height=300,
                    key=f"ideal_{selected_task_id_detail}_{selected_model_id_detail}",
                    disabled=True,
                )
            with col_m:
                st.write("**Model Response**")
                st.text_area(
                    "Model Response Content",
                    value=detail_row.get("model_response", ""),
                    height=300,
                    key=f"model_{selected_task_id_detail}_{selected_model_id_detail}",
                    disabled=True,
                )

            # Display Judge Evaluation Details
            st.write("**Judge Evaluation Details**")
            judge_details = {}
            rubric_details = {}
            for col, value in detail_row.items():
                if col.startswith("judge_") and not col.startswith("judge_rubric_"):
                    judge_details[col.replace("judge_", "")] = value
                elif col.startswith("judge_rubric_"):
                    # Group score and justification
                    parts = col.replace("judge_rubric_", "").split("_")
                    is_score = parts[-1] == "score"
                    is_justification = parts[-1] == "justification"
                    criterion_name = " ".join(
                        parts[:-1] if (is_score or is_justification) else parts
                    ).title()

                    if criterion_name not in rubric_details:
                        rubric_details[criterion_name] = {
                            "score": "N/A",
                            "justification": "N/A",
                        }

                    if is_score:
                        rubric_details[criterion_name]["score"] = (
                            str(value).title() if pd.notna(value) else "N/A"
                        )
                    elif is_justification:
                        rubric_details[criterion_name]["justification"] = (
                            value if pd.notna(value) else "N/A"
                        )
                    elif (
                        not is_score and not is_justification
                    ):  # Handle cases where it's just the score directly under the name
                        rubric_details[criterion_name]["score"] = (
                            str(value).title() if pd.notna(value) else "N/A"
                        )

            # Display general judge fields first
            st.json(judge_details, expanded=False)

            # Display rubric scores neatly
            st.write("**Rubric Scores & Justifications**")
            if rubric_details:
                for criterion, details in rubric_details.items():
                    score_color = COLOR_MAP.get(
                        details["score"], "#6c757d"
                    )  # Default grey
                    st.markdown(
                        f"- **{criterion}:** <span style='color:{score_color}; font-weight:bold;'>{details['score']}</span>",
                        unsafe_allow_html=True,
                    )
                    if details["justification"] != "N/A":
                        st.caption(f"  Justification: {details['justification']}")
            else:
                st.info("No rubric score details found.")


def render_evaluation_progress(
    output_lines: List[str], is_running: bool, duration_str: Optional[str]
):
    """Displays the progress/output of the evaluation run."""
    st.header("Run Evaluation & View Results")
    st.markdown("---")

    run_button_disabled = (
        st.session_state.evaluation_running
        or not st.session_state.config_complete
        or not st.session_state.uploaded_files_info
    )
    run_button_label = (
        "Running Evaluation..."
        if st.session_state.evaluation_running
        else "Run Evaluation"
    )

    col_run, col_stop, col_status = st.columns([2, 1, 3])

    with col_run:
        if st.button(run_button_label, disabled=run_button_disabled, type="primary"):
            start_core_evaluation()
            # Rerun immediately to show "Running..." state and progress area
            st.rerun()

    with col_stop:
        if st.button("Stop Run", disabled=not st.session_state.evaluation_running):
            stop_evaluation()
            st.info("Stop signal sent. Evaluation will halt shortly.")
            # No rerun here, let the output handler update status

    with col_status:
        if st.session_state.evaluation_running:
            st.info("⏳ Evaluation in progress...")
        elif (
            st.session_state.eval_start_time and not st.session_state.evaluation_error
        ):  # Completed successfully
            st.success(f"✅ Evaluation completed in {duration_str}.")
        elif st.session_state.evaluation_error:
            st.error(f"❌ Evaluation failed: {st.session_state.evaluation_error}")
        elif (
            st.session_state.evaluation_results_paths
            and not st.session_state.eval_start_time
        ):  # Loaded results
            st.success("✅ Results loaded successfully.")
        # else: Initial state or after clearing

    # --- Progress/Output Area (within an expander) ---
    if is_running or output_lines:
        with st.expander("Run Output / Log", expanded=is_running):  # Expand if running
            progress_area = st.container(
                height=300
            )  # Use container for scrollable area
            log_text = "\n".join(output_lines)
            progress_area.code(log_text, language="log")


# --- Main App Logic ---


def main() -> None:
    """Main function to run the Streamlit app."""
    st.title("CogniBench Evaluation Runner")

    initialize_session_state()

    # --- UI Sections ---
    render_file_uploader()
    st.markdown("---")
    render_config_ui()
    st.markdown("---")
    render_config_summary()
    st.markdown("---")

    # --- Evaluation Execution and Progress ---
    render_evaluation_progress(
        st.session_state.last_run_output,
        st.session_state.evaluation_running,
        st.session_state.eval_duration_str,
    )

    # --- Handle Evaluation Output Queue ---
    if st.session_state.evaluation_running:
        try:
            while not st.session_state.output_queue.empty():
                item = st.session_state.output_queue.get_nowait()
                if item is None:  # End signal
                    st.session_state.evaluation_running = False
                    if st.session_state.eval_start_time:
                        duration = time.time() - st.session_state.eval_start_time
                        st.session_state.eval_duration_str = str(
                            timedelta(seconds=int(duration))
                        )
                    logger.info("Evaluation thread signaled completion.")
                    # Load results automatically if paths were received
                    if (
                        st.session_state.evaluation_results_paths
                        and not st.session_state.evaluation_error
                    ):
                        logger.info(
                            "Attempting to auto-load results after successful run."
                        )
                        (
                            st.session_state.results_df,
                            st.session_state.aggregated_summary,
                        ) = load_and_process_results(
                            st.session_state.evaluation_results_paths
                        )
                        if st.session_state.results_df is None:
                            st.session_state.evaluation_error = (
                                "Failed to load results after run."
                            )
                            logger.error(
                                "Auto-load failed after successful run signal."
                            )
                        else:
                            logger.info("Auto-load successful.")
                    st.rerun()  # Rerun to update UI after completion/loading
                    break  # Exit loop after handling None
                elif isinstance(item, dict):  # Structured message (results or error)
                    if item.get("type") == "results":
                        st.session_state.evaluation_results_paths = item.get(
                            "paths", []
                        )
                        logger.info(
                            f"Received result paths: {st.session_state.evaluation_results_paths}"
                        )
                    elif item.get("type") == "error":
                        st.session_state.evaluation_error = item.get(
                            "message", "Unknown error from worker."
                        )
                        st.session_state.evaluation_running = False  # Stop on error
                        logger.error(
                            f"Received error from worker: {st.session_state.evaluation_error}"
                        )
                        st.rerun()  # Update UI to show error
                        break
                elif isinstance(item, str):  # Log message
                    st.session_state.last_run_output.append(item)
                    # Limit log lines displayed to prevent browser slowdown
                    max_log_lines = 500
                    if len(st.session_state.last_run_output) > max_log_lines:
                        st.session_state.last_run_output = (
                            st.session_state.last_run_output[-max_log_lines:]
                        )
                    # No rerun here, let the progress renderer handle display updates periodically

            # Add a small delay and rerun to periodically update the output display
            if st.session_state.evaluation_running:
                time.sleep(0.5)
                st.rerun()

        except queue.Empty:
            # Queue is empty, wait for next cycle
            if st.session_state.evaluation_running:
                time.sleep(0.5)
                st.rerun()
        except Exception as e:
            st.error(f"Error processing evaluation output queue: {e}")
            logger.exception("Error processing output queue:")
            st.session_state.evaluation_running = False
            st.session_state.evaluation_error = str(e)
            st.rerun()

    st.markdown("---")

    # --- Results Loading and Display ---
    render_results_selector()  # Allow loading previous results

    if st.session_state.results_df is not None:
        st.header("4. Evaluation Results")
        display_summary_stats(st.session_state.aggregated_summary)
        st.markdown("---")
        display_performance_plots(st.session_state.results_df)
        st.markdown("---")
        display_rubric_plots(st.session_state.results_df)
        st.markdown("---")
        display_results_table(st.session_state.results_df)
    elif (
        st.session_state.evaluation_results_paths
        and not st.session_state.evaluation_running
        and not st.session_state.evaluation_error
    ):
        # This case might happen if loading failed silently or cache needs invalidation
        st.info("Attempting to load results data...")
        # Force reload if df is None but paths exist
        st.session_state.results_df, st.session_state.aggregated_summary = (
            load_and_process_results(st.session_state.evaluation_results_paths)
        )
        if st.session_state.results_df is None:
            st.error("Failed to load results data.")
        st.rerun()  # Rerun to display loaded data or error


if __name__ == "__main__":
    main()
