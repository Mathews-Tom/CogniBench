"""
CogniBench Streamlit Application.

Provides a web-based user interface for running CogniBench evaluations,
configuring models and prompts, uploading data, viewing results, and
managing the evaluation process.
"""

import asyncio
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
from core.constants import (
    APP_DIR,
    AVAILABLE_MODELS,
    BASE_CONFIG_PATH,
    COGNIBENCH_ROOT,
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

# --- Logging Setup ---
logger = logging.getLogger("frontend")
logger = logging.getLogger("frontend")
if "logging_setup_complete" not in st.session_state:
    setup_logging(log_level=logging.DEBUG)
    st.session_state.logging_setup_complete = True
    logger = logging.getLogger("frontend")
    logger = logging.getLogger("frontend")
    logger.setLevel(logging.DEBUG)
    logger.info("Logger 'frontend' level forced to DEBUG.")
    logger.info("Logger 'frontend' level forced to DEBUG.")
    logger.info("Initial logging setup complete.")
    logger.info("Streamlit app started.")

# --- Page Config ---
st.set_page_config(layout="wide", page_title="CogniBench Runner")

# --- Helper Functions ---

def get_templates(directory: Path) -> Dict[str, str]:
    """Scans a directory for .txt files and returns a dictionary of name: path."""
    try:
        # Find all .txt files
        txt_files = [
            f for f in directory.iterdir() if f.is_file() and f.suffix == ".txt"
        ]
        # Sort files by modification time, newest first
        txt_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        # Return dictionary with stem as key and path as value
        return {f.stem: str(f) for f in txt_files}
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
        "uploaded_files_info": [],
        "uploaded_files_info": [],
        "last_uploaded_files_key": None,
        "structuring_provider_select": list(AVAILABLE_MODELS.keys())[0],
        "structuring_api_key_input": "",
        "judging_provider_select": list(AVAILABLE_MODELS.keys())[0],
        "judging_api_key_input": "",
        "structuring_template_select": list(AVAILABLE_STRUCTURING_TEMPLATES.keys())[0] if AVAILABLE_STRUCTURING_TEMPLATES else None,
        "judging_template_select": list(AVAILABLE_JUDGING_TEMPLATES.keys())[0] if AVAILABLE_JUDGING_TEMPLATES else None,
        "show_structuring": False,
        "show_judging": False,
        "show_config": False,
        "evaluation_running": False,
        "evaluation_results_paths": [],
        "last_run_output": [],
        "evaluation_results_paths": [],
        "last_run_output": [],
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
        "newly_completed_run_folder": None,
        "newly_completed_run_folder": None,
        "view_selected_task_id": None,
        "view_selected_model_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "structuring_model_select" not in st.session_state:
        st.session_state["structuring_model_select"] = DEFAULT_STRUCTURING_MODEL
        logger.info(f"Initialized structuring_model_select to {DEFAULT_STRUCTURING_MODEL}")
    if "judging_model_select" not in st.session_state:
        st.session_state["judging_model_select"] = DEFAULT_JUDGING_MODEL
        logger.info(f"Initialized judging_model_select to {DEFAULT_JUDGING_MODEL}")

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

        if st.session_state.last_uploaded_files_key != current_upload_key or not st.session_state.uploaded_files_info:
            logger.info(f"Processing {len(uploaded_files)} uploaded files...")
            st.session_state.uploaded_files_info = []
            temp_dir = st.session_state.temp_dir_path
            for uploaded_file in uploaded_files:
                try:
                    dest_path = temp_dir / uploaded_file.name
                    with open(dest_path, "wb") as f: f.write(uploaded_file.getvalue())
                    st.session_state.uploaded_files_info.append({"name": uploaded_file.name, "path": str(dest_path)})
                    logger.info(f"Saved uploaded file to temporary path: {dest_path}")
                except Exception as e:
                    st.error(f"Error saving file {uploaded_file.name}: {e}")
                    logger.error(f"Error saving file {uploaded_file.name}: {e}")

            st.session_state.last_uploaded_files_key = current_upload_key
            logger.info(f"Finished processing uploads. {len(st.session_state.uploaded_files_info)} files ready.")
            st.rerun()

        st.write(f"Using {len(st.session_state.uploaded_files_info)} uploaded file(s):")
        for file_info in st.session_state.uploaded_files_info: st.write(f"- {file_info['name']}")
    else:
        st.info("Please upload at least one batch file.")
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
            st.selectbox("Provider", options=list(AVAILABLE_MODELS.keys())[0], key="structuring_provider_select")
            st.selectbox("Model", options=list(AVAILABLE_MODELS[st.session_state.structuring_provider_select].keys()), key="structuring_model_select")
            st.text_input("API Key (Optional)", type="password", placeholder="Leave blank to use environment variable", key="structuring_api_key_input")
        with col_judging:
            st.subheader("Judging Model")
            st.selectbox("Provider", options=list(AVAILABLE_MODELS.keys())[0], key="judging_provider_select")
            st.selectbox("Model", options=list(AVAILABLE_MODELS[st.session_state.judging_provider_select].keys()), key="judging_model_select")
            st.text_input("API Key (Optional)", type="password", placeholder="Leave blank to use environment variable", key="judging_api_key_input")

    with st.expander("Prompt Configurations", expanded=False):
        col_prompt1, col_prompt2 = st.columns(2)
        with col_prompt1:
            st.selectbox("Structuring Prompt Template", options=list(AVAILABLE_STRUCTURING_TEMPLATES.keys())[0] if AVAILABLE_STRUCTURING_TEMPLATES else None, key="structuring_template_select", help="Select the template file for structuring.")
            if st.button("View Structuring Prompt"): st.session_state.show_structuring = not st.session_state.show_structuring
        with col_prompt2:
            st.selectbox("Judging Prompt Template", options=list(AVAILABLE_JUDGING_TEMPLATES.keys())[0] if AVAILABLE_JUDGING_TEMPLATES else None, key="judging_template_select", help="Select the template file for judging.")
            if st.button("View Judging Prompt"): st.session_state.show_judging = not st.session_state.show_judging
        if st.button("View Base Config.yaml"): st.session_state.show_config = not st.session_state.show_config

    if st.session_state.show_structuring and st.session_state.structuring_template_select:
        with st.expander("Structuring Prompt Content", expanded=True):
            try: path = AVAILABLE_STRUCTURING_TEMPLATES[st.session_state.structuring_template_select]; content = Path(path).read_text(); st.code(content, language="text")
            except Exception as e: st.error(f"Error reading structuring prompt: {e}")
    if st.session_state.show_judging and st.session_state.judging_template_select:
        with st.expander("Judging Prompt Content", expanded=True):
            try: path = AVAILABLE_JUDGING_TEMPLATES[st.session_state.judging_template_select]; content = Path(path).read_text(); st.code(content, language="text")
            except Exception as e: st.error(f"Error reading judging prompt: {e}")
    if st.session_state.show_config:
        with st.expander("Base Config.yaml Content", expanded=True):
            try: content = BASE_CONFIG_PATH.read_text(); st.code(content, language="yaml")
            except Exception as e: st.error(f"Error reading config.yaml: {e}")

    st.session_state.config_complete = all([st.session_state.structuring_provider_select, st.session_state.structuring_model_select, st.session_state.judging_provider_select, st.session_state.judging_model_select, st.session_state.structuring_template_select, st.session_state.judging_template_select])
    if st.session_state.config_complete: st.success("✅ Configuration is complete.")
    else: st.warning("Configuration is incomplete. Please ensure all model and prompt fields are selected.")

def render_config_summary():
    """Displays a summary of the selected configuration."""
    st.subheader("Current Configuration Summary")
    struct_provider = st.session_state.structuring_provider_select
    struct_model_name = st.session_state.structuring_model_select
    struct_model_id = AVAILABLE_MODELS.get(struct_provider, {}).get(struct_model_name, "N/A")
    struct_template_name = st.session_state.structuring_template_select
    judge_provider = st.session_state.judging_provider_select
    judge_model_name = st.session_state.judging_model_select
    judge_model_id = AVAILABLE_MODELS.get(judge_provider, {}).get(judge_model_name, "N/A")
    judge_template_name = st.session_state.judging_template_select
    st.markdown(f"""
- **Structuring Model:** `{struct_provider}` - `{struct_model_name}` (`{struct_model_id}`)     |     Prompt: `{struct_template_name}`
- **Judging Model:**     `{judge_provider}` - `{judge_model_name}` (`{judge_model_id}`)     |     Prompt: `{judge_template_name}`
    """)

# --- Core Logic Functions ---

def generate_run_config() -> Optional[AppConfig]:
    """Generates the AppConfig object based on UI selections."""
    if not st.session_state.config_complete or not st.session_state.uploaded_files_info:
        st.error("Cannot generate config. Ensure files are uploaded and configuration is complete.")
        return None
    try:
        with open(BASE_CONFIG_PATH, "r") as f: base_config_data = yaml.safe_load(f)
        struct_provider = st.session_state.structuring_provider_select
        struct_model_name = st.session_state.structuring_model_select
        struct_model_id = AVAILABLE_MODELS[struct_provider][struct_model_name]
        judge_provider = st.session_state.judging_provider_select
        judge_model_name = st.session_state.judging_model_select
        judge_model_id = AVAILABLE_MODELS[judge_provider][judge_model_name]
        struct_template_name = st.session_state.structuring_template_select
        struct_template_path = AVAILABLE_STRUCTURING_TEMPLATES[struct_template_name]
        judge_template_name = st.session_state.judging_template_select
        judge_template_path = AVAILABLE_JUDGING_TEMPLATES[judge_template_name]
        struct_api_key = st.session_state.structuring_api_key_input or None
        judge_api_key = st.session_state.judging_api_key_input or None
        input_file_paths = [f["path"] for f in st.session_state.uploaded_files_info]

        input_stems = sorted([Path(p).stem for p in input_file_paths])
        cleaned_stems = []
        common_suffixes = ["_ingested", "_tasks"]
        for stem in input_stems:
            for suffix in common_suffixes:
                if stem.endswith(suffix): stem = stem[: -len(suffix)]; break
            cleaned_stems.append(stem)
        max_len_per_stem = 20; shortened_stems = [s[:max_len_per_stem] for s in cleaned_stems]
        combined_stem = "_".join(shortened_stems); max_total_stem_len = 60
        if len(combined_stem) > max_total_stem_len: combined_stem = combined_stem[:max_total_stem_len] + "_etc"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M"); run_specific_dir_name = f"{combined_stem}_{timestamp}"
        run_output_dir = DATA_DIR / run_specific_dir_name; logger.info(f"Generated run-specific output directory: {run_output_dir}")

        override_config = {
            "input_options": {"file_paths": input_file_paths},
            "structuring_settings": {"llm_client": {"provider": struct_provider.lower(), "model": struct_model_id, "api_key": struct_api_key}, "prompt_template_path": struct_template_path, "structuring_model": struct_model_id},
            "evaluation_settings": {"llm_client": {"provider": judge_provider.lower(), "model": judge_model_id, "api_key": judge_api_key}, "prompt_template_path": judge_template_path, "judge_model": judge_model_id},
            "output_options": {"output_dir": str(run_output_dir), "save_evaluations_jsonl": True, "save_evaluations_json": True, "save_final_results_json": True},
        }
        def deep_update(source: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in overrides.items():
                if isinstance(value, dict) and key in source and isinstance(source[key], dict): deep_update(source[key], value)
                else: source[key] = value
            return source
        final_config_data = deep_update(base_config_data, override_config)
        config = AppConfig(**final_config_data)

        # Ensure the run-specific output directory exists (Convert to Path first)
        output_dir_path = Path(config.output_options.output_dir) # FIX: Convert to Path
        output_dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir_path}")

        return config
    except Exception as e:
        st.error(f"Error generating configuration: {e}")
        logger.error(f"Configuration generation error: {traceback.format_exc()}")
        logger.error(f"Configuration generation error: {traceback.format_exc()}")
        return None

# --- Logging Handler for Streamlit ---
class QueueLogHandler(logging.Handler):
    """Sends log records to a queue."""
    def __init__(self, log_queue: queue.Queue): super().__init__(); self.log_queue = log_queue
    def emit(self, record: logging.LogRecord): self.log_queue.put(self.format(record))

# --- Evaluation Worker Thread ---
def evaluation_worker(config: AppConfig, output_queue: queue.Queue, stop_event: threading.Event):
    """Runs the core evaluation logic in a separate thread."""
    try:
        queue_handler = QueueLogHandler(output_queue); formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"); queue_handler.setFormatter(formatter)
        queue_handler.setLevel(logging.INFO) # FIX: Set handler level to filter logs sent to queue
        root_logger = logging.getLogger(); root_logger.addHandler(queue_handler); root_logger.setLevel(logging.INFO) # Logger level still INFO
        core_logger = logging.getLogger("core"); core_logger.addHandler(queue_handler); core_logger.setLevel(logging.INFO) # Logger level still INFO
        output_queue.put("STATUS: Starting evaluation..."); logger.info("Evaluation worker thread started."); output_queue.put("INFO: Evaluation worker thread started.")
        clear_openai_cache(); output_queue.put("INFO: Cleared OpenAI cache (if applicable).")

        # FIX: Pass correct arguments to run_batch_evaluation_core
        async def run_async_wrapper(): return await run_batch_evaluation_core(config=config, output_dir=config.output_options.output_dir, stop_event=stop_event)
        results_paths = asyncio.run(run_async_wrapper())

        if stop_event.is_set(): output_queue.put("STATUS: Evaluation stopped by user."); logger.warning("Evaluation stopped by user signal.")
        elif results_paths:
            output_queue.put(f"SUCCESS: Evaluation completed successfully.")
            output_queue.put(f"RESULTS_PATHS:{json.dumps(results_paths)}"); logger.info(f"Evaluation completed. Results paths: {results_paths}")
        else: output_queue.put("WARNING: Evaluation finished, but no result paths were returned."); logger.warning("Evaluation finished, but no result paths were returned.")
    except Exception as e: error_msg = f"ERROR: Evaluation failed: {e}\n{traceback.format_exc()}"; output_queue.put(error_msg); logger.error(f"Evaluation worker error: {error_msg}")
    finally:
        output_queue.put("STATUS: Evaluation finished."); logger.info("Evaluation worker thread finished.")
        root_logger.removeHandler(queue_handler); core_logger.removeHandler(queue_handler)

# --- Control Functions ---
# --- Control Functions ---
def start_core_evaluation():
    """Starts the evaluation process in a background thread."""
    if st.session_state.evaluation_running: st.warning("Evaluation is already in progress."); return
    config = generate_run_config()
    if not config: st.error("Failed to start evaluation due to configuration errors."); return
    st.session_state.evaluation_running = True; st.session_state.last_run_output = []; st.session_state.evaluation_error = None
    st.session_state.eval_start_time = time.time(); st.session_state.eval_duration_str = "Running..."; st.session_state.stop_event.clear()
    st.session_state.newly_completed_run_folder = None
    while not st.session_state.output_queue.empty():
        try: st.session_state.output_queue.get_nowait()
        except queue.Empty: break
    st.session_state.worker_thread = threading.Thread(target=evaluation_worker, args=(config, st.session_state.output_queue, st.session_state.stop_event), daemon=True)
    st.session_state.worker_thread.start(); st.success("Evaluation started in the background."); logger.info("Evaluation thread started.")
    st.rerun()

def stop_evaluation():
    """Signals the evaluation worker thread to stop."""
    if st.session_state.evaluation_running and st.session_state.worker_thread:
        st.session_state.stop_event.set(); st.warning("Stop signal sent. Waiting for current tasks to finish gracefully..."); logger.info("Stop signal sent to evaluation worker.")
    else: st.info("No evaluation is currently running.")

# --- Results Processing and Display ---

def format_score(score_value):
    """Helper to format scores consistently."""
    if isinstance(score_value, (int, float)): return str(int(score_value)) if float(score_value).is_integer() else f"{score_value:.2f}"
    elif isinstance(score_value, str): return score_value
    else: return str(score_value)

@st.cache_data(show_spinner="Loading results data...")
def load_and_process_results(selected_folders: List[str]) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]], List[str]]:
    """Loads final_results.json from selected folders, processes into DataFrame, calculates summary."""
    all_results_data = []; failed_files = []; loaded_files_count = 0
    if not selected_folders: logger.warning("load_and_process_results called with no selected folders."); return None, None, []
    logger.info(f"Loading results from {len(selected_folders)} selected folders.")
    for folder_path_str in selected_folders:
        folder_path = Path(folder_path_str)
        # FIX: Use glob to find the results file
        results_file_list = list(folder_path.glob("*_final_results.json"))
        if results_file_list:
            results_file = results_file_list[0] # Assume only one matching file
            try:
                with open(results_file, "r") as f: data = json.load(f); all_results_data.append(data); loaded_files_count += 1; logger.debug(f"Successfully loaded: {results_file}")
            except json.JSONDecodeError: logger.error(f"Error decoding JSON from: {results_file}"); failed_files.append(str(results_file))
            except Exception as e: logger.error(f"Error loading file {results_file}: {e}"); failed_files.append(str(results_file))
        else: logger.warning(f"Results file not found in folder: {folder_path}"); failed_files.append(str(folder_path) + " (No Results File)")
    if not all_results_data: logger.warning("No valid results data loaded."); return None, None, failed_files
    logger.info(f"Successfully loaded {loaded_files_count} result files.")

    processed_rows = []; all_rubric_keys = set()
    for result_set in all_results_data:
        for task in result_set.get("results", []):
            for evaluation in task.get("evaluations", []):
                judge_evaluation = evaluation.get("judge_evaluation", {})
                if isinstance(judge_evaluation, dict):
                    parsed_rubric = judge_evaluation.get("parsed_rubric_scores", {})
                    if isinstance(parsed_rubric, dict): all_rubric_keys.update(parsed_rubric.keys())
    sorted_rubric_keys = sorted(list(all_rubric_keys)); logger.debug(f"Collected all unique rubric keys: {all_rubric_keys}")

    for result_set_index, result_set in enumerate(all_results_data):
        summary = result_set.get("summary", {}); run_folder_name = Path(selected_folders[result_set_index]).name
        for task in result_set.get("results", []):
            task_id = task.get("task_id", "N/A")
            for evaluation in task.get("evaluations", []):
                model_id = evaluation.get("model_id", "N/A")
                judge_evaluation = evaluation.get("judge_evaluation", {})
                if not isinstance(judge_evaluation, dict): judge_evaluation = {}
                task_info = {
                    "task_id": task_id, "model_id": model_id, "run_folder": run_folder_name,
                    "evaluation_id": judge_evaluation.get("evaluation_id", "N/A"),
                    "aggregated_score": judge_evaluation.get("aggregated_score", "N/A"),
                    "needs_review": judge_evaluation.get("needs_human_review", False),
                    "review_comments": judge_evaluation.get("human_review_comments", ""), # Get human review comments
                    "judging_error": judge_evaluation.get("parsing_error"), # Use parsing_error for judging issues
                }
                parsed_rubric = judge_evaluation.get("parsed_rubric_scores", {})
                if not isinstance(parsed_rubric, dict): parsed_rubric = {}
                for key in sorted_rubric_keys:
                    rubric_item = parsed_rubric.get(key, {})
                    if isinstance(rubric_item, dict): score = rubric_item.get("score", "N/A"); reason = rubric_item.get("justification", "")
                    else: score = "Error"; reason = "Invalid rubric item format"
                    task_info[f"rubric_score:{key}"] = score; task_info[f"rubric_reason:{key}"] = reason
                processed_rows.append(task_info)

    if not processed_rows: logger.warning("No rows processed from the loaded data."); return None, None, failed_files
    results_df = pd.DataFrame(processed_rows); logger.info(f"Created DataFrame with {len(results_df)} rows.")

    total_evaluations = len(results_df); needs_review_count = results_df["needs_review"].sum()
    pass_fail_counts = results_df["aggregated_score"].value_counts().to_dict() if "aggregated_score" in results_df.columns else {}
    numeric_rubric_avg = {}
    numeric_rubric_cols = [col for col in results_df.columns if col.startswith("rubric_score:")]
    for col in numeric_rubric_cols:
        numeric_scores = pd.to_numeric(results_df[col], errors="coerce")
        if not numeric_scores.isnull().all():
            avg_score = numeric_scores.mean(skipna=True)
            numeric_rubric_avg[col.split(":", 1)[1]] = f"{avg_score:.2f}" if pd.notna(avg_score) else "N/A"
    
    # FIX: Include the original summary data in the returned tuple
    original_summary_data = all_results_data[0].get("summary", {}) if all_results_data else {}

    aggregated_summary = {
        "total_files_loaded": loaded_files_count, "total_folders_selected": len(selected_folders),
        "total_evaluations": total_evaluations, "needs_review_count": needs_review_count,
        "overall_score_distribution": pass_fail_counts, "average_numeric_rubric_scores": numeric_rubric_avg,
        # FIX: Add original summary data here if needed for display_summary_stats
        "original_summary": original_summary_data
    }
    logger.info(f"Calculated aggregated summary: {aggregated_summary}")
    return results_df, aggregated_summary, failed_files

def render_results_selector():
    """Renders the UI for selecting previously run result folders."""
    st.header("📂 Select Evaluation Results") # Moved header here
    potential_folders = []
    if DATA_DIR.exists() and DATA_DIR.is_dir():
        for item in DATA_DIR.iterdir():
            if item.is_dir():
                # FIX: Use glob to find *any* file ending with _final_results.json
                if list(item.glob("*_final_results.json")):
                    potential_folders.append(item)
    if not potential_folders:
        st.info("No previous evaluation result folders found in the data directory.")
        st.session_state.selected_results_folders = []; return
    try:
        all_folders = sorted(potential_folders, key=lambda x: x.stat().st_mtime, reverse=True)
        all_folder_paths = [str(f) for f in all_folders]; all_folder_names = [f.name for f in all_folders]
    except Exception as e:
        st.error(f"Error sorting result folders: {e}"); logger.error(f"Error sorting result folders: {e}")
        all_folder_paths = [str(f) for f in potential_folders]; all_folder_names = [f.name for f in potential_folders]

    pre_selected_folders = st.session_state.selected_results_folders
    if st.session_state.newly_completed_run_folder:
        new_folder_path = str(st.session_state.newly_completed_run_folder)
        if new_folder_path in all_folder_paths and new_folder_path not in pre_selected_folders:
            pre_selected_folders = [new_folder_path]; logger.info(f"Pre-selecting newly completed run folder: {new_folder_path}")
        st.session_state.newly_completed_run_folder = None

    selected_folder_names = st.multiselect(
        "Choose result folder(s) to view:", options=all_folder_names,
        default=[name for name in all_folder_names if str(DATA_DIR / name) in pre_selected_folders],
        help="Select one or more completed evaluation runs.",
    )
    new_selection = [str(DATA_DIR / name) for name in selected_folder_names]
    if set(st.session_state.selected_results_folders) != set(new_selection):
        st.session_state.selected_results_folders = new_selection; logger.info(f"Result folder selection updated: {new_selection}")
        st.cache_data.clear(); st.rerun()

def display_summary_stats(original_summary: Optional[Dict[str, Any]]):
    """Displays the summary statistics directly from the results file."""
    if not original_summary: st.info("No summary statistics available for the selected results."); return
    st.header("📈 Overall Summary Statistics")
    metric_keys = [("total_files_loaded", "Files Loaded"), ("total_folders_selected", "Folders Selected"), ("total_evaluations", "Total Evaluations"), ("needs_review_count", "Needs Review")]
    cols = st.columns(len(metric_keys))
    for idx, (key, name) in enumerate(metric_keys): cols[idx].metric(name, original_summary.get(key, "N/A"))
    st.subheader("Overall Score Distribution")
    score_dist = original_summary.get("overall_score_distribution", {})
    if score_dist:
        score_df = pd.DataFrame(list(score_dist.items()), columns=["Score", "Count"])
        score_order = ["Pass", "Partial", "Fail", "N/A"]; score_df["Score"] = pd.Categorical(score_df["Score"], categories=score_order, ordered=True)
        score_df = score_df.sort_values("Score").reset_index(drop=True)
        fig_score_dist = px.bar(score_df, x="Score", y="Count", title="Distribution of Aggregated Scores", color="Score", color_discrete_map=COLOR_MAP, labels={"Score": "Aggregated Score", "Count": "Number of Evaluations"})
        fig_score_dist.update_layout(xaxis_title=None); st.plotly_chart(fig_score_dist, use_container_width=True)
    else: st.info("No aggregated score distribution data available.")
    st.subheader("Average Numeric Rubric Scores")
    avg_scores = original_summary.get("average_numeric_rubric_scores", {})
    if avg_scores: avg_df = pd.DataFrame(list(avg_scores.items()), columns=["Rubric", "Average Score"]); st.dataframe(avg_df, use_container_width=True)
    else: st.info("No average numeric rubric scores calculated (or no numeric rubrics found).")

def display_performance_plots(df: pd.DataFrame):
    """Displays plots related to overall performance (Pass/Fail/Partial)."""
    if df is None or df.empty or "aggregated_score" not in df.columns: st.warning("No data available or 'aggregated_score' column missing for performance plots."); return
    st.header("🚀 Performance Analysis (Aggregated Score)")

    # Overall Performance Distribution
    try:
        score_order = ["Pass", "Partial", "Fail", "N/A"]
        # Ensure all possible scores are in the counts, even if 0
        performance_counts = df["aggregated_score"].value_counts().reindex(score_order, fill_value=0).reset_index()
        performance_counts.columns = ["score", "count"]
        chart_title = "Overall Performance Distribution"
        fig_perf = px.bar(performance_counts, x="score", y="count", title=chart_title, color="score", color_discrete_map=COLOR_MAP, labels={"score": "Aggregated Score", "count": "Number of Evaluations"})
        fig_perf.update_layout(xaxis_title=None)
        st.plotly_chart(fig_perf, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating overall performance plot: {e}")
        logger.error(f"Plotting error for overall performance: {traceback.format_exc()}")

    # Distribution by Model (if 'model_id' column exists)
    st.subheader("Distribution by Model")
    if "model_id" in df.columns:
        try:
            # Group by model_id and count occurrences of each aggregated_score
            model_perf_counts = df.groupby("model_id")["aggregated_score"].value_counts().unstack(fill_value=0).reset_index()
            # Ensure all score columns exist, fill missing with 0
            for score in score_order:
                if score not in model_perf_counts.columns:
                    model_perf_counts[score] = 0
            # Melt the DataFrame for Plotly
            model_perf_melted = model_perf_counts.melt(id_vars="model_id", value_vars=score_order, var_name="score", value_name="count")
            chart_title = "Performance Distribution by Model"
            fig_model_perf = px.bar(model_perf_melted, x="model_id", y="count", color="score", title=chart_title, barmode="group", color_discrete_map=COLOR_MAP, labels={"model_id": "Model ID", "count": "Number of Evaluations", "score": "Aggregated Score"})
            fig_model_perf.update_layout(xaxis_title=None)
            st.plotly_chart(fig_model_perf, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating performance plot by model: {e}")
            logger.error(f"Plotting error for performance by model: {traceback.format_exc()}")
    else:
        st.info("Column 'model_id' not found. Cannot show performance breakdown by model.")


def display_rubric_plots(df: pd.DataFrame):
    """Displays plots related to rubric scores."""
    rubric_cols = [col for col in df.columns if col.startswith("rubric_score:")]
    if df is None or df.empty or not rubric_cols:
        st.warning("No data available or no rubric score columns found for rubric plots."); return

    st.header("📊 Rubric Score Analysis")

    # Overall Rubric Distribution
    st.subheader("Overall Rubric Distribution")
    try:
        # Melt the DataFrame to long format for easier plotting
        rubric_melted = df[rubric_cols].melt(var_name="rubric", value_name="score")
        rubric_melted["rubric"] = rubric_melted["rubric"].str.replace("rubric_score:", "") # Clean up rubric names

        if not rubric_melted.empty:
            # Count occurrences of each score for each rubric
            rubric_counts = rubric_melted.groupby("rubric")["score"].value_counts().unstack(fill_value=0).stack().reset_index(name="count")
            rubric_counts.columns = ["rubric", "score", "count"]

            # Define a consistent order for scores
            score_order = ["Yes", "Partial", "No", "N/A"]
            rubric_counts["score"] = pd.Categorical(rubric_counts["score"], categories=score_order, ordered=True)
            rubric_counts = rubric_counts.sort_values(["rubric", "score"])

            chart_title = "Distribution of Rubric Scores Across All Evaluations"
            fig_rubric = px.bar(rubric_counts, x="rubric", y="count", color="score", title=chart_title, barmode="group", color_discrete_map=COLOR_MAP, labels={"rubric": "Rubric Criterion", "count": "Number of Evaluations", "score": "Score"})
            fig_rubric.update_layout(xaxis_title=None)
            st.plotly_chart(fig_rubric, use_container_width=True)
        else:
            st.info("No specific rubric scores found to display distribution.")
    except Exception as e:
        st.warning(f"Could not generate overall rubric distribution plot: {e}")
        logger.error(f"Plotting error for overall rubric distribution: {traceback.format_exc()}")

    # Rubric Scores by Model (if 'model_id' column exists and rubrics exist)
    st.subheader("Rubric Scores by Model")
    has_rubrics = any(col.startswith("rubric_score:") for col in df.columns)
    if "model_id" in df.columns and has_rubrics:
        try:
            # Allow selecting a specific rubric criterion to view by model
            available_rubrics = [col.replace("rubric_score:", "") for col in rubric_cols]
            selected_criterion = st.selectbox("Select Rubric Criterion to view by Model:", options=available_rubrics, key="rubric_model_criterion_select")

            if selected_criterion:
                rubric_score_col = f"rubric_score:{selected_criterion}"
                if rubric_score_col in df.columns:
                    # Group by model_id and the selected rubric score, then count
                    rubric_model_counts = df.groupby("model_id")[rubric_score_col].value_counts().unstack(fill_value=0).reset_index()

                    # Ensure all possible scores are in the columns, fill missing with 0
                    score_order = ["Yes", "Partial", "No", "N/A"]
                    for score in score_order:
                        if score not in rubric_model_counts.columns:
                            rubric_model_counts[score] = 0

                    # Melt the DataFrame for Plotly
                    rubric_model_melted = rubric_model_counts.melt(id_vars="model_id", value_vars=score_order, var_name="score", value_name="count")

                    chart_title = f"Distribution of '{selected_criterion}' Rubric Scores by Model"
                    fig_rubric_model = px.bar(rubric_model_melted, x="model_id", y="count", color="score", title=chart_title, barmode="group", color_discrete_map=COLOR_MAP, labels={"model_id": "Model ID", "count": "Number of Evaluations", "score": "Score"})
                    fig_rubric_model.update_layout(xaxis_title=None)
                    st.plotly_chart(fig_rubric_model, use_container_width=True)
                else:
                    st.warning(f"Rubric criterion '{selected_criterion}' not found in the data.")
        except Exception as e:
            st.error(f"Error generating rubric plot by model: {e}")
            logger.error(f"Plotting error for rubric by model: {traceback.format_exc()}")
    else:
        st.info("Column 'model_id' not found or no rubric scores available. Cannot show rubric breakdown by model.")


def display_results_table(df: pd.DataFrame):
    """Displays the detailed results in a sortable, filterable table."""
    if df is None or df.empty: st.warning("No detailed results data available."); return
    st.header("📋 Detailed Evaluation Results")

    # Select columns to display
    all_columns = df.columns.tolist()
    default_columns = ["task_id", "model_id", "run_folder", "aggregated_score", "needs_review", "judging_error", "review_comments"]
    rubric_score_cols = sorted([col for col in all_columns if col.startswith("rubric_score:")])
    rubric_reason_cols = sorted([col for col in all_columns if col.startswith("rubric_reason:")])

    # Combine default, score, and reason columns, ensuring no duplicates
    display_columns_options = list(dict.fromkeys(default_columns + rubric_score_cols + rubric_reason_cols))

    selected_columns = st.multiselect(
        "Select columns to display:",
        options=display_columns_options,
        default=default_columns + rubric_score_cols, # Default to showing scores but not reasons
        help="Choose which columns to show in the results table."
    )

    if not selected_columns:
        st.warning("Please select at least one column to display.")
        return

    # Filter by 'Needs Review'
    filter_needs_review = st.checkbox("Show only tasks needing human review")
    filtered_df = df[df["needs_review"] == True] if filter_needs_review and "needs_review" in df.columns else df

    # Filter by Aggregated Score
    if "aggregated_score" in filtered_df.columns:
        available_scores = filtered_df["aggregated_score"].dropna().unique().tolist()
        selected_scores = st.multiselect(
            "Filter by Aggregated Score:",
            options=sorted(available_scores),
            default=sorted(available_scores),
            help="Select which aggregated scores to display."
        )
        if selected_scores:
            filtered_df = filtered_df[filtered_df["aggregated_score"].isin(selected_scores)]
        elif available_scores:
             st.warning("No aggregated scores selected. Showing all results.")


    # Filter by Model ID
    if "model_id" in filtered_df.columns:
        available_models = filtered_df["model_id"].dropna().unique().tolist()
        selected_models = st.multiselect(
            "Filter by Model ID:",
            options=sorted(available_models),
            default=sorted(available_models),
            help="Select which model IDs to display."
        )
        if selected_models:
            filtered_df = filtered_df[filtered_df["model_id"].isin(selected_models)]
        elif available_models:
             st.warning("No models selected. Showing all results.")

    # Display the filtered DataFrame with selected columns
    if not filtered_df.empty:
        # Ensure only selected columns that actually exist in the DataFrame are passed
        cols_to_display = [col for col in selected_columns if col in filtered_df.columns]
        st.dataframe(filtered_df[cols_to_display], use_container_width=True, hide_index=True)

        # Add a button to view full details of a selected task/model combination
        st.subheader("View Full Task/Evaluation Details")
        task_model_combinations = filtered_df.apply(lambda row: f"Task ID: {row['task_id']} | Model ID: {row['model_id']} | Run: {row['run_folder']}", axis=1).tolist()
        selected_combination_str = st.selectbox("Select a task/model combination to view full details:", options=[""] + task_model_combinations)

        if selected_combination_str:
            try:
                # Parse the selected string to get task_id, model_id, and run_folder
                parts = selected_combination_str.split(" | ")
                selected_task_id = parts[0].replace("Task ID: ", "")
                selected_model_id = parts[1].replace("Model ID: ", "")
                selected_run_folder_name = parts[2].replace("Run: ", "")

                # Find the corresponding row in the original DataFrame
                selected_row = df[
                    (df["task_id"] == selected_task_id) &
                    (df["model_id"] == selected_model_id) &
                    (df["run_folder"] == selected_run_folder_name)
                ].iloc[0] # Get the first match

                st.write("### Full Details:")
                st.json(selected_row.to_dict()) # Display the row as JSON
            except Exception as e:
                st.error(f"Error retrieving full details: {e}")
                logger.error(f"Error retrieving full details for selected combination: {traceback.format_exc()}")

    else:
        st.info("No results match the selected filters.")


def display_human_review_tasks(df: pd.DataFrame):
    """Displays tasks flagged for human review."""
    if df is None or df.empty or "needs_review" not in df.columns: return
    review_tasks = df[df["needs_review"] == True]
    if not review_tasks.empty:
        st.header("🚩 Tasks Flagged for Human Review")
        st.dataframe(review_tasks[["task_id", "model_id", "run_folder", "aggregated_score", "judging_error", "review_comments"]], use_container_width=True, hide_index=True)
    # else:
    #     st.info("No tasks currently flagged for human review.")


def render_evaluation_progress(output_queue: queue.Queue, last_run_output: List[str], eval_start_time: Optional[float]):
    """Renders the evaluation progress and output."""
    st.header("🏃‍♀️ Evaluation Progress")
    status_placeholder = st.empty()
    output_area = st.empty()
    duration_placeholder = st.empty()

    current_output = last_run_output[:]
    results_paths = None
    evaluation_error = None

    while True:
        try:
            line = output_queue.get_nowait()
            if line.startswith("STATUS:"):
                status_placeholder.info(line.replace("STATUS:", "").strip())
            elif line.startswith("SUCCESS:"):
                status_placeholder.success(line.replace("SUCCESS:", "").strip())
            elif line.startswith("WARNING:"):
                status_placeholder.warning(line.replace("WARNING:", "").strip())
            elif line.startswith("ERROR:"):
                error_msg = line.replace("ERROR:", "").strip()
                status_placeholder.error(error_msg)
                evaluation_error = error_msg
            elif line.startswith("RESULTS_PATHS:"):
                try:
                    results_paths = json.loads(line.replace("RESULTS_PATHS:", "").strip())
                    logger.info(f"Received results paths from queue: {results_paths}")
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode results paths from queue line: {line}")
                    status_placeholder.error("Error processing results paths.")
            else:
                current_output.append(line)
            output_area.code("\n".join(current_output), language="text")

        except queue.Empty:
            pass # No new output

        if st.session_state.worker_thread and not st.session_state.worker_thread.is_alive():
            st.session_state.evaluation_running = False
            st.session_state.last_run_output = current_output
            st.session_state.evaluation_error = evaluation_error
            if eval_start_time:
                duration = time.time() - eval_start_time
                st.session_state.eval_duration_str = f"Finished in {timedelta(seconds=duration)}"
                duration_placeholder.text(f"Duration: {st.session_state.eval_duration_str}")
            if results_paths:
                 # Assuming results_paths is a list of file paths, extract the directory name
                 if results_paths and isinstance(results_paths, list) and len(results_paths) > 0:
                     # Get the directory of the first result file
                     result_dir = Path(results_paths[0]).parent
                     st.session_state.newly_completed_run_folder = result_dir
                     logger.info(f"Setting newly_completed_run_folder to: {result_dir}")
                 st.session_state.evaluation_results_paths = results_paths # Store paths in session state
            st.rerun() # Trigger rerun to update UI with final status and results
            break # Exit the loop

        if eval_start_time:
            duration = time.time() - eval_start_time
            duration_placeholder.text(f"Duration: {timedelta(seconds=duration)}")

        time.sleep(0.1) # Small delay to prevent excessive CPU usage

# --- Main Application ---
def main() -> None:
    """Main function to run the Streamlit application."""
    initialize_session_state()
    st.title("CogniBench Evaluation Runner")

    render_file_uploader()
    render_config_ui()

    if st.session_state.config_complete and st.session_state.uploaded_files_info:
        render_config_summary()
        if not st.session_state.evaluation_running:
            if st.button("🚀 Start Evaluation"):
                start_core_evaluation()
        else:
            st.button("⏹ Stop Evaluation", on_click=stop_evaluation)
            if st.session_state.eval_duration_str:
                 st.text(f"Duration: {st.session_state.eval_duration_str}")

    if st.session_state.evaluation_running or st.session_state.last_run_output:
         render_evaluation_progress(st.session_state.output_queue, st.session_state.last_run_output, st.session_state.eval_start_time)

    render_results_selector()

    if st.session_state.selected_results_folders:
        results_df, aggregated_summary, failed_files = load_and_process_results(st.session_state.selected_results_folders)

        if failed_files:
            st.warning(f"Failed to load or parse the following result files/folders: {', '.join(failed_files)}")

        if results_df is not None and aggregated_summary is not None:
            display_summary_stats(aggregated_summary)
            display_performance_plots(results_df)
            display_rubric_plots(results_df)
            display_human_review_tasks(results_df) # Display tasks needing human review
            display_results_table(results_df) # Display the detailed table

if __name__ == "__main__":
    main()
    logger.debug("Streamlit main function finished.")
    main()
    logger.debug("Streamlit main function finished.")
