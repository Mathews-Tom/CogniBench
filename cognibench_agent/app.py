"""
CogniBench Streamlit Application.

Provides a web-based user interface for running CogniBench evaluations,
configuring models and prompts, uploading data, viewing results, and
managing the evaluation process.
"""

import asyncio  # Added import
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
from core.constants import (APP_DIR, AVAILABLE_MODELS, BASE_CONFIG_PATH,
                            COLOR_MAP, DATA_DIR, DEFAULT_JUDGING_MODEL,
                            DEFAULT_STRUCTURING_MODEL, JUDGING_TEMPLATES_DIR,
                            PROMPT_TEMPLATES_DIR_ABS,
                            STRUCTURING_TEMPLATES_DIR)
from core.evaluation_runner import run_batch_evaluation_core
from core.llm_clients.openai_client import clear_openai_cache
from core.log_setup import setup_logging

# --- Logging Setup ---
logger = logging.getLogger("frontend")
if "logging_setup_complete" not in st.session_state:
    setup_logging(log_level=logging.DEBUG)
    st.session_state.logging_setup_complete = True
    logger = logging.getLogger("frontend")
    logger.setLevel(logging.DEBUG)
    logger.info("Logger 'frontend' level forced to DEBUG.")
    logger.info("Initial logging setup complete.")
    logger.info("Streamlit app started.")

# --- Page Config ---
st.set_page_config(layout="wide", page_title="CogniBench Runner")

# --- Helper Functions ---

def get_templates(directory: Path) -> Dict[str, str]:
    """Scans a directory for .txt files and returns a dictionary of name: path."""
    try:
        return {
            f.stem: str(f)
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
            st.selectbox("Provider", options=list(AVAILABLE_MODELS.keys()), key="structuring_provider_select")
            st.selectbox("Model", options=list(AVAILABLE_MODELS[st.session_state.structuring_provider_select].keys()), key="structuring_model_select")
            st.text_input("API Key (Optional)", type="password", placeholder="Leave blank to use environment variable", key="structuring_api_key_input")
        with col_judging:
            st.subheader("Judging Model")
            st.selectbox("Provider", options=list(AVAILABLE_MODELS.keys()), key="judging_provider_select")
            st.selectbox("Model", options=list(AVAILABLE_MODELS[st.session_state.judging_provider_select].keys()), key="judging_model_select")
            st.text_input("API Key (Optional)", type="password", placeholder="Leave blank to use environment variable", key="judging_api_key_input")

    with st.expander("Prompt Configurations", expanded=False):
        col_prompt1, col_prompt2 = st.columns(2)
        with col_prompt1:
            st.selectbox("Structuring Prompt Template", options=list(AVAILABLE_STRUCTURING_TEMPLATES.keys()), key="structuring_template_select", help="Select the template file for structuring.")
            if st.button("View Structuring Prompt"): st.session_state.show_structuring = not st.session_state.show_structuring
        with col_prompt2:
            st.selectbox("Judging Prompt Template", options=list(AVAILABLE_JUDGING_TEMPLATES.keys()), key="judging_template_select", help="Select the template file for judging.")
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

    aggregated_summary = {
        "total_files_loaded": loaded_files_count, "total_folders_selected": len(selected_folders),
        "total_evaluations": total_evaluations, "needs_review_count": needs_review_count,
        "overall_score_distribution": pass_fail_counts, "average_numeric_rubric_scores": numeric_rubric_avg,
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
    score_order = ["Pass", "Partial", "Fail", "N/A"]
    st.subheader("Overall Distribution")
    try:
        performance_counts = df["aggregated_score"].value_counts().reindex(score_order, fill_value=0).reset_index(); performance_counts.columns = ["score", "count"]
        fig_perf = px.bar(performance_counts, x="score", y="count", title="Overall Aggregated Score Distribution", color="score", color_discrete_map=COLOR_MAP, category_orders={"score": score_order}, labels={"score": "Aggregated Score", "count": "Number of Evaluations"})
        fig_perf.update_layout(xaxis_title=None); st.plotly_chart(fig_perf, use_container_width=True)
    except Exception as e: st.error(f"Error generating overall performance plot: {e}"); logger.error(f"Plotting error for overall performance: {traceback.format_exc()}")
    st.subheader("Distribution by Model")
    if "model_id" in df.columns:
        try:
            model_perf_counts = df.groupby("model_id")["aggregated_score"].value_counts().unstack(fill_value=0).reset_index()
            for score in score_order:
                if score not in model_perf_counts.columns: model_perf_counts[score] = 0
            melt_vars = [s for s in score_order if s in model_perf_counts.columns]
            model_perf_melted = model_perf_counts.melt(id_vars="model_id", value_vars=melt_vars, var_name="score", value_name="count")
            fig_model_perf = px.bar(model_perf_melted, x="model_id", y="count", color="score", title="Aggregated Score Distribution by Model", barmode="group", color_discrete_map=COLOR_MAP, category_orders={"score": score_order}, labels={"model_id": "Model", "count": "Number of Evaluations", "score": "Aggregated Score"})
            st.plotly_chart(fig_model_perf, use_container_width=True)
        except Exception as e: st.error(f"Error generating performance plot by model: {e}"); logger.error(f"Plotting error for performance by model: {traceback.format_exc()}")
    else: st.info("Column 'model_id' not found. Cannot show performance breakdown by model.")

# --- Reverted display_rubric_plots function ---
def display_rubric_plots(df: pd.DataFrame):
    """Displays plots for rubric scores (Original version)."""
    if df is None or df.empty:
        st.warning("No data available to display rubric plots.")
        return

    st.header("📊 Rubric Score Analysis")

    rubric_cols = sorted([col for col in df.columns if col.startswith("rubric_score:")])
    has_rubrics = bool(rubric_cols)

    st.subheader("Distribution for a Selected Rubric")
    if has_rubrics:
        selected_rubric_chart1 = st.selectbox(
            "Select Rubric Criterion:",
            options=rubric_cols,
            format_func=lambda x: x.split(":", 1)[1] if ":" in x else x,
            key="rubric_selector_chart1_reverted", # Use new key to avoid state conflict
            help="Choose a rubric criterion to see its score distribution across all models.",
        )

        if selected_rubric_chart1:
            try:
                if selected_rubric_chart1 not in df.columns:
                     st.warning(f"Rubric column '{selected_rubric_chart1}' not found in the data.")
                else:
                    id_vars = ["model_id"] # Minimal id_vars needed
                    id_vars = [col for col in id_vars if col in df.columns]

                    rubric_melted = df.melt(
                        id_vars=id_vars,
                        value_vars=[selected_rubric_chart1],
                        var_name="rubric_criterion",
                        value_name="score",
                    )
                    score_order = ["Yes", "Partial", "No", "N/A"]
                    rubric_counts = (
                        rubric_melted["score"]
                        .value_counts()
                        .reindex(score_order, fill_value=0)
                        .reset_index()
                    )
                    rubric_counts.columns = ["score", "count"]
                    chart1_title = f"Distribution for Rubric: {selected_rubric_chart1.split(':', 1)[1]} (All Models)"
                    fig_rubric = px.bar(
                        rubric_counts, x="score", y="count", title=chart1_title,
                        color="score", color_discrete_map=COLOR_MAP, category_orders={"score": score_order},
                        labels={"score": "Score", "count": "Number of Evaluations"},
                    )
                    fig_rubric.update_layout(xaxis_title=None)
                    st.plotly_chart(fig_rubric, use_container_width=True)

            except Exception as e:
                st.error(f"Error generating plot for rubric '{selected_rubric_chart1}': {e}")
                logger.error(f"Plotting error for {selected_rubric_chart1}: {traceback.format_exc()}")
    else:
        st.info("No specific rubric scores found to display distribution.")


    st.subheader("Rubric Scores by Model")
    if has_rubrics:
        selected_rubric_chart2 = st.selectbox(
            "Select Rubric Criterion for Model Breakdown:",
            options=rubric_cols, # Only show actual rubrics
            format_func=lambda x: x.split(":", 1)[1] if ":" in x else x,
            key="rubric_selector_chart2_reverted", # Use new key
            help="Choose a rubric criterion to see the score breakdown by model.",
        )

        if selected_rubric_chart2:
            try:
                if selected_rubric_chart2 not in df.columns:
                    st.warning(f"Selected rubric column '{selected_rubric_chart2}' not found in the data.")
                elif "model_id" not in df.columns:
                     st.warning("Column 'model_id' not found. Cannot show breakdown by model.")
                else:
                    rubric_model_counts = (
                        df.groupby("model_id")[selected_rubric_chart2]
                        .value_counts()
                        .unstack(fill_value=0)
                        .reset_index()
                    )
                    score_order = ["Yes", "Partial", "No", "N/A"]
                    for score in score_order:
                        if score not in rubric_model_counts.columns: rubric_model_counts[score] = 0
                    rubric_model_counts = rubric_model_counts.melt(
                        id_vars="model_id", value_vars=[s for s in score_order if s in rubric_model_counts.columns],
                        var_name="score", value_name="count",
                    )
                    chart2_title = f"Scores for Rubric: {selected_rubric_chart2.split(':', 1)[1]} by Model"
                    fig_rubric_model = px.bar(
                        rubric_model_counts, x="model_id", y="count", color="score", title=chart2_title,
                        barmode="group", color_discrete_map=COLOR_MAP, category_orders={"score": score_order},
                        labels={"model_id": "Model", "count": "Number of Evaluations", "score": "Score"},
                    )
                    st.plotly_chart(fig_rubric_model, use_container_width=True)

            except Exception as e:
                criterion_name = selected_rubric_chart2.split(":", 1)[1] if ":" in selected_rubric_chart2 else selected_rubric_chart2
                st.error(f"Error generating plot for criterion '{criterion_name}' by model: {e}")
                logger.error(f"Plotting error for {selected_rubric_chart2} by model: {traceback.format_exc()}")
    else:
        st.info("No specific rubric scores found to display breakdown by model.")
# --- End Reverted display_rubric_plots ---

def display_results_table(df: pd.DataFrame):
    """Displays the detailed results in a filterable table and allows drilling down."""
    if df is None or df.empty: st.warning("No data available to display in the results table."); return
    st.header("📋 Detailed Evaluation Results")
    st.subheader("Filter Results")
    col_filter1, col_filter2, col_filter3, col_filter4 = st.columns(4)
    with col_filter1: model_options = ["All"] + sorted(df["model_id"].unique()); selected_model = st.selectbox("Filter by Model:", options=model_options)
    with col_filter2: struct_success_options = ["All", "Success", "Failure"]; selected_struct_success = st.selectbox("Filter by Structuring Success:", options=struct_success_options) # Note: Column might not exist
    with col_filter3: judge_success_options = ["All", "Success", "Failure"]; selected_judge_success = st.selectbox("Filter by Judging Success:", options=judge_success_options) # Note: Column might not exist
    with col_filter4: review_options = ["All", "Yes", "No"]; selected_review_status = st.radio("Filter by Needs Review:", options=review_options, horizontal=True)

    filtered_df = df.copy()
    if selected_model != "All": filtered_df = filtered_df[filtered_df["model_id"] == selected_model]
    if selected_struct_success != "All" and "structuring_success" in filtered_df.columns: filtered_df = filtered_df[filtered_df["structuring_success"] == (selected_struct_success == "Success")]
    if selected_judge_success != "All" and "judging_success" in filtered_df.columns: filtered_df = filtered_df[filtered_df["judging_success"] == (selected_judge_success == "Success")]
    if selected_review_status != "All": filtered_df = filtered_df[filtered_df["needs_review"] == (selected_review_status == "Yes")]

    st.subheader("Filtered Results Overview")
    if filtered_df.empty: st.info("No results match the current filter criteria.")
    else:
        cols_to_show = ["task_id", "model_id", "aggregated_score", "needs_review", "evaluation_id", "run_folder"]
        rubric_score_cols_main = sorted([col for col in filtered_df.columns if col.startswith("rubric_score:")])
        cols_to_show.extend(rubric_score_cols_main)
        cols_to_show = [col for col in cols_to_show if col in filtered_df.columns]
        column_rename_map_table = {"task_id": "Task ID", "model_id": "Model", "aggregated_score": "Overall Score", "needs_review": "Needs Review?", "evaluation_id": "Evaluation ID", "run_folder": "Source Run"}
        for col in rubric_score_cols_main: formatted_name = col.split(":", 1)[1].replace("_", " ").title() if ":" in col else col.replace("_", " ").title(); column_rename_map_table[col] = f"Rubric: {formatted_name}"
        display_table_df = filtered_df[cols_to_show].copy(); display_table_df.rename(columns=column_rename_map_table, inplace=True)
        st.dataframe(display_table_df, use_container_width=True, height=400)

        st.subheader("Drill Down into Specific Evaluation")
        col_select1, col_select2 = st.columns(2)
        with col_select1:
            task_id_options = ["Select Task ID..."] + sorted(filtered_df["task_id"].unique().tolist()); selected_task_id_str = st.selectbox("Select Task ID:", options=task_id_options, key="drill_task_id", index=0)
            selected_task_id = selected_task_id_str if selected_task_id_str != "Select Task ID..." else None
        with col_select2:
            available_models_for_task = ["Select Model..."]
            if selected_task_id:
                try: available_models_for_task.extend(sorted(filtered_df[filtered_df["task_id"] == selected_task_id]["model_id"].unique().tolist()))
                except Exception as e: logger.error(f"Error filtering models for task {selected_task_id}: {e}")
            selected_model_id = st.selectbox("Select Model:", options=available_models_for_task, key="drill_model_id", index=0, disabled=(selected_task_id is None))
            selected_model_id = selected_model_id if selected_model_id != "Select Model..." else None

        if selected_task_id and selected_model_id:
            try:
                selected_rows = filtered_df[(filtered_df["task_id"] == selected_task_id) & (filtered_df["model_id"] == selected_model_id)]
                if not selected_rows.empty:
                    selected_row = selected_rows.iloc[0]
                    st.markdown(f"**Details for Task:** `{selected_task_id}` | **Model:** `{selected_model_id}` | **Run:** `{selected_row.get('run_folder', 'N/A')}`")
                    # Display Judging Info
                    with st.expander("Judging Details", expanded=True):
                        st.metric("Aggregated Score", selected_row.get("aggregated_score", "N/A"))
                        st.metric("Needs Review", selected_row.get("needs_review", "N/A"))
                        comments = selected_row.get("review_comments"); error = selected_row.get("judging_error")
                        if comments: st.info(f"Review Comments: {comments}")
                        if error: st.error(f"Judging Error: {error}")
                        st.subheader("Raw Judge Response"); raw_response = selected_row.get("raw_judge_response")
                        if raw_response:
                            try: st.json(json.loads(raw_response))
                            except json.JSONDecodeError: st.text(raw_response)
                        else: st.text("N/A")
                        st.subheader("Parsed Rubric Scores & Reasons")
                        rubric_score_cols = sorted([col for col in selected_row.index if col.startswith("rubric_score:")])
                        if rubric_score_cols:
                            rubric_data_to_display = {}
                            for score_col in rubric_score_cols:
                                reason_col = score_col.replace("rubric_score:", "rubric_reason:")
                                rubric_name = score_col.split(":", 1)[1]; score = selected_row.get(score_col, "N/A"); reason = selected_row.get(reason_col, "")
                                rubric_data_to_display[rubric_name] = {"Score": score, "Reason": reason}
                            rubric_df = pd.DataFrame.from_dict(rubric_data_to_display, orient="index"); st.dataframe(rubric_df, use_container_width=True)
                        else: st.info("No rubric scores found for this evaluation.")
                else: st.warning(f"No data found for Task ID '{selected_task_id}' and Model '{selected_model_id}' with current filters.")
            except Exception as e: st.error(f"Error displaying drill-down details: {e}"); logger.error(f"Drill-down error for {selected_task_id}/{selected_model_id}: {traceback.format_exc()}")
        else: st.info("Select a Task ID and Model ID above to see detailed information.")

def display_human_review_tasks(df: pd.DataFrame):
    """Displays tasks flagged for human review."""
    if df is None or df.empty or "needs_review" not in df.columns: st.warning("No data available or 'needs_review' column missing."); return
    st.header("👀 Tasks Flagged for Human Review")
    review_df = df[df["needs_review"] == True].copy()
    if review_df.empty: st.success("No tasks are currently flagged for human review."); return
    st.info(f"Found {len(review_df)} tasks flagged for review.")
    cols_to_show_review = ["task_id", "model_id", "run_folder", "aggregated_score", "review_comments", "judging_error"]
    rubric_score_cols = sorted([col for col in review_df.columns if col.startswith("rubric_score:")]); cols_to_show_review.extend(rubric_score_cols); cols_to_show_review.append("evaluation_id")
    cols_to_show_review = [col for col in cols_to_show_review if col in review_df.columns]
    display_review_df = review_df[cols_to_show_review]
    column_rename_map = {"task_id": "Task ID", "model_id": "Model", "run_folder": "Source Run", "aggregated_score": "Current Score", "review_comments": "Comments", "judging_error": "Judging Error", "evaluation_id": "Evaluation ID"}
    for col in rubric_score_cols: formatted_name = col.split(":", 1)[1].replace("_", " ").title() if ":" in col else col.replace("_", " ").title(); column_rename_map[col] = f"Rubric: {formatted_name}"
    display_review_df = display_review_df.rename(columns=column_rename_map)
    final_column_order = ["Task ID", "Model", "Source Run", "Current Score", "Comments", "Judging Error"] + [col for col in display_review_df.columns if col.startswith("Rubric:")] + ["Evaluation ID"]
    final_column_order = [col for col in final_column_order if col in display_review_df.columns]
    st.dataframe(display_review_df[final_column_order], use_container_width=True, height=300)
    st.markdown("""**Next Steps:**\n1. Review details using the table above.\n2. Use external tools/scripts to update review status/comments.\n3. Reload results here.""")

# --- Progress Display ---
def render_evaluation_progress(output_queue: queue.Queue, last_run_output: List[str], eval_start_time: Optional[float]):
    """Displays the progress/output of the evaluation run."""
    st.header("🚀 Evaluation Progress")
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
    run_button_disabled = st.session_state.evaluation_running or not st.session_state.config_complete or not st.session_state.uploaded_files_info
    run_tooltip = ""
    if not st.session_state.config_complete: run_tooltip = "Configuration is incomplete."
    elif not st.session_state.uploaded_files_info: run_tooltip = "Please upload input files."
    elif st.session_state.evaluation_running: run_tooltip = "Evaluation is already running."
    with col_ctrl1:
        if st.button("▶️ Run Evaluation", disabled=run_button_disabled, help=run_tooltip or "Start evaluation.", use_container_width=True):
            start_core_evaluation(); st.rerun()
    with col_ctrl2:
        if st.button("⏹️ Stop Evaluation", disabled=not st.session_state.evaluation_running, help="Gracefully stop evaluation.", use_container_width=True):
            stop_evaluation(); st.rerun()
    with col_ctrl3:
        if st.session_state.evaluation_running:
            elapsed_time = time.time() - eval_start_time if eval_start_time else 0; duration_str = str(timedelta(seconds=int(elapsed_time)))
            st.info(f"⏳ Evaluation Running... Duration: {duration_str}"); st.progress(100)
        elif st.session_state.eval_duration_str and st.session_state.eval_duration_str != "Running...":
            st.success(f"✅ Evaluation Finished. Duration: {st.session_state.eval_duration_str}")
        if st.session_state.evaluation_error: st.error(f"Evaluation failed: {st.session_state.evaluation_error}")

    st.subheader("Console Output / Logs"); log_container = st.container(height=400, border=True)
    new_messages = []
    while not output_queue.empty():
        try:
            msg = output_queue.get_nowait(); new_messages.append(msg)
            if msg.startswith("STATUS:"):
                status = msg.split(":", 1)[1].strip()
                if status in ["Evaluation finished.", "Evaluation stopped by user."]:
                    st.session_state.evaluation_running = False
                    if eval_start_time: final_duration = time.time() - eval_start_time; st.session_state.eval_duration_str = str(timedelta(seconds=int(final_duration)))
                elif status.startswith("Evaluation failed"):
                    st.session_state.evaluation_running = False; st.session_state.evaluation_error = status.split(":", 1)[-1].strip()
                    if eval_start_time: final_duration = time.time() - eval_start_time; st.session_state.eval_duration_str = str(timedelta(seconds=int(final_duration)))
            elif msg.startswith("RESULTS_PATHS:"):
                try:
                    paths_json = msg.split(":", 1)[1]; paths_list = json.loads(paths_json); st.session_state.evaluation_results_paths = paths_list
                    if paths_list:
                        first_path = Path(paths_list[0]); run_folder = first_path.parent
                        if run_folder.exists() and run_folder.is_dir(): st.session_state.newly_completed_run_folder = run_folder; logger.info(f"Identified newly completed run folder: {run_folder}")
                        else: logger.warning(f"Could not determine run folder from path: {first_path}")
                except Exception as e: logger.error(f"Error processing RESULTS_PATHS message: {e}")
            elif msg.startswith("ERROR:") or "Traceback" in msg:
                if not st.session_state.evaluation_error: st.session_state.evaluation_error = msg.split("\n")[0]
        except queue.Empty: break
        except Exception as e: logger.error(f"Error processing message from queue: {e}"); new_messages.append(f"Error displaying log message: {e}")
    st.session_state.last_run_output.extend(new_messages)
    with log_container:
        for line in st.session_state.last_run_output:
            if line.startswith("ERROR:") or "Traceback" in line: st.error(line)
            elif line.startswith("WARNING:"): st.warning(line)
            elif line.startswith("SUCCESS:"): st.success(line)
            elif not line.startswith("STATUS:") and not line.startswith("RESULTS_PATHS:"): st.code(line, language="log")
    if st.session_state.evaluation_running: time.sleep(0.5); st.rerun()

# --- Main Application ---
def main() -> None:
    """Main function to run the Streamlit application."""
    initialize_session_state(); logger.debug("Session state initialized.")
    st.title("🧠 CogniBench Evaluation Runner")
    tab_run, tab_results = st.tabs(["🚀 Run Evaluation", "📊 View Results"])
    with tab_run:
        st.header("Setup and Run New Evaluation"); col_setup1, col_setup2 = st.columns(2)
        with col_setup1: render_file_uploader()
        with col_setup2: render_config_ui()
        if st.session_state.config_complete: render_config_summary()
        render_evaluation_progress(st.session_state.output_queue, st.session_state.last_run_output, st.session_state.eval_start_time)
    with tab_results:
        st.header("Analyze Evaluation Results")
        render_results_selector() # Moved selector here
        if not st.session_state.selected_results_folders: st.info("⬅️ Please select one or more result folders to view analysis.")
        else:
            results_df, summary_stats, failed_files = load_and_process_results(st.session_state.selected_results_folders)
            if failed_files:
                st.error(f"Failed to load some result files:")
                # FIX: Use st.code for each file path to handle potential long paths better
                for file in failed_files:
                    st.code(file)
            if results_df is not None:
                with st.expander("📈 Overall Summary Statistics", expanded=True): display_summary_stats(summary_stats)
                with st.expander("🚀 Performance Analysis (Aggregated Score)", expanded=False): display_performance_plots(results_df)
                with st.expander("📊 Rubric Score Analysis", expanded=False): display_rubric_plots(results_df) # Reverted version
                with st.expander("📋 Detailed Evaluation Results", expanded=False): display_results_table(results_df)
                with st.expander("👀 Tasks Flagged for Human Review", expanded=False): display_human_review_tasks(results_df)
            elif not failed_files: st.warning("No valid evaluation data found in the selected folder(s).")

# --- Entry Point ---
if __name__ == "__main__":
    main(); logger.debug("Streamlit main function finished.")
