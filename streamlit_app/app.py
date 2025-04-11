import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional  # Added imports

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

# --- Constants ---
APP_DIR = Path(__file__).parent
COGNIBENCH_ROOT = APP_DIR.parent
BASE_CONFIG_PATH = COGNIBENCH_ROOT / "config.yaml"
RUN_BATCH_SCRIPT_PATH = COGNIBENCH_ROOT / "scripts" / "run_batch_evaluation.py"
PROMPT_TEMPLATES_DIR_ABS = COGNIBENCH_ROOT / "prompts"

st.set_page_config(layout="wide", page_title="CogniBench Runner")

st.title("CogniBench Evaluation Runner")

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
    for uploaded_file in uploaded_files:
        st.write(f"- {uploaded_file.name}")
else:
    st.info("Please upload at least one batch file.")

# Placeholder for Folder Picker (if implemented later)
# st.write("*(Folder selection coming soon)*")

# --- Phase 1.5: Configuration Options ---
st.header("2. Configure LLM Judge")  # Add main header for section 2

# Define available models based on the plan
# Use requested provider names as keys
AVAILABLE_MODELS = {
    "OpenAI": {
        "GPT-4O": "gpt-4o",
        "GPT-4 Turbo": "gpt-4-turbo",
        "GPT-4": "gpt-4",
        "GPT-3.5 Turbo": "gpt-3.5-turbo",
    },
    "Anthropic": {
        "Claude 3.5 Haiku": "claude-3-5-haiku-latest",  # Placeholder ID
        "Claude 3.5 Sonnet": "claude-3-5-sonnet-20240620",
        "Claude 3.7 Sonnet": "claude-3-7-sonnet-latest",  # Placeholder ID
        "Claude 3 Opus": "claude-3-opus-20240229",
    },
    "Google": {
        "Gemini 1.5 Flash": "gemini-1.5-flash-latest",
        "Gemini 1.5 Pro": "gemini-1.5-pro-latest",
        "Gemini 2.0 Flash": "gemini-2.0-flash",  # Placeholder ID
        "Gemini 2.5 Pro Preview": "gemini-2.5-pro-preview-03-25",  # Placeholder ID
    },
    # "Local LLM": { # Keep commented out or add specific local models
    #     # Add local models here if needed
    # }
}

# Define available prompt templates
# Define available prompt templates using absolute path
try:
    prompt_files = [
        f for f in os.listdir(PROMPT_TEMPLATES_DIR_ABS) if f.endswith(".txt")
    ]
    # Store the relative path from CogniBench root for the config file
    AVAILABLE_TEMPLATES = {
        os.path.basename(f): str(Path("prompts") / f) for f in prompt_files
    }
except FileNotFoundError:
    st.error(
        f"Prompt templates directory not found at expected location: {PROMPT_TEMPLATES_DIR_ABS}"
    )
    AVAILABLE_TEMPLATES = {}


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
        list(AVAILABLE_TEMPLATES.keys())[0] if AVAILABLE_TEMPLATES else None
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


# --- Configuration Widgets (within Expander) ---
with st.expander(
    "Configuration Details", expanded=False
):  # Simplified label, main header is above
    col_config1, col_config2 = st.columns(2)

    with col_config1:
        # Provider Selection
        st.session_state.selected_provider = st.selectbox(
            "Select LLM Provider",
            options=list(AVAILABLE_MODELS.keys()),
            index=list(AVAILABLE_MODELS.keys()).index(
                st.session_state.selected_provider
            ),
            key="provider_select",  # Use key to avoid issues with re-rendering
        )

        # Model Selection (dynamic based on provider)
        available_model_names = list(
            AVAILABLE_MODELS[st.session_state.selected_provider].keys()
        )
        # Ensure the previously selected model is still valid for the new provider, else default
        current_model_index = 0
        if st.session_state.selected_model_name in available_model_names:
            current_model_index = available_model_names.index(
                st.session_state.selected_model_name
            )
        else:
            st.session_state.selected_model_name = available_model_names[
                0
            ]  # Default to first model of new provider

        st.session_state.selected_model_name = st.selectbox(
            "Select Judge Model",
            options=available_model_names,
            index=current_model_index,
            key="model_select",
        )

    with col_config2:
        # API Key Input
        st.session_state.api_key = st.text_input(
            "API Key (Optional)",
            type="password",
            placeholder="Leave blank to use environment variable",
            value=st.session_state.api_key,
            key="api_key_input",
        )

        # Prompt Template Selection
        if AVAILABLE_TEMPLATES:
            st.session_state.selected_template_name = st.selectbox(
                "Select Evaluation Prompt Template",
                options=list(AVAILABLE_TEMPLATES.keys()),
                index=list(AVAILABLE_TEMPLATES.keys()).index(
                    st.session_state.selected_template_name
                )
                if st.session_state.selected_template_name in AVAILABLE_TEMPLATES
                else 0,
                key="template_select",
            )
        else:
            st.warning(
                "No prompt templates found. Please add templates to the 'prompts/' directory."
            )
            st.session_state.selected_template_name = None

        # --- Buttons Side-by-Side ---
        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            # --- Button to View Template ---
            view_template_button = st.button(
                "View Selected Prompt Template",
                key="view_template_btn",
                disabled=not st.session_state.selected_template_name
                or not AVAILABLE_TEMPLATES,
                use_container_width=True # Make button fill column
            )

            if view_template_button:
                if st.session_state.selected_template_name and AVAILABLE_TEMPLATES:
                    try:
                        selected_template_rel_path = AVAILABLE_TEMPLATES[
                            st.session_state.selected_template_name
                        ]
                        selected_template_abs_path = (
                            COGNIBENCH_ROOT / selected_template_rel_path
                        )
                        if selected_template_abs_path.is_file():
                            template_content = selected_template_abs_path.read_text()

                            @st.dialog(
                                f"Prompt Template: {st.session_state.selected_template_name}"
                            )
                            def show_template_dialog():
                                st.text_area(
                                    "Template Content",
                                    value=template_content,
                                    height=400,  # Increased height for dialog
                                    disabled=True,
                                    key="dialog_template_content_display",
                                )
                                if st.button("Close", key="close_template_dialog"):
                                    st.rerun()  # Close dialog by rerunning

                            show_template_dialog()  # Call the dialog function

                        else:
                            # Display warning within the main expander if file not found
                            st.warning(
                                f"Selected template file not found: {selected_template_abs_path}"
                            )
                    except Exception as e:
                        # Display error within the main expander if reading fails
                        st.error(f"Error reading template file: {e}")
                else:
                    # Display warning within the main expander if no template selected
                    st.warning("Please select a template first.")

        with btn_col2:
            # --- Button to View Config ---
            view_config_button = st.button(
                "View config.yaml",
                key="view_config_btn",
                use_container_width=True # Make button fill column
            )

            if view_config_button:
                try:
                    if BASE_CONFIG_PATH.is_file():
                        config_content = BASE_CONFIG_PATH.read_text()

                        @st.dialog("Config File: config.yaml")
                        def show_config_dialog():
                            st.text_area(
                                "Config Content",
                                value=config_content,
                                height=400,
                                disabled=True,
                                key="dialog_config_content_display",
                            )
                            if st.button("Close", key="close_config_dialog"):
                                st.rerun() # Close dialog by rerunning

                        show_config_dialog() # Call the dialog function

                    else:
                        # Display warning within the main expander if file not found
                        st.warning(f"Config file not found: {BASE_CONFIG_PATH}")
                except Exception as e:
                    # Display error within the main expander if reading fails
                    st.error(f"Error reading config file: {e}")

# --- Display Current Configuration Summary (Moved Outside Expander) ---
st.subheader("Current Judge Configuration Summary")
if st.session_state.selected_template_name:
    selected_model_api_id = AVAILABLE_MODELS[st.session_state.selected_provider][
        st.session_state.selected_model_name
    ]
    selected_template_path = AVAILABLE_TEMPLATES[
        st.session_state.selected_template_name
    ]
    api_key_status = (
        "**Provided**" if st.session_state.api_key else "**Using Environment Variable**"
    )

    # Improved formatting using markdown with line breaks
    st.markdown(f"""
**Provider:** `{st.session_state.selected_provider}`\n
**Model:** `{st.session_state.selected_model_name}` (`{selected_model_api_id}`)\n
**API Key:** {api_key_status}\n
**Prompt Template:** `{selected_template_path}`
""")
else:
    st.warning(
        "Configuration incomplete: Please select a provider, model, and prompt template in the 'Configuration Details' section above."
    )


# Separator moved outside the expander, before the next section
st.markdown("---")
# Removed duplicate/erroneous lines

# --- Phase 2: Run Evaluation ---
st.header("3. Run Evaluations")  # Renamed header

col1, col2 = st.columns(2)

with col1:
    run_button = st.button(
        "🚀 Run Evaluations",  # Renamed button text
        type="primary",
        disabled=not uploaded_files
        or not st.session_state.selected_template_name
        or st.session_state.get("evaluation_running", False),
    )

with col2:
    clear_cache_button = st.button("🧹 Clear LLM Cache")


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
def run_evaluation_script(input_file_path, config_file_path, output_queue):
    """Runs the batch evaluation script and puts output lines into a queue."""
    # output_queue.put("DEBUG (Thread): Entered run_evaluation_script function.") # Keep commented
    command = [
        sys.executable,  # Use the same python interpreter running streamlit
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
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            text=True,
            encoding="utf-8",
            errors="replace",  # Handle potential encoding errors
            cwd=COGNIBENCH_ROOT,  # Run script from the project root
            bufsize=1,  # Line buffered
        )

        # Read stdout line by line
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                output_queue.put(line.strip())
            process.stdout.close()

        process.wait()  # Wait for the process to complete
        output_queue.put(f"INFO: Process finished with exit code {process.returncode}")

    except FileNotFoundError:
        output_queue.put(
            f"ERROR: Python executable or script not found. Command: {' '.join(command)}"
        )
    except Exception as e:
        output_queue.put(f"ERROR: An unexpected error occurred: {e}")
    finally:
        # output_queue.put("DEBUG (Thread): Reached finally block, putting None.") # Keep commented
        output_queue.put(None)  # Signal completion


# --- Run Evaluation Logic ---
if run_button and uploaded_files and st.session_state.selected_template_name:
    st.session_state.evaluation_running = True
    st.session_state.previous_evaluation_running = False  # Reset previous state tracker
    st.session_state.last_run_output = []
    st.session_state.evaluation_results_paths = []
    st.session_state.results_df = None
    st.session_state.eval_start_time = time.time()
    st.session_state.eval_duration_str = None
    st.session_state.current_file_index = 0
    st.session_state.output_queue = queue.Queue()
    st.session_state.worker_thread = None
    st.session_state.temp_dir = tempfile.TemporaryDirectory()
    st.session_state.temp_dir_path = Path(st.session_state.temp_dir.name)
    st.rerun()  # Rerun to disable button and show progress area

if st.session_state.get("evaluation_running", False):
    st.header("Evaluation Progress")
    progress_area = st.container()
    # Add progress bar and text
    progress_bar = progress_area.progress(0.0)  # Initialize with float
    progress_text = progress_area.text("Starting evaluation...")
    log_expander = st.expander("Show Full Logs", expanded=False)  # Collapsed by default
    log_placeholder = log_expander.empty()
    # Display current logs immediately
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
            selected_template_path = AVAILABLE_TEMPLATES[
                st.session_state.selected_template_name
            ]
            api_key = st.session_state.api_key

            config_data["evaluation_settings"]["judge_model"] = selected_model_api_id
            config_data["evaluation_settings"]["prompt_template"] = (
                selected_template_path
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

            # 5. Start evaluation thread
            st.session_state.output_queue = queue.Queue()
            st.session_state.worker_thread = threading.Thread(
                target=run_evaluation_script,
                args=(temp_input_path, temp_config_path, st.session_state.output_queue),
                daemon=True,
            )
            st.session_state.worker_thread.start()
            progress_area.info(f"Started evaluation thread for {uploaded_file.name}...")
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

            # Parse for results file path
            match = re.search(
                r"Successfully combined ingested data and evaluations into (.*_final_results\.json)",
                line,
            )
            if match:
                # st.write(f"DEBUG (Log Check): Regex matched! Group 1: '{match.group(1).strip()}'") # Keep commented
                results_path = match.group(1).strip()
                try:
                    abs_path = Path(results_path)
                    if abs_path.is_absolute():
                        if COGNIBENCH_ROOT in abs_path.parents:
                            results_path = str(abs_path.relative_to(COGNIBENCH_ROOT))
                        else:
                            results_path = str(abs_path)
                    else:
                        results_path = str(Path(results_path))

                    if results_path not in st.session_state.evaluation_results_paths:
                        # st.write(f"DEBUG (Log Check): Appending path: '{results_path}'") # Keep commented
                        st.session_state.evaluation_results_paths.append(results_path)
                        progress_area.success(f"Found results file: {results_path}")
                except Exception as path_e:
                    progress_area.warning(
                        f"Could not process results path '{match.group(1).strip()}': {path_e}"
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
            # Calculate duration BEFORE cleaning up state
            end_time = time.time()
            duration_seconds = end_time - st.session_state.eval_start_time

            # Format duration human-readably
            parts = []
            hours, rem = divmod(duration_seconds, 3600)
            minutes, seconds = divmod(rem, 60)
            if hours >= 1:
                parts.append(f"{int(hours)} hour{'s' if int(hours) != 1 else ''}")
            if minutes >= 1:
                parts.append(f"{int(minutes)} minute{'s' if int(minutes) != 1 else ''}")
            if (
                seconds >= 1 or not parts
            ):  # Show seconds if > 0 or if it's the only unit
                parts.append(f"{int(seconds)} second{'s' if int(seconds) != 1 else ''}")
            st.session_state.eval_duration_str = ", ".join(parts)
            # Example: "1 hour, 5 minutes, 30 seconds" or "5 minutes, 30 seconds" or "30 seconds"
            # Clean up state AFTER calculations but before final display update
            if "current_file_index" in st.session_state:
                del st.session_state.current_file_index
            if "output_queue" in st.session_state:
                del st.session_state.output_queue
            if "worker_thread" in st.session_state:
                del st.session_state.worker_thread
            # Display final messages
            progress_area.success(
                f"All evaluations complete! (Duration: {st.session_state.eval_duration_str})"
            )
            if st.session_state.evaluation_results_paths:
                progress_area.write("Found results files:")
                for path in st.session_state.evaluation_results_paths:
                    progress_area.code(path)
            else:
                progress_area.warning(
                    "Could not find paths to results files in the logs."
                )
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
if clear_cache_button:
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


# --- Function to load and process results ---
@st.cache_data(show_spinner=False)  # Re-enable cache
def load_and_process_results(results_paths):
    """Loads data from _final_results.json files and processes into a DataFrame."""
    all_results_data = []
    for relative_path in results_paths:
        try:
            file_path = COGNIBENCH_ROOT / relative_path
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                task_count = 0
                for task in data:
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
                    # if eval_count == 0: # Keep commented unless debugging load function
                    #     st.write(f"DEBUG (load_and_process_results):   No evaluations found for task {task.get('task_id')}")
                # if task_count == 0: # Keep commented unless debugging load function
                #     st.write(f"DEBUG (load_and_process_results): No tasks found in file {relative_path}")
        except FileNotFoundError:
            st.error(f"Results file not found: {relative_path}")
            return None
        except json.JSONDecodeError:
            st.error(f"Error decoding JSON from file: {relative_path}")
            return None
        except Exception as e:
            st.error(f"Error processing file {relative_path}: {e}")
            return None

    if not all_results_data:
        return None

    df = pd.DataFrame(all_results_data)
    return df


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
if st.session_state.results_df is not None:
    df = st.session_state.results_df
    st.success(f"Loaded {len(df)} evaluation results.")
    # Display evaluation duration if available
    if st.session_state.get("eval_duration_str"):
        st.info(
            f"Total evaluation time: {st.session_state.eval_duration_str}"
        )  # Display the new format

    # --- Filters ---
    st.sidebar.header("Filters")
    available_models = (
        df["model_id"].unique().tolist() if "model_id" in df.columns else []
    )
    available_tasks = df["task_id"].unique().tolist() if "task_id" in df.columns else []
    available_subjects = (
        df["subject"].unique().tolist() if "subject" in df.columns else []
    )

    selected_models = st.sidebar.multiselect(
        "Filter by Model:", available_models, default=available_models
    )
    selected_tasks = st.sidebar.multiselect(
        "Filter by Task ID:", available_tasks, default=[]
    )
    selected_subjects = st.sidebar.multiselect(
        "Filter by Subject:", available_subjects, default=available_subjects
    )

    # Apply filters
    filtered_df = df.copy()
    if selected_models:
        filtered_df = filtered_df[filtered_df["model_id"].isin(selected_models)]
    if selected_tasks:
        filtered_df = filtered_df[filtered_df["task_id"].isin(selected_tasks)]
    if selected_subjects:
        filtered_df = filtered_df[filtered_df["subject"].isin(selected_subjects)]

    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
    else:
        # --- Overall Performance Chart (Using aggregated_score) ---
        st.subheader("Overall Performance by Model")
        agg_score_col = "aggregated_score"
        if agg_score_col in filtered_df.columns and "model_id" in filtered_df.columns:
            performance_counts = (
                filtered_df.groupby(["model_id", agg_score_col])
                .size()
                .reset_index(name="count")
            )
            category_orders = {agg_score_col: ["Pass", "Fail", "N/A"]}
            color_map = {"Pass": "green", "Fail": "red", "N/A": "grey"}

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
                color_discrete_map=color_map,
            )
            st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.warning(
                f"Could not generate Overall Performance chart. Required columns ('model_id', '{agg_score_col}') not found."
            )

        # --- Rubric Score Breakdown ---
        st.subheader("Rubric Score Analysis")
        rubric_cols = [
            col
            for col in filtered_df.columns
            if col.startswith("judge_rubric_") and col.endswith("_score")
        ]

        if rubric_cols and "model_id" in filtered_df.columns:
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

            # --- Rubric Score Distribution per Model (New Graph) ---
            st.subheader("Rubric Score Distribution per Model")
            # Reuse rubric_counts from previous step
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
            # --- Human Review Status ---
            st.subheader("Human Review Status")
            review_status_col = (
                "judge_human_review_status"  # Assuming this is the column name
            )
            if (
                review_status_col in filtered_df.columns
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
                color_map = {
                    "Needs Review": "orange",
                    "Not Required": "lightblue",
                    "N/A": "grey",
                }  # Use actual values

                # --- Create Graph ---
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
                    color_discrete_map=color_map,
                )
                st.plotly_chart(fig_review, use_container_width=True)

                # --- Human Review Explorer ---
                st.subheader("Tasks Flagged for Human Review")
                # Display the review_needed_df created *before* fillna
                if not review_needed_df.empty:
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
                    st.info(
                        "No tasks flagged for human review based on current filters."
                    )

            else:
                st.warning(
                    f"Could not generate Human Review Status chart/explorer. Required columns ('model_id', '{review_status_col}') not found."
                )
        # This else corresponds to the `if rubric_cols and "model_id" in filtered_df.columns:` check
        else:
            st.warning(
                "Could not generate Rubric Score Analysis / Human Review sections. Rubric score columns not found or 'model_id' missing."
            )

        # --- Task-Level Explorer (Should be outside the rubric/review if/else, but inside the main results display else) ---
        st.subheader("Task-Level Explorer")
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
