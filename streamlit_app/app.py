import streamlit as st
import os
import subprocess
import tempfile
import yaml
from pathlib import Path
import sys
import threading
import queue
import re
import time # For demo spinner

# --- Constants ---
APP_DIR = Path(__file__).parent
COGNIBENCH_ROOT = APP_DIR.parent
BASE_CONFIG_PATH = COGNIBENCH_ROOT / "config.yaml"
RUN_BATCH_SCRIPT_PATH = COGNIBENCH_ROOT / "scripts" / "run_batch_evaluation.py"
PROMPT_TEMPLATES_DIR_ABS = COGNIBENCH_ROOT / "prompts"

st.set_page_config(layout="wide", page_title="CogniBench Runner")

st.title("CogniBench Evaluation Runner")

# --- Phase 1: Input Selection ---
st.header("1. Upload Batch File(s)")

uploaded_files = st.file_uploader(
    "Select CogniBench JSON batch file(s)",
    type=['json'],
    accept_multiple_files=True,
    help="Upload one or more JSON files containing tasks for evaluation."
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
st.header("2. Configure Evaluation")

# Define available models based on the plan
AVAILABLE_MODELS = {
    "openai": {
        "GPT-4O": "gpt-4o",
        "GPT-4 Turbo": "gpt-4-turbo",
        "GPT-4": "gpt-4",
        "GPT-3.5 Turbo": "gpt-3.5-turbo",
    },
    "anthropic": {
        "Claude 3.5 Haiku": "claude-3-5-haiku-latest", # Placeholder ID
        "Claude 3.5 Sonnet": "claude-3-5-sonnet-20240620",
        "Claude 3.7 Sonnet": "claude-3-7-sonnet-latest", # Placeholder ID
        "Claude 3 Opus": "claude-3-opus-20240229",
    },
    "google": {
        "Gemini 1.5 Flash": "gemini-1.5-flash-latest",
        "Gemini 1.5 Pro": "gemini-1.5-pro-latest",
        "Gemini 2.0 Flash": "gemini-2.0-flash", # Placeholder ID
        "Gemini 2.5 Pro Preview": "gemini-2.5-pro-preview-03-25", # Placeholder ID
    },
    "local_llm": {
        # Add local models here if needed
    }
}

# Define available prompt templates
# Define available prompt templates using absolute path
try:
    prompt_files = [f for f in os.listdir(PROMPT_TEMPLATES_DIR_ABS) if f.endswith('.txt')]
    # Store the relative path from CogniBench root for the config file
    AVAILABLE_TEMPLATES = {
        os.path.basename(f): str(Path("prompts") / f) for f in prompt_files
    }
except FileNotFoundError:
    st.error(f"Prompt templates directory not found at expected location: {PROMPT_TEMPLATES_DIR_ABS}")
    AVAILABLE_TEMPLATES = {}


# Initialize session state
if 'selected_provider' not in st.session_state:
    st.session_state.selected_provider = list(AVAILABLE_MODELS.keys())[0] # Default to openai
if 'selected_model_name' not in st.session_state:
    st.session_state.selected_model_name = list(AVAILABLE_MODELS[st.session_state.selected_provider].keys())[0]
if 'selected_template_name' not in st.session_state:
    st.session_state.selected_template_name = list(AVAILABLE_TEMPLATES.keys())[0] if AVAILABLE_TEMPLATES else None
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
    if 'evaluation_running' not in st.session_state:
        st.session_state.evaluation_running = False
    if 'evaluation_results_paths' not in st.session_state:
        st.session_state.evaluation_results_paths = []
    if 'last_run_output' not in st.session_state:
        st.session_state.last_run_output = []

# --- Configuration Widgets ---
col_config1, col_config2 = st.columns(2)

with col_config1:
    # Provider Selection
    st.session_state.selected_provider = st.selectbox(
        "Select LLM Provider",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(st.session_state.selected_provider),
        key="provider_select" # Use key to avoid issues with re-rendering
    )

    # Model Selection (dynamic based on provider)
    available_model_names = list(AVAILABLE_MODELS[st.session_state.selected_provider].keys())
    # Ensure the previously selected model is still valid for the new provider, else default
    current_model_index = 0
    if st.session_state.selected_model_name in available_model_names:
        current_model_index = available_model_names.index(st.session_state.selected_model_name)
    else:
        st.session_state.selected_model_name = available_model_names[0] # Default to first model of new provider

    st.session_state.selected_model_name = st.selectbox(
        "Select Judge Model",
        options=available_model_names,
        index=current_model_index,
        key="model_select"
    )

with col_config2:
    # API Key Input
    st.session_state.api_key = st.text_input(
        "API Key (Optional)",
        type="password",
        placeholder="Leave blank to use environment variable",
        value=st.session_state.api_key,
        key="api_key_input"
    )

    # Prompt Template Selection
    if AVAILABLE_TEMPLATES:
        st.session_state.selected_template_name = st.selectbox(
            "Select Evaluation Prompt Template",
            options=list(AVAILABLE_TEMPLATES.keys()),
            index=list(AVAILABLE_TEMPLATES.keys()).index(st.session_state.selected_template_name) if st.session_state.selected_template_name in AVAILABLE_TEMPLATES else 0,
            key="template_select"
        )
    else:
        st.warning("No prompt templates found. Please add templates to the 'prompts/' directory.")
        st.session_state.selected_template_name = None


# --- Display Current Configuration ---
st.subheader("Current Configuration:")
if st.session_state.selected_template_name:
    selected_model_api_id = AVAILABLE_MODELS[st.session_state.selected_provider][st.session_state.selected_model_name]
    selected_template_path = AVAILABLE_TEMPLATES[st.session_state.selected_template_name]
    api_key_status = "Provided" if st.session_state.api_key else "Using Environment Variable"

    st.json({
        "Provider": st.session_state.selected_provider,
        "Model Name": st.session_state.selected_model_name,
        "Model API ID": selected_model_api_id,
        "API Key Status": api_key_status,
        "Prompt Template": selected_template_path
    })
else:
    st.warning("Configuration incomplete due to missing prompt templates.")

# --- Phase 2: Run Evaluation ---
st.header("3. Run Evaluation")

col1, col2 = st.columns(2)

with col1:
    run_button = st.button(
        "🚀 Run Evaluation",
        type="primary",
        disabled=not uploaded_files or not st.session_state.selected_template_name or st.session_state.get('evaluation_running', False)
    )

with col2:
    clear_cache_button = st.button("🧹 Clear LLM Cache")

# --- Function to run evaluation in a separate thread ---
def run_evaluation_script(input_file_path, config_file_path, output_queue):
    """Runs the batch evaluation script and puts output lines into a queue."""
    command = [
        sys.executable, # Use the same python interpreter running streamlit
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
            stderr=subprocess.STDOUT, # Redirect stderr to stdout
            text=True,
            encoding='utf-8',
            errors='replace', # Handle potential encoding errors
            cwd=COGNIBENCH_ROOT, # Run script from the project root
            bufsize=1 # Line buffered
        )

        # Read stdout line by line
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                output_queue.put(line.strip())
            process.stdout.close()

        process.wait() # Wait for the process to complete
        output_queue.put(f"INFO: Process finished with exit code {process.returncode}")

    except FileNotFoundError:
        output_queue.put(f"ERROR: Python executable or script not found. Command: {' '.join(command)}")
    except Exception as e:
        output_queue.put(f"ERROR: An unexpected error occurred: {e}")
    finally:
        output_queue.put(None) # Signal completion

# --- Run Evaluation Logic ---
if run_button and uploaded_files and st.session_state.selected_template_name:
    st.session_state.evaluation_running = True
    st.session_state.last_run_output = []
    st.session_state.evaluation_results_paths = []
    st.rerun() # Rerun to disable button and show progress area

if st.session_state.get('evaluation_running', False):
    st.header("Evaluation Progress")
    progress_area = st.container()
    log_expander = st.expander("Show Full Logs", expanded=False)
    log_placeholder = log_expander.empty()

    # Check if this is the first run in the evaluation sequence
    if 'current_file_index' not in st.session_state:
        st.session_state.current_file_index = 0
        st.session_state.output_queue = queue.Queue()
        st.session_state.worker_thread = None
        st.session_state.temp_dir = tempfile.TemporaryDirectory()
        st.session_state.temp_dir_path = Path(st.session_state.temp_dir.name)

    current_index = st.session_state.current_file_index
    total_files = len(uploaded_files)

    # --- Start worker thread if not already running for the current file ---
    if st.session_state.worker_thread is None:
        if current_index < total_files:
            uploaded_file = uploaded_files[current_index]
            progress_area.info(f"Processing file {current_index + 1}/{total_files}: **{uploaded_file.name}**")

            # 1. Save uploaded file temporarily
            temp_input_path = st.session_state.temp_dir_path / uploaded_file.name
            with open(temp_input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 2. Load and modify config
            try:
                with open(BASE_CONFIG_PATH, 'r') as f:
                    config_data = yaml.safe_load(f)

                # Apply UI selections
                selected_provider = st.session_state.selected_provider
                selected_model_name = st.session_state.selected_model_name
                selected_model_api_id = AVAILABLE_MODELS[selected_provider][selected_model_name]
                selected_template_path = AVAILABLE_TEMPLATES[st.session_state.selected_template_name]
                api_key = st.session_state.api_key

                config_data['evaluation_settings']['judge_model'] = selected_model_api_id
                config_data['evaluation_settings']['prompt_template'] = selected_template_path
                # Update client config as well, assuming judge uses it
                config_data['llm_client']['provider'] = selected_provider
                config_data['llm_client']['model'] = selected_model_api_id
                if api_key:
                    config_data['llm_client']['api_key'] = api_key
                else:
                    # Attempt to set appropriate env var placeholder if needed (basic example)
                    if selected_provider == 'openai':
                        config_data['llm_client']['api_key'] = '${OPENAI_API_KEY}'
                    elif selected_provider == 'anthropic':
                         config_data['llm_client']['api_key'] = '${ANTHROPIC_API_KEY}' # Assuming this var name
                    elif selected_provider == 'google':
                         config_data['llm_client']['api_key'] = '${GOOGLE_API_KEY}' # Assuming this var name
                    # Add more providers as needed

                # 3. Save temporary config
                temp_config_path = st.session_state.temp_dir_path / f"temp_config_{current_index}.yaml"
                with open(temp_config_path, 'w') as f:
                    yaml.dump(config_data, f)

                # 4. Start evaluation thread
                st.session_state.output_queue = queue.Queue() # Reset queue for new file
                st.session_state.worker_thread = threading.Thread(
                    target=run_evaluation_script,
                    args=(temp_input_path, temp_config_path, st.session_state.output_queue),
                    daemon=True
                )
                st.session_state.worker_thread.start()
                progress_area.info(f"Started evaluation for {uploaded_file.name}...")

            except Exception as e:
                progress_area.error(f"Error preparing evaluation for {uploaded_file.name}: {e}")
                st.session_state.worker_thread = "Error" # Mark as error to stop processing

        else: # All files processed
             st.session_state.evaluation_running = False
             st.session_state.temp_dir.cleanup()
             del st.session_state.temp_dir # Allow garbage collection
             del st.session_state.temp_dir_path
             del st.session_state.current_file_index
             del st.session_state.output_queue
             del st.session_state.worker_thread
             progress_area.success("All evaluations complete!")
             if st.session_state.evaluation_results_paths:
                 progress_area.write("Found results files:")
                 for path in st.session_state.evaluation_results_paths:
                     progress_area.code(path)
             else:
                 progress_area.warning("Could not find paths to results files in the logs.")
             st.rerun() # Final rerun to update UI state

    # --- Process output queue while thread is running ---
    if isinstance(st.session_state.worker_thread, threading.Thread) and st.session_state.worker_thread.is_alive():
        log_lines = []
        while not st.session_state.output_queue.empty():
            line = st.session_state.output_queue.get()
            if line is None: # End signal from thread
                st.session_state.worker_thread.join() # Ensure thread finishes
                st.session_state.worker_thread = None # Mark as finished
                st.session_state.current_file_index += 1 # Move to next file
                st.rerun() # Rerun to process next file or finish
                break
            else:
                st.session_state.last_run_output.append(line)
                log_lines.append(line)
                # Basic progress update (can be improved if backend prints specific format)
                if "Evaluating task" in line:
                     progress_area.info(line) # Show task progress directly
                # Parse for results file path
                match = re.search(r"Successfully combined ingested data and evaluations into (.*_final_results\.json)", line)
                if match:
                    results_path = match.group(1).strip()
                    # Make path relative to CogniBench root if possible
                    try:
                        abs_path = Path(results_path)
                        if abs_path.is_absolute():
                             # Check if it's within the CogniBench root
                             if COGNIBENCH_ROOT in abs_path.parents:
                                 results_path = str(abs_path.relative_to(COGNIBENCH_ROOT))
                             else: # Keep absolute if outside project
                                 results_path = str(abs_path)
                        else: # Assume relative to CogniBench root already
                             results_path = str(Path(results_path)) # Normalize

                        if results_path not in st.session_state.evaluation_results_paths:
                            st.session_state.evaluation_results_paths.append(results_path)
                            progress_area.success(f"Found results file: {results_path}")
                    except Exception as path_e:
                         progress_area.warning(f"Could not process results path '{match.group(1).strip()}': {path_e}")


        # Update log display (limit history for performance)
        log_placeholder.code("\n".join(st.session_state.last_run_output[-1000:]), language="log")
        time.sleep(0.1) # Small delay to prevent busy-waiting
        st.rerun() # Rerun to check queue again

    elif st.session_state.worker_thread == "Error":
         # Handle error state - stop processing
         st.session_state.evaluation_running = False
         st.session_state.temp_dir.cleanup()
         del st.session_state.temp_dir
         del st.session_state.temp_dir_path
         del st.session_state.current_file_index
         del st.session_state.output_queue
         del st.session_state.worker_thread
         st.rerun()


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
        st.warning(f"Cleared {cleared_count} cache file(s), but encountered errors with: {', '.join(errors)}")

# --- Phase 3: Visualize Results (Placeholder) ---
st.header("4. Results")
st.info("Evaluation results will be displayed here after running (Phase 3).")
# TODO: Add results visualization