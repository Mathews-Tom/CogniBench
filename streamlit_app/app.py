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
import time
import pandas as pd
import plotly.express as px
import json

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
        if 'results_df' not in st.session_state:
            st.session_state.results_df = None # To store the loaded DataFrame

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
    # Add progress bar and text
    progress_bar = progress_area.progress(0.0) # Initialize with float
    progress_text = progress_area.text("Starting evaluation...")
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
                # Progress update based on backend message
                progress_match = re.match(r"PROGRESS: Task (\d+)/(\d+)", line)
                if progress_match:
                    current_task = int(progress_match.group(1))
                    total_tasks_backend = int(progress_match.group(2))
                    # Ensure total_tasks_backend is not zero
                    if total_tasks_backend > 0:
                        progress_percentage = float(current_task) / total_tasks_backend
                        progress_bar.progress(progress_percentage) # Update progress bar
                        progress_text.text(f"Evaluating Task {current_task}/{total_tasks_backend}...") # Update text
                    else:
                        progress_text.text(f"Evaluating Task {current_task}/Unknown Total...")
                elif "Evaluating task" in line: # Fallback if specific format not found
                     progress_text.text(line) # Show task progress directly in text area
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

# --- Phase 3: Visualize Results ---
st.header("4. Results")

# --- Function to load and process results ---
@st.cache_data # Cache the loaded data
def load_and_process_results(results_paths):
    """Loads data from _final_results.json files and processes into a DataFrame."""
    all_results_data = []
    for relative_path in results_paths:
        try:
            # Construct absolute path relative to CogniBench root
            file_path = COGNIBENCH_ROOT / relative_path
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Data is a list of tasks
                for task in data:
                    task_id = task.get("task_id")
                    prompt = task.get("prompt")
                    ideal_response = task.get("ideal_response")
                    final_answer_gt = task.get("final_answer") # Ground truth final answer
                    metadata = task.get("metadata", {})
                    subject = metadata.get("subject", "N/A")
                    complexity = metadata.get("complexity", "N/A")

                    for evaluation in task.get("evaluations", []):
                        model_id = evaluation.get("model_id")
                        model_response = evaluation.get("model_response")
                        human_eval = evaluation.get("human_evaluation", {})
                        judge_eval = evaluation.get("judge_evaluation", {})

                        # Flatten judge_evaluation details (rubric scores, etc.)
                        flat_judge_eval = {}
                        if isinstance(judge_eval, dict):
                             for key, value in judge_eval.items():
                                 # Handle nested dicts like 'rubric_scores' if they exist
                                 if isinstance(value, dict):
                                     for sub_key, sub_value in value.items():
                                         flat_judge_eval[f"judge_{key}_{sub_key}"] = sub_value
                                 else:
                                     flat_judge_eval[f"judge_{key}"] = value
                        else:
                             # Handle case where judge_eval might not be a dict
                             flat_judge_eval['judge_evaluation_raw'] = judge_eval


                        # Determine overall Pass/Fail/Partial based on judge rubric scores
                        # Assumes judge_evaluation contains keys like 'judge_problem_understanding', 'judge_logical_implications', etc.
                        # with values 'Yes', 'No', 'Partial'. Adjust logic based on actual keys.
                        overall_status = "Pass" # Default
                        rubric_keys = [k for k in flat_judge_eval if k.startswith('judge_rubric_')] # Example prefix
                        if not rubric_keys: # Fallback if specific rubric keys aren't found
                            rubric_keys = [k for k in flat_judge_eval if k.startswith('judge_') and k != 'judge_evaluation_id']

                        has_no = False
                        has_partial = False
                        for r_key in rubric_keys:
                            score = str(flat_judge_eval.get(r_key, '')).lower()
                            if score == 'no':
                                has_no = True
                                break # 'No' overrides everything
                            elif score == 'partial':
                                has_partial = True

                        if has_no:
                            overall_status = "Fail"
                        elif has_partial:
                            overall_status = "Partial"


                        all_results_data.append({
                            "task_id": task_id,
                            "model_id": model_id,
                            "subject": subject,
                            "complexity": complexity,
                            "overall_status": overall_status,
                            "prompt": prompt,
                            "ideal_response": ideal_response,
                            "model_response": model_response,
                            "final_answer_ground_truth": final_answer_gt,
                            **flat_judge_eval, # Add flattened judge scores
                            # Add human eval fields if needed
                            "human_preference": human_eval.get("preference"),
                            "human_rating": human_eval.get("rating"),
                        })
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

    return pd.DataFrame(all_results_data)

# --- Load data if results paths exist ---
if st.session_state.evaluation_results_paths and st.session_state.results_df is None:
    with st.spinner("Loading and processing results..."):
        st.session_state.results_df = load_and_process_results(st.session_state.evaluation_results_paths)

# --- Display Results if DataFrame is loaded ---
if st.session_state.results_df is not None:
    df = st.session_state.results_df

    st.success(f"Loaded {len(df)} evaluation results.")

    # --- Filters ---
    st.sidebar.header("Filters")
    # Get unique values, handling potential missing columns gracefully
    available_models = df['model_id'].unique().tolist() if 'model_id' in df.columns else []
    available_tasks = df['task_id'].unique().tolist() if 'task_id' in df.columns else []
    available_subjects = df['subject'].unique().tolist() if 'subject' in df.columns else []

    selected_models = st.sidebar.multiselect("Filter by Model:", available_models, default=available_models)
    selected_tasks = st.sidebar.multiselect("Filter by Task ID:", available_tasks, default=[])
    selected_subjects = st.sidebar.multiselect("Filter by Subject:", available_subjects, default=available_subjects)

    # Apply filters
    filtered_df = df.copy()
    if selected_models:
        filtered_df = filtered_df[filtered_df['model_id'].isin(selected_models)]
    if selected_tasks:
        filtered_df = filtered_df[filtered_df['task_id'].isin(selected_tasks)]
    if selected_subjects:
        filtered_df = filtered_df[filtered_df['subject'].isin(selected_subjects)]

    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
    else:
        # --- Overall Performance Chart ---
        st.subheader("Overall Performance by Model")
        if 'overall_status' in filtered_df.columns and 'model_id' in filtered_df.columns:
            performance_counts = filtered_df.groupby(['model_id', 'overall_status']).size().reset_index(name='count')
            fig_perf = px.bar(performance_counts, x='model_id', y='count', color='overall_status',
                              title="Evaluation Status Count per Model",
                              labels={'model_id': 'Model', 'count': 'Number of Tasks', 'overall_status': 'Overall Status'},
                              barmode='group',
                              color_discrete_map={'Pass': 'green', 'Partial': 'orange', 'Fail': 'red'})
            st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.warning("Could not generate Overall Performance chart. Required columns ('model_id', 'overall_status') not found.")


        # --- Rubric Score Breakdown ---
        st.subheader("Rubric Score Analysis")
        # Identify rubric score columns (adjust prefix/logic if needed)
        rubric_cols = [col for col in filtered_df.columns if col.startswith('judge_rubric_') or (col.startswith('judge_') and col not in ['judge_evaluation_id', 'judge_evaluation_raw'])]
        # Exclude columns that might not be rubric scores if the fallback was used
        rubric_cols = [col for col in rubric_cols if col not in ['judge_model_final_answer', 'judge_final_answer_correct']] # Example exclusions

        if rubric_cols and 'model_id' in filtered_df.columns:
            rubric_melted = filtered_df.melt(id_vars=['model_id'], value_vars=rubric_cols, var_name='rubric_criterion', value_name='score')
            # Clean criterion names for display
            rubric_melted['rubric_criterion'] = rubric_melted['rubric_criterion'].str.replace('judge_rubric_', '').str.replace('judge_', '').str.replace('_', ' ').str.title()
            rubric_counts = rubric_melted.groupby(['model_id', 'rubric_criterion', 'score']).size().reset_index(name='count')

            fig_rubric = px.bar(rubric_counts, x='rubric_criterion', y='count', color='score',
                                title="Rubric Score Distribution per Criterion",
                                labels={'rubric_criterion': 'Rubric Criterion', 'count': 'Count', 'score': 'Score'},
                                barmode='group',
                                facet_col='model_id', # Show one chart per model
                                category_orders={"score": ["Yes", "Partial", "No", "N/A"]}, # Ensure consistent order
                                color_discrete_map={'Yes': 'green', 'Partial': 'orange', 'No': 'red', 'N/A': 'grey'})
            fig_rubric.update_xaxes(tickangle=45)
            st.plotly_chart(fig_rubric, use_container_width=True)
        else: # Corresponds to 'if rubric_cols and 'model_id' in filtered_df.columns:'
            st.warning("Could not generate Rubric Score Analysis. Rubric score columns not found or 'model_id' missing.")

        # --- Score Distribution Chart ---
        st.subheader("Score Distribution Analysis")
        # Example: Using 'human_rating' if it's numerical
        score_col_to_plot = 'human_rating'
        if score_col_to_plot in filtered_df.columns:
            # Ensure the column is numeric, coercing errors to NaN
            numeric_scores = pd.to_numeric(filtered_df[score_col_to_plot], errors='coerce')
            numeric_scores = numeric_scores.dropna() # Remove non-numeric entries

            if not numeric_scores.empty:
                fig_dist = px.histogram(numeric_scores, x=score_col_to_plot,
                                        title=f"Distribution of {score_col_to_plot.replace('_', ' ').title()}",
                                        labels={score_col_to_plot: score_col_to_plot.replace('_', ' ').title()},
                                        marginal="box", # Add box plot marginal
                                        nbins=20) # Adjust number of bins as needed
                st.plotly_chart(fig_dist, use_container_width=True)
            else:
                st.info(f"No valid numerical data found in '{score_col_to_plot}' column for distribution plot.")
        else:
            st.info(f"Column '{score_col_to_plot}' not found in results, skipping distribution plot.")

        # This section was duplicated and misplaced, removing it.

        # --- Task-Level Explorer ---
        st.subheader("Task-Level Explorer")
        # Select and rename columns for better readability
        display_cols = {
            "task_id": "Task ID",
            "model_id": "Model",
            "subject": "Subject",
            "complexity": "Complexity",
            "overall_status": "Overall Status",
            # Add more judge scores if needed, e.g.:
            # "judge_problem_understanding": "Judge: Problem Understanding",
            # "judge_logical_implications": "Judge: Logical Implications",
            "human_preference": "Human Preference",
            "human_rating": "Human Rating",
            "prompt": "Prompt",
            "ideal_response": "Ideal Response",
            "model_response": "Model Response",
        }
        # Filter df columns to only those we want to display and exist
        cols_to_show = [col for col in display_cols.keys() if col in filtered_df.columns]
        st.dataframe(filtered_df[cols_to_show].rename(columns=display_cols))


# Correctly align these elif blocks with the main 'if st.session_state.results_df is not None:'
elif st.session_state.evaluation_results_paths:
    # This case handles if loading failed after paths were found
    st.error("Failed to load or process evaluation results.")
elif not st.session_state.get('evaluation_running', False):
    # Displayed if no results paths exist and not currently running
    st.info("Run an evaluation to see results.")