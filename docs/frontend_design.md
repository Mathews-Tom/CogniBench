## Frontend/Agent Design

The `CogniBench/cognibench_agent/` directory houses the primary user-facing component of the CogniBench project: a web application built using the Streamlit framework.

**Key Components & Functionality:**

* **Streamlit Application (`app.py`):** This is the core of the frontend. It provides a web-based graphical user interface (GUI) for interacting with the CogniBench evaluation pipeline.
* **User Interaction:** Users interact with the application through their web browser. The interface allows them to:
* Upload raw evaluation task data (JSON files).
* Configure the evaluation run, including selecting structuring and judging models (e.g., different OpenAI models), providing optional API keys, and choosing specific prompt templates (`.txt` files) from the `CogniBench/prompts/` subdirectories.
* Initiate the evaluation process.
* View the progress and logs of the running evaluation in real-time.
* Explore and analyze the results of completed evaluations, including aggregated statistics, performance charts (using Plotly), detailed rubric scores, and filterable data tables (using Pandas).
* **Configuration (`constants.py`, interaction with `core.config`):** While `constants.py` exists, the application primarily relies on constants and configuration handling imported from the `CogniBench/core/` package (`core.constants`, `core.config`). It uses these to populate UI elements like model selection dropdowns and to generate the final configuration (`AppConfig`) passed to the evaluation runner.

**Technology Stack:**

* **UI Framework:** Streamlit (`streamlit`)
* **Data Handling:** Pandas (`pandas`)
* **Visualization:** Plotly (`plotly`)
* **Configuration Parsing (likely indirect):** PyYAML (`pyyaml`) - Used by the core configuration handling.

**Backend Connection:**

* This Streamlit application does **not** interact with the separate FastAPI defined in `CogniBench/api/main.py`.
* Instead, it functions as a direct frontend to the core evaluation logic. It imports necessary modules and functions directly from the `CogniBench/core/` package (e.g., `run_batch_evaluation_core` from `evaluation_runner.py`, `AppConfig` from `config.py`, various constants).

**Key Function Contracts (`app.py`):**

* **`initialize_session_state()`:**
* **Purpose:** Sets up default values for Streamlit's session state (`st.session_state`) if they don't already exist. This includes initializing UI selections (models, templates), flags for UI visibility, tracking uploaded files, evaluation status, results data, and threading objects for background processing.
* **Inputs:** `st.session_state` (implicit).
* **Outputs:** Populates `st.session_state` with default keys and values. Creates a temporary directory for uploaded files.
* **`render_file_uploader()`:**
* **Purpose:** Renders the `st.file_uploader` widget, allowing users to select one or more JSON files. Saves the uploaded files to a temporary directory managed by the session state.
* **Inputs:** User interaction (file selection). Reads/writes `st.session_state.uploaded_files_info`, `st.session_state.temp_dir_path`.
* **Outputs:** Renders the file uploader UI. Updates `st.session_state` with information about the uploaded files (name, temporary path). Triggers `st.rerun()` if files change.
* **`render_config_ui()`:**
* **Purpose:** Renders UI elements (select boxes, text inputs, buttons) for configuring structuring/judging models, API keys, and prompt templates. Uses constants from `core.constants` and helper functions like `get_templates` to populate options. Allows viewing prompt/config content.
* **Inputs:** Reads `st.session_state` for current selections and visibility flags. Uses `AVAILABLE_MODELS`, `AVAILABLE_STRUCTURING_TEMPLATES`, `AVAILABLE_JUDGING_TEMPLATES`.
* **Outputs:** Renders configuration expanders and widgets. Updates `st.session_state` based on user selections and button clicks (e.g., `structuring_model_select`, `show_structuring`). Sets `st.session_state.config_complete` flag.
* **`generate_run_config() -> Optional[AppConfig]`:**
* **Purpose:** Constructs the `AppConfig` object needed to run the core evaluation logic. It reads the base `config.yaml`, merges it with selections made in the UI (models, templates, API keys, input file paths), and generates a unique output directory path within `CogniBench/data/`.
* **Inputs:** Reads `st.session_state` for UI selections and uploaded file paths. Reads `BASE_CONFIG_PATH` (`CogniBench/config.yaml`). Uses `AVAILABLE_MODELS`, `AVAILABLE_*_TEMPLATES`, `DATA_DIR`.
* **Outputs:** Returns an `AppConfig` object if configuration is valid and files are uploaded, otherwise returns `None` and shows an error. Logs the generated output directory path.
* **`start_core_evaluation()`:**
* **Purpose:** Initiates the backend evaluation process in a separate thread to avoid blocking the UI. It first calls `generate_run_config()` to get the configuration. If successful, it clears OpenAI cache, sets status flags in `st.session_state`, creates a `threading.Thread` targeting `evaluation_worker`, and starts the thread.
* **Inputs:** Reads `st.session_state` (config completeness, uploaded files). Calls `generate_run_config()`.
* **Outputs:** Updates `st.session_state` (e.g., `evaluation_running`, `eval_start_time`, `worker_thread`, `stop_event`). Starts the background evaluation thread. Renders status messages via `st.info`/`st.error`.
* **`evaluation_worker(...)`:**
* **Purpose:** The function executed by the background thread. It calls the core `run_batch_evaluation_core` function with the generated `AppConfig`. It uses queues (`output_queue`, `log_queue`) to communicate status, logs, and results back to the main Streamlit thread. Handles exceptions during the core run.
* **Inputs:** `app_config: AppConfig`, `output_queue: queue.Queue`, `log_queue: queue.Queue`, `stop_event: threading.Event`.
* **Outputs:** Puts status messages, logs, and final result paths (`_final_results.json`) into the `output_queue`. Puts log records into the `log_queue`. Updates `st.session_state.evaluation_results_paths` (indirectly via queue).
* **`render_evaluation_progress(...)`:**
* **Purpose:** Displays the real-time progress and logs from the running evaluation. It checks the `output_queue` for messages from the `evaluation_worker` thread and displays them. Also shows elapsed time and provides a "Stop Evaluation" button.
* **Inputs:** Reads `st.session_state` (running status, start time, output queue).
* **Outputs:** Renders the run/stop buttons, progress area (`st.container`), log messages (`st.text_area`), and elapsed time. Calls `start_core_evaluation` or `stop_evaluation` on button clicks.
* **`load_and_process_results(...)`:**
* **Purpose:** Loads evaluation results from selected `_final_results.json` files (identified by their parent directory names). It concatenates data from multiple runs, processes it into a Pandas DataFrame, calculates aggregated summary statistics, and caches the results using `@st.cache_data`.
* **Inputs:** `selected_folder_paths: List[str]` (paths to result directories selected by the user). Reads `_final_results.json` files within these paths.
* **Outputs:** Returns a tuple: `(pd.DataFrame, Dict[str, Any])` containing the combined results DataFrame and the aggregated summary dictionary.
* **`display_*` functions (`display_summary_stats`, `display_performance_plots`, `display_rubric_plots`, `display_results_table`, `display_human_review_tasks`):**
* **Purpose:** These functions take the processed DataFrame and/or summary dictionary from `load_and_process_results` and render various visualizations and tables using Streamlit, Plotly, and Pandas. They include widgets for filtering and exploring the data.
* **Inputs:** `df: pd.DataFrame`, `summary_data: Dict[str, Any]`. Read `st.session_state` for filter selections.
* **Outputs:** Render Streamlit UI elements (headers, tables via `st.dataframe`, plots via `st.plotly_chart`, filters via `st.selectbox`/`st.radio`, etc.).

**Data Structures:**

* **Input JSON Format:**
* The application expects users to upload JSON files that conform to the *ingested* data format produced by scripts like `CogniBench/scripts/ingest_rlhf_data.py`. This format typically represents a list of evaluation tasks, where each task includes fields like `task_id`, `prompt`, `ideal_response`, and potentially metadata.
* The Streamlit app itself does not perform deep validation of the JSON structure; it passes the file path(s) to the `run_batch_evaluation_core` function, which expects the correct ingested format. Refer to the backend documentation or the `ingest_rlhf_data.py` script for the precise schema.
* **Results Data Structure (`_final_results.json` -> Pandas DataFrame):**
* When viewing results, the `load_and_process_results` function reads the `_final_results.json` file(s) generated by evaluation runs.
* Each `_final_results.json` contains a list of dictionaries, where each dictionary represents the complete evaluation data for a single task, including the original input, structured output, judge's scores (overall performance, rubric scores, justifications), model IDs, and metadata.
* This list of dictionaries is loaded directly into a Pandas DataFrame within the Streamlit app (`st.session_state.results_df`).
* Columns in the DataFrame correspond to the keys in the JSON dictionaries (e.g., `task_id`, `model_id`, `prompt`, `ideal_response`, `structured_response`, `judge_prompt`, `judge_response`, `performance_score`, `rubric_scores.Completeness`, `rubric_justifications.Completeness`, etc.).
* This DataFrame is then used as the input for various display functions (`display_performance_plots`, `display_results_table`, etc.) to generate tables and Plotly charts. An aggregated summary dictionary (`st.session_state.aggregated_summary`) is also computed from this DataFrame.

**File/Folder Interaction:**

* **Prompt Templates (`CogniBench/prompts/`):**
* The `get_templates` function scans the `CogniBench/prompts/structuring/` and `CogniBench/prompts/judging/` subdirectories (paths obtained from `core.constants.STRUCTURING_TEMPLATES_DIR` and `core.constants.JUDGING_TEMPLATES_DIR`) for files ending in `.txt`.
* The names (stems) and paths of these `.txt` files are used to populate the "Structuring Prompt Template" and "Judging Prompt Template" dropdowns (`st.selectbox`) in the `render_config_ui` function.
* **Base Configuration (`CogniBench/config.yaml`):**
* The application reads the base configuration settings directly from `CogniBench/config.yaml`. The path to this file is obtained from `core.constants.BASE_CONFIG_PATH`.
* The `generate_run_config` function loads this YAML file as the starting point and then merges the user's selections from the UI on top of it.
* **Output Directory (`CogniBench/data/`):**
* When an evaluation is initiated from the Streamlit app, the `generate_run_config` function dynamically creates a unique output subdirectory within `CogniBench/data/` (path obtained from `core.constants.DATA_DIR`).
* The subdirectory name is generated based on the stem(s) of the input JSON file(s) and a timestamp (e.g., `Batch-001_20250417_0843`).
* This generated path (e.g., `/Users/druk/WorkSpace/Turing/Benchmarking/CogniBench/data/Batch-001_20250417_0843`) is included in the `AppConfig` object passed to `run_batch_evaluation_core`.
* All output files for that specific run (`_evaluations.jsonl`, `_final_results.json`, `_evaluations_formatted.json`, logs, etc.) are saved within this dynamically created subdirectory.

**Conclusion:**

The `cognibench_agent` directory provides a dedicated web interface for managing and executing CogniBench evaluations. It is not a generic agent or API client but rather a tightly integrated Streamlit application that leverages the project's core Python modules directly, offering a user-friendly way to configure runs, view progress, and analyze results.
