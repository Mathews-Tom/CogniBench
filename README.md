# ⚖️ CogniBench 🔍

CogniBench is a framework designed for evaluating Large Language Models (LLMs) using an LLM-as-a-Judge approach, with a specific focus on advanced Mathematics and Science, Technology, and Engineering (STE) domains.

## Overview

Evaluating LLMs on complex reasoning tasks, especially in specialized fields like advanced math and STE, presents unique challenges. CogniBench aims to provide a standardized workflow and tools for:

* Preprocessing evaluation prompts and ideal responses.
* Generating responses from various LLMs.
* Evaluating model responses against ideal answers using another LLM (the "judge").
* Postprocessing and analyzing evaluation results.

## Features

* **LLM-as-a-Judge Evaluation:** Leverages a powerful LLM to assess the quality and correctness of other LLMs' responses.
* **Specialized Domain Focus:** Tailored for evaluating performance on advanced Math & STE problems.
* **Modular Workflow:** Clearly defined steps for preprocessing (primarily text normalization), structuring (LLM-based extraction of key elements including the final answer), judging (LLM-based evaluation), and postprocessing (verification of the LLM-extracted answer, aggregation).
* **Configurable Evaluation:**
  * Judge LLM (provider, model), prompt template, expected rubric criteria, and allowed scores are defined in `config.yaml`. The judging and structuring prompts have been updated to align precisely with these rubric criteria, ensuring consistency and accuracy in evaluations.
  * Support structure for multiple LLM providers (OpenAI, Anthropic, Google placeholders).
  * Configuration validation on script startup.
* **Robust Answer Verification:**
  * Mathematical equivalence checking using `sympy` (if installed) for accurate verification of math/symbolic answers (comparing the `final_answer` extracted by the structuring LLM against the correct answer), with fallback to string comparison.
  * Recommended temperature setting for evaluations is `0.0` to ensure deterministic, consistent, and reproducible outputs.
* **Improved Error Handling:** Response parser reports all validation errors found, not just the first.

* **Recent Enhancements (April 2025):**
  * **Streamlit App Refactor:** Renamed the Streamlit directory from `streamlit_app` to `cognibench_agent` for clarity.
  * **Streamlit Ingestion Fix:** Modified the Streamlit app (`cognibench_agent/app.py`) to correctly perform the data ingestion step (calling `scripts/ingest_rlhf_data.py`) before invoking the core evaluation runner, aligning its behavior with the command-line batch script and resolving runtime errors.
  * **Core Runner Data Handling Fix:** Updated the core evaluation runner (`core/evaluation_runner.py`) to correctly read `prompt`, `ideal_response`, `final_answer`, and `model_responses` directly from the ingested data format, resolving issues where tasks were skipped.
  * **Aggregation Logic Fix:** Corrected a case mismatch (`taskId` vs `task_id`) in the results aggregation logic within the core evaluation runner, ensuring final results are correctly populated.
  * **Streamlit State Fix:** Resolved a `StreamlitAPIException` related to modifying widget state directly within the 'Clear Caches' button logic.
  * **Robust JSON Parsing:** Implemented `safe_json_parse` to gracefully handle empty or malformed JSON inputs.
  * **Improved SymPy Parsing:** Added `safe_sympy_parse` for robust mathematical expression parsing.
  * **Enhanced Logging:** Improved logging clarity and consistency across modules.
  * **Graph Regeneration from Existing Data:** Added functionality in the Streamlit UI to regenerate evaluation graphs from existing `_final_results.json` files.
  * **Mutually Exclusive Actions:** Implemented UI controls to prevent simultaneous evaluation runs and graph regeneration.
  * **Folder Sorting by Modification Time:** Enhanced folder selection in the UI.
* **Data Management:** Structured way to handle prompts, ideal responses, model responses, and evaluation results. Output files are organized into timestamped subdirectories for each batch run.
* **Batch Processing:** Includes scripts for ingesting raw data and running evaluations on entire batches.
* **Combined Results:** Generates a final JSON file (`_final_results.json`) grouping results by task for easier comparison across models. This file now includes the raw `model_response` text alongside structured and judged evaluations.
* **Configurable Logging:** Timestamped log files and configurable console output levels. Includes detailed logs for structuring and judging LLM calls within the core workflow. Logs are now stored in timestamped directories (e.g., `logs/YYYYMMDD_HHMM/`) with separate files for backend (`backend.log`) and Streamlit (`streamlit.log`) operations.
* **API Interface:** (Optional) Provides an API for programmatic interaction (loads config on startup).
* **Streamlit UI:** A user-friendly interface (`cognibench_agent/`) for:
  * Uploading batch files.
  * Configuring Structuring and Judging models (provider, model, template, API key).
  * Viewing configuration summaries and source files (prompts, `config.yaml`).
  * Running evaluations via direct integration with the core logic (`core.evaluation_runner`) in a background thread (improving responsiveness and replacing previous `subprocess` calls).
  * Monitoring progress with live, detailed log output captured from core modules within a collapsible expander.
  * Gracefully stopping ongoing evaluations.
  * Visualizing results with enhanced, interactive charts (clustered bars, model/criteria filters).
  * Exploring detailed results and drilling down into specific task/model evaluations.
  * Loading results from previous runs stored persistently.
  * Clearing LLM and results loading caches.
  * **Output:** Saves results persistently to `CogniBench/data/<BatchFileName>_YYYYMMDD_HHMM/`.

## Project Structure

```plaintext
CogniBench/
├── api/                  # FastAPI application for serving the evaluation workflow
├── core/                 # Core logic for the evaluation workflow
│   ├── llm_clients/      # Clients for interacting with different LLM APIs
│   ├── __init__.py
│   ├── output_writer.py  # Handles writing evaluation results
│   ├── postprocessing.py # Logic for processing results after evaluation
│   ├── preprocessing.py  # Logic for preparing data before evaluation
│   ├── prompt_templates.py # (Legacy - templates now loaded via path in config)
│   ├── response_parser.py # Parses LLM responses
│   └── workflow.py       # Main evaluation workflow orchestration
├── data/                 # Default directory for evaluation outputs
│   └── Batch-XXX_YYYYMMDD_HHMM/ # Timestamped subdirectory for each batch run
│       ├── Batch-XXX_ingested_YYYYMMDD_HHMM.json # Ingested data ready for evaluation
│       ├── Batch-XXX_evaluations.jsonl         # Detailed evaluation results (JSON Lines)
│       ├── Batch-XXX_evaluations_formatted.json # Formatted JSON version of evaluations
│       └── Batch-XXX_final_results.json        # Combined ingested data + evaluations (including raw model_response), grouped by task
├── logs/                 # Log files directory
│   └── YYYYMMDD_HHMM/    # Timestamped directory for each run
│       ├── backend.log   # Logs from core scripts, API, etc.
│       └── streamlit.log # Logs specifically from the Streamlit UI
├── prompts/              # Raw prompt files (e.g., judge prompt templates)
├── scripts/              # Utility and execution scripts
│   ├── ingest_rlhf_data.py       # Script to convert raw data to CogniBench format
│   └── run_batch_evaluation.py   # Script to run ingestion and evaluation for a batch
├── cognibench_agent/     # Streamlit application for UI-based interaction
├── tests/                # Unit and integration tests
├── .gitignore
├── .python-version       # Specifies Python version (likely for pyenv)
├── Dockerfile            # Containerization configuration
├── LICENSE               # Project license information
├── docs/overview.md      # High-level overview document (in docs/)
├── pyproject.toml        # Project metadata and dependencies (PEP 621)
├── README.md             # This file
├── roadmap.md            # Project roadmap and future plans
├── scripts/run_single_evaluation.py # Script to execute evaluation on pre-ingested data
└── uv.lock               # Dependency lock file for uv package manager
```

## Installation

1. **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd CogniBench
    ```

2. **Set up Python environment:**
    * It's recommended to use a virtual environment. If you use `pyenv`, ensure the version specified in `.python-version` is installed.
    * Install `uv` if you haven't already: `pip install uv`
3. **Install dependencies:**

    ```bash
    uv pip install -r requirements.txt # Or potentially 'uv pip sync' if using pyproject.toml directly
    ```

    *(Note: Check `pyproject.toml` for the exact dependency management setup. If `requirements.txt` doesn't exist, you might need `uv pip install .`)*
    * For enhanced mathematical answer verification, install `sympy`: `uv pip install sympy`
    * Configuration loading requires `PyYAML`: `uv pip install pyyaml`

## Usage

**Using the Streamlit UI:**

Provides a graphical interface for running evaluations.

```bash
streamlit run cognibench_agent/app.py
```

1. Launch the app using the command above from the `CogniBench` root directory.
2. Upload one or more raw batch JSON files (e.g., from `Task_JSONs/`).
3. Configure the Structuring and Judging models, API keys (optional), and prompt templates.
4. Click "Run Evaluation". Progress and detailed logs will appear in the "Run Output / Log" expander.
5. Results (plots, tables) will appear upon completion. Intermediate and final files are saved persistently to `CogniBench/data/<BatchFileName>_YYYYMMDD_HHMM/`.
6. Use "Load Existing Results" to select and view results from previous runs saved in the `data` directory.
7. Use "Clear Caches (Results & LLM)" to clear LLM API call caches and Streamlit's data loading cache if needed.

**Running Evaluations via Scripts:**

**Running a Single Evaluation (on pre-ingested data):**

This script processes a JSON file already in the CogniBench ingested format (like the output of `ingest_rlhf_data.py`).

```bash
python3 scripts/run_single_evaluation.py --input-data <path_to_ingested_data.json> --config <path_to_config.yaml> --output-jsonl <path_to_output.jsonl>
```

* `--input-data`: Path to the JSON file containing tasks, prompts, ideal responses, and model responses.
* `--config`: Path to the YAML configuration file (e.g., `config.yaml`).
* `--output-jsonl`: Path where the detailed evaluation results (in JSON Lines format) will be saved.

**Running a Batch Evaluation (End-to-End):**

This script handles the full workflow: ingesting raw data, running evaluations, and generating final combined results.

```bash
python3 scripts/run_batch_evaluation.py <path_to_raw_batch_file.json> --config <path_to_config.yaml>
```

* `<path_to_raw_batch_file.json>`: Path to the input file containing the raw data (e.g., RLHF format).
* `--config`: Path to the YAML configuration file (e.g., `config.yaml`).

This script will:

1. Create a timestamped subdirectory in `data/` (e.g., `data/Batch-XXX_YYYYMMDD_HHMM/`).
2. Run the ingestion script, saving the ingested data JSON inside the subdirectory.
3. Run the evaluation script (`scripts/run_single_evaluation.py`) using the ingested data, saving the detailed results (`_evaluations.jsonl`) inside the subdirectory.
4. Create a formatted JSON version (`_evaluations_formatted.json`) inside the subdirectory.
5. Create a final combined results file (`_final_results.json`) inside the subdirectory, grouping results by task ID.
6. Log detailed output to a timestamped file in `logs/`.
7. Display a `tqdm` progress bar on the console during the evaluation step.

### Running Evaluations with Batch API Mode (Experimental)

If you have enabled the OpenAI Batch API mode in your `config.yaml` (by setting `batch_settings.enabled: true`), the evaluation process becomes asynchronous and requires manual steps to manage the batch jobs submitted to OpenAI.

**Important:** This mode relies on OpenAI's Batch API, which can take significant time (up to 24 hours per batch) to complete. You will need to manually track **Batch IDs** and execute scripts sequentially.

The workflow is as follows:

**Step 1: Submit Structuring Batch**

1. Run the primary batch script. This will *only* submit the initial structuring batch request to OpenAI if Batch API mode is enabled.

```bash
python3 scripts/run_batch_evaluation.py <path_to_raw_batch_file.json> --config <path_to_config.yaml>
```

* This script handles the ingestion of the raw data file first.
* If batch mode is enabled, it then prepares and submits *only* the structuring requests to the OpenAI Batch API.

2. The script will output a **Structuring Batch ID** (e.g., `batch_abc123`). **Save this ID carefully.**
3. **Wait** for OpenAI to process this batch. You can monitor its status via the OpenAI Batch Dashboard or API. This can take several hours or longer.

**Step 2: Retrieve Structuring Results & Submit Judging Batch**

1. Once the structuring batch shows as 'completed' in OpenAI, retrieve its results using the saved ID:

```bash
python3 scripts/retrieve_batch_results.py --batch-id <Structuring_Batch_ID> --stage structuring --output-path <path_for_structured_output.json> --config <path_to_config.yaml>
```

* Replace `<Structuring_Batch_ID>` with the ID from Step 1.
* Specify an `<output-path>` (e.g., `data/<YourBatchName>_YYYYMMDD_HHMM/structured_results.json`) to save the retrieved structuring results. This script downloads the output file associated with the completed batch job from OpenAI.

2. Prepare and submit the judging batch using the structured results:

```bash
python3 scripts/prepare_judging_batch.py --structured-input <path_from_previous_step.json> --config <path_to_config.yaml>
```

* Use the `<output-path>` from the previous command as the `--structured-input`. This script reads the structured results, prepares the judging prompts, and submits the judging requests as a new batch job to the OpenAI Batch API.

3. This script will output a **Judging Batch ID**. **Save this ID carefully.**
4. **Wait** again for OpenAI to process the judging batch.

**Step 3: Retrieve Judging Results & Finalize**

1. Once the judging batch shows as 'completed' in OpenAI, retrieve the final results:

```bash
python3 scripts/retrieve_batch_results.py --batch-id <Judging_Batch_ID> --stage judging --output-path <path_for_final_results.jsonl> --config <path_to_config.yaml>
```

* Replace `<Judging_Batch_ID>` with the ID from Step 2.
* Specify the final `<output-path>` (e.g., `data/<YourBatchName>_YYYYMMDD_HHMM/final_evaluations.jsonl`) where the processed judging results will be saved in JSON Lines format.

2. This script retrieves the judging results, performs final postprocessing (like answer verification using the original ingested data), and saves the complete evaluation data to the specified file using the `save_evaluation_result` function from `output_writer.py`.

This multi-step process gives you control over the asynchronous Batch API workflow but requires careful management of IDs and waiting periods between steps.

## Troubleshooting

If you encounter errors when running evaluation scripts (`run_single_evaluation.py` or `run_batch_evaluation.py`), particularly after a fresh clone or environment setup, follow these steps:

1. **Check Dependencies:** The most common issue after cloning is missing dependencies. Ensure you have installed `uv` and then installed the project requirements from the `CogniBench` directory:

    ```bash
    # Ensure uv is installed (if not already)
    pip install uv
    # Sync dependencies using the lock file
    uv pip sync
    # Install the CogniBench package itself in editable mode
    uv pip install -e .
    ```

    *Initial errors like `ModuleNotFoundError: No module named 'core'` or failures immediately upon running the script often point to missing dependencies or the local package not being installed correctly.*

2. **Verify Python Path for Subprocesses:** The `run_batch_evaluation.py` script calls `run_single_evaluation.py` as a subprocess. Python needs to correctly resolve the project's internal modules (like `core`) within this subprocess. If you encounter `ModuleNotFoundError: No module named 'core'` specifically when running the batch script, it might be due to path issues in the subprocess.

    * **Solution:** Ensure `run_batch_evaluation.py` invokes the single evaluation script using Python's module execution flag (`-m`). The script has been updated to do this, but verify the `evaluation_command` list (around line 357) looks like this:

        ```python
        evaluation_command = [
            sys.executable,
            "-m", # Module execution flag
            "scripts.run_single_evaluation", # Module path
            "--config",
            # ... other args
        ]
        ```

    * *This ensures the subprocess inherits or correctly determines the necessary paths to find modules like `core`.*

3. **Check Log Files:** If errors persist, examine the detailed log files. The logging setup (`core/log_setup.py`) creates timestamped directories within `CogniBench/logs/` (e.g., `logs/YYYYMMDD_HHMM/`). Check the `backend.log` file within the relevant timestamped directory for specific error messages or tracebacks from the workflow.

4. **Use the Correct Workflow:** For standard end-to-end evaluations, always use the `run_batch_evaluation.py` script with your raw input data file (e.g., `../Task_JSONs/single_task.json`). Avoid calling `run_single_evaluation.py` directly unless you are working with already ingested data and understand the required arguments (like `--input-data` and `--output-jsonl`). The batch script handles the creation of necessary directories and intermediate files.

## Contributing

Please refer to `CONTRIBUTING.md` (if available) for guidelines on how to contribute to CogniBench.

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
