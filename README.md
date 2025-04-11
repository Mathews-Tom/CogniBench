# CogniBench ⚖️

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
* **Modular Workflow:** Clearly defined steps for preprocessing, LLM invocation, response parsing, evaluation, and postprocessing.
* **Configurable Evaluation:**
  * Judge LLM (provider, model), prompt template, expected rubric criteria, and allowed scores are defined in `config.yaml`.
  * Support structure for multiple LLM providers (OpenAI, Anthropic, Google placeholders).
  * Configuration validation on script startup.
* **Robust Answer Verification:**
  * Enhanced answer extraction patterns (including LaTeX `$$...$$`).
  * Mathematical equivalence checking using `sympy` (if installed) for more accurate verification of math/symbolic answers, with fallback to string comparison.
* **Improved Error Handling:** Response parser reports all validation errors found, not just the first.
* **Data Management:** Structured way to handle prompts, ideal responses, model responses, and evaluation results. Output files are organized into timestamped subdirectories for each batch run.
* **Batch Processing:** Includes scripts for ingesting raw data and running evaluations on entire batches.
* **Combined Results:** Generates a final JSON file grouping results by task for easier comparison across models.
* **Configurable Logging:** Timestamped log files and configurable console output levels.
* **API Interface:** (Optional) Provides an API for programmatic interaction (loads config on startup).
* **Streamlit UI:** A user-friendly interface (`streamlit_app/`) for uploading batch files, configuring the judge (provider, model, template, API key), viewing the configuration summary, running evaluations (with progress bar and live log output), viewing persistent logs, and visualizing results (overall performance, rubric breakdown per criterion/model, human review status counts, and explorers for all tasks and those needing review).

## Project Structure

```
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
│       └── Batch-XXX_final_results.json        # Combined ingested data + evaluations, grouped by task
├── logs/                 # Log files
│   └── CogniBench_YYYYMMDD_HHMM.log # Timestamped log file for each run
├── prompts/              # Raw prompt files (e.g., judge prompt templates)
├── scripts/              # Utility and execution scripts
│   ├── ingest_rlhf_data.py       # Script to convert raw data to CogniBench format
│   └── run_batch_evaluation.py   # Script to run ingestion and evaluation for a batch
├── streamlit_app/        # Streamlit application for UI-based interaction
├── tests/                # Unit and integration tests
├── .gitignore
├── .python-version       # Specifies Python version (likely for pyenv)
├── Dockerfile            # Containerization configuration
├── LICENSE               # Project license information
├── overview.md           # High-level overview document
├── pyproject.toml        # Project metadata and dependencies (PEP 621)
├── README.md             # This file
├── roadmap.md            # Project roadmap and future plans
├── run_single_evaluation.py # Script to execute evaluation on pre-ingested data
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

**Running a Single Evaluation (on pre-ingested data):**

This script processes a JSON file already in the CogniBench ingested format (like the output of `ingest_rlhf_data.py`).

```bash
python3 run_single_evaluation.py --input-data <path_to_ingested_data.json> --config <path_to_config.yaml> --output-jsonl <path_to_output.jsonl>
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
3. Run the evaluation script (`run_single_evaluation.py`) using the ingested data, saving the detailed results (`_evaluations.jsonl`) inside the subdirectory.
4. Create a formatted JSON version (`_evaluations_formatted.json`) inside the subdirectory.
5. Create a final combined results file (`_final_results.json`) inside the subdirectory, grouping results by task ID.
6. Log detailed output to a timestamped file in `logs/`.
7. Display a `tqdm` progress bar on the console during the evaluation step.

## Contributing

Please refer to `CONTRIBUTING.md` (if available) for guidelines on how to contribute to CogniBench.

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
