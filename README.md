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
* **Modular Workflow:** Clearly defined steps for preprocessing, response generation, evaluation, and postprocessing.
* **Extensible LLM Clients:** Easily integrate different LLM APIs (e.g., OpenAI).
* **Data Management:** Structured way to handle prompts, ideal responses, model responses, and evaluation results.
* **API Interface:** (Optional) Provides an API for programmatic interaction.

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
│   ├── prompt_templates.py # Templates for evaluation prompts
│   ├── response_parser.py # Parses LLM responses
│   └── workflow.py       # Main evaluation workflow orchestration
├── data/                 # Data files (prompts, responses, evaluations)
├── prompts/              # Raw prompt files
├── scripts/              # Utility scripts for data analysis, review, etc.
├── tests/                # Unit and integration tests
├── .gitignore
├── .python-version       # Specifies Python version (likely for pyenv)
├── Dockerfile            # Containerization configuration
├── LICENSE               # Project license information
├── overview.md           # High-level overview document
├── pyproject.toml        # Project metadata and dependencies (PEP 621)
├── README.md             # This file
├── roadmap.md            # Project roadmap and future plans
├── run_single_evaluation.py # Script to execute a single evaluation run
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

## Usage

To run an evaluation:

```bash
python run_single_evaluation.py --config <path_to_config_file>
```

*(Note: The exact command and configuration details might vary. Refer to `run_single_evaluation.py` or other documentation for specifics.)*

## Contributing

Please refer to `CONTRIBUTING.md` (if available) for guidelines on how to contribute to CogniBench.

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
