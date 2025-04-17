## Backend Architecture

This document outlines the backend architecture of the CogniBench project, focusing on the components responsible for orchestrating and executing Large Language Model (LLM) evaluations. The backend is primarily composed of three main parts: a core evaluation library, a set of operational scripts, and an optional API layer.

```mermaid
 graph TD
     subgraph UserInteraction [User Interaction]
         direction LR
         CLI[CLI User] -- runs --> ScriptRunner{"Scripts (run_*, prepare_*, retrieve_*)"}
         APIUser[API User] -- calls --> API(api/main.py)
         StreamlitUser[Streamlit User] -- interacts --> StreamlitApp(cognibench_agent/app.py)
     end

     subgraph ProcessingPipeline [Processing Pipeline]
         direction TB

         subgraph DataInput
             RawRLHF[(Raw RLHF JSON)] -- input --> Ingest(scripts/ingest_rlhf_data.py)
             IngestedData[(Ingested JSON Format)]
             Ingest -- produces --> IngestedData
             StructuredData[(Structured Output JSON)]
             APIInputData[(API Input Data)]
         end

         subgraph CoreLogic ["Core Library (core/)"]
             Config(config.py / config.yaml)
             Workflow(workflow.py)
             Runner(evaluation_runner.py)
             LLMClient(llm_clients/*)
             BatchProcessor(batch_processor.py)
             Parser(response_parser.py)
             Postprocessor(postprocessing.py)
             Writer(output_writer.py)
             Prompts(prompt_templates.py / prompts/*)
         end

         subgraph ScriptsLayer ["Scripts (scripts/)"]
             BatchScript(run_batch_evaluation.py)
             SingleScript(run_single_evaluation.py)
             PrepareJudgingScript(prepare_judging_batch.py)
             RetrieveResultsScript(retrieve_batch_results.py)
             OtherScripts(...)
         end

         %% Connections
         ScriptRunner -- uses --> CoreLogic
         ScriptRunner -- reads/writes --> DataStorage
         API -- uses --> Runner
         API -- uses --> Workflow
         StreamlitApp -- uses --> Runner

         BatchScript -- triggers --> Ingest
         BatchScript -- uses --> Runner
         SingleScript -- uses --> Runner %% Assumed
         PrepareJudgingScript -- reads --> StructuredData
         PrepareJudgingScript -- uses --> BatchProcessor
         PrepareJudgingScript -- uses --> LLMClient
         RetrieveResultsScript -- uses --> BatchProcessor
         RetrieveResultsScript -- uses --> LLMClient
         RetrieveResultsScript -- reads --> IntermediateDataMap
         RetrieveResultsScript -- writes --> StructuredData
         RetrieveResultsScript -- uses --> Workflow %% For post-processing
         RetrieveResultsScript -- uses --> Writer

         Runner -- reads --> IngestedData
         Runner -- calls --> Workflow
         Runner -- uses --> BatchProcessor %% For batch mode submission
         Runner -- writes --> IntermediateDataMap

         Workflow -- uses --> Config
         Workflow -- uses --> LLMClient
         Workflow -- uses --> Prompts
         Workflow -- uses --> Parser
         Workflow -- uses --> Postprocessor
         Workflow -- uses --> Writer
         Workflow -- uses --> BatchProcessor %% For formatting requests

         BatchProcessor -- uses --> LLMClient

         LLMClient -- interacts --> ExternalLLM["External LLM API (OpenAI Sync/Batch)"]

         Writer -- writes --> ResultsJSONL[(_evaluations.jsonl)]
         Runner -- aggregates --> ResultsJSON[(_final_results.json)] %% In sync mode
         RetrieveResultsScript -- writes --> ResultsJSONL %% In batch mode (via Writer)
     end

     subgraph DataStorage ["Data Storage (data/)"]
         direction TB
         BatchOutputDir("Batch-XXX_YYYYMMDD_HHMM/")
         IntermediateDataDir("batch_intermediate_data/")
         BatchOutputDir --> IngestedDataFile("..._ingested_...json")
         BatchOutputDir --> EvalJSONL("..._evaluations.jsonl")
         BatchOutputDir --> FinalResultsJSON("..._final_results.json")
         IntermediateDataDir --> IntermediateDataMap("intermediate_data_...json")
         BatchOutputDir --> StructuredOutputFile("structured_output_...json") %% Output of retrieve --stage structuring
     end

     UserInteraction --> ProcessingPipeline
     ProcessingPipeline --> DataStorage

     %% Styling
     style CoreLogic fill:#c5cae9, stroke:#333
     style ScriptsLayer fill:#ffe0b2, stroke:#333
     style DataInput fill:#b2dfdb, stroke:#333
     style DataStorage fill:#d1c4e9, stroke:#333
 ```

### Core Library (`CogniBench/core/`)

The `core/` directory contains the central logic for the evaluation process.

*   **`workflow.py`**: Defines `run_evaluation_workflow` (executes evaluation steps for a single task instance, handling synchronous LLM calls or preparing requests for batch mode), `process_judging_output` (parses judge response and runs post-processing), and `finalize_and_save_evaluation` (calls output writer).
*   **`evaluation_runner.py`**: Provides higher-level functions (`run_single_task_evaluation_core`, `run_batch_evaluation_core`) to manage evaluation runs, handling task iteration, caching, cancellation, and conditional logic for synchronous vs. batch mode (initiating batch submission).
*   **`batch_processor.py`**: (New) Contains functions specifically for interacting with the OpenAI Batch API (formatting JSONL, uploading files, creating jobs, checking status, downloading/parsing results).
*   **`llm_clients/`**: Contains the interface for interacting with LLMs (`base.py`, `openai_client.py`). `openai_client.py` now includes helpers for Batch API file uploads, batch creation, status retrieval, and result downloading.
*   **`config.py`**: Defines Pydantic models (`AppConfig`, including new `BatchSettings`) for loading and validating `config.yaml`.
*   **`response_parser.py`**: Contains `parse_judge_response` for extracting structured data from the judge LLM's raw output.
*   **`preprocessing.py`**: Includes utility functions like `normalize_text_formats`.
*   **`postprocessing.py`**: Implements `perform_postprocessing` for final answer verification, score aggregation, and review flagging.
*   **`output_writer.py`**: Handles saving evaluation results (`save_evaluation_result`) to JSONL files. Verified to handle potentially missing metrics from batch context.
*   **`prompt_templates.py`**: Provides `load_prompt_template`.
*   **Other Files**: `constants.py`, `log_setup.py`, `schemas/`.

#### Key Function Contracts (Core)

*   **`run_evaluation_workflow` (`workflow.py`)**
    *   **Purpose:** Executes the evaluation steps for a single task instance. If `aggregate_structuring` is `True`, prepares and returns request dictionaries for batch processing. If `False` (synchronous mode), performs structuring LLM calls, judging LLM call, calls `process_judging_output`, and `finalize_and_save_evaluation`. Handles retries for synchronous LLM calls.
    *   **Inputs:** `prompt` (str), `response` (Any), `ideal_response` (Any), `correct_answer` (str), `config` (Dict), `task_id` (Optional[str]), `model_id` (Optional[str]), `llm_client` (Optional[BaseLLMClient]), `output_jsonl_path` (Optional[Path]), `structured_ideal_cache` (Optional[Dict[str, Any]]), `aggregate_structuring` (bool).
    *   **Outputs:** (Dict[str, Any]) - In synchronous mode, dictionary indicating success/error status and evaluation ID. In aggregation mode, dictionary containing request dictionaries (`model_request`, `ideal_request`).

*   **`run_single_task_evaluation_core` (`evaluation_runner.py`)**
    *   **Purpose:** Processes a single task dictionary containing potentially multiple model responses. Calls `run_evaluation_workflow` for each model response, passing the `aggregate_structuring` flag based on the overall mode. Manages ideal response cache and cancellation.
    *   **Inputs:** `task_index` (int), `task_data` (Dict[str, Any]), `config` (AppConfig), `output_jsonl_path` (Optional[Path]), `structured_ideal_cache` (Optional[Dict]), `stop_event` (Optional[threading.Event]), `aggregate_structuring` (bool).
    *   **Outputs:** (Tuple[List[Dict], bool]) - A list of results from `run_evaluation_workflow` calls (either evaluation statuses or request dictionaries) and a boolean indicating overall task success/failure.

*   **`run_batch_evaluation_core` (`evaluation_runner.py`)**
    *   **Purpose:** Orchestrates evaluation for multiple tasks across input files. Checks `config.batch_settings.enabled`.
       *   **If Batch Mode Enabled:** Iterates through tasks calling `run_single_task_evaluation_core` with `aggregate_structuring=True`. Collects structuring requests, uses `batch_processor` to submit the structuring batch job, saves the intermediate data map, logs the batch ID, and returns.
       *   **If Batch Mode Disabled (Synchronous):** Iterates through tasks calling `run_single_task_evaluation_core` with `aggregate_structuring=False`. Processes results immediately, manages output files (`_evaluations.jsonl`, `_final_results.json`), aggregates final results, and returns paths to final reports.
    *   **Inputs:** `config` (AppConfig), `output_dir` (Path), `stop_event` (Optional[threading.Event]). *(Removed `use_structured` as structuring is now mandatory)*.
    *   **Outputs:** (Optional[List[str]]) - Synchronous mode: List of paths to `_final_results.json` files. Batch mode: `None` (as processing is asynchronous).

*   **`parse_judge_response` (`response_parser.py`)**
    *   **Purpose:** Parses the raw string output from the "judge" LLM. Extracts a JSON block (handling markdown fences), parses it (with `json5` fallback), validates its structure (requires `evaluation` key), and validates the content within `evaluation` against expected criteria and allowed scores from the config.
    *   **Inputs:** `raw_response_content` (str), `expected_criteria` (List[str]), `allowed_scores` (List[str]).
    *   **Outputs:** (Dict[str, Any]) - On success, the original parsed dictionary but with the `evaluation` key replaced by validated content. On failure, `{'error': str}`.

*   **`perform_postprocessing` (`postprocessing.py`)**
    *   **Purpose:** Orchestrates post-evaluation steps. Extracts the model's final answer from its structured response object, verifies it against the correct answer (using SymPy if available), aggregates rubric scores into an overall score (Pass/Fail/Partial) based on config rules, and flags evaluations for human review based on verification results, aggregation issues, or parsing errors.
    *   **Inputs:** `parsed_judge_response` (Dict), `structured_model_response_obj` (Optional[Dict]), `correct_final_answer` (Optional[str]), `config` (AppConfig).
    *   **Outputs:** (Dict[str, Any]) - Dictionary containing `final_answer_verified` (Optional[bool]), `verification_message` (str), `aggregated_score` (Literal["Pass", "Fail", "Partial"]), `needs_human_review` (bool), `review_reasons` (List[str]).

### API Layer (`CogniBench/api/`)

*   **`main.py`**: FastAPI application defining endpoints:
    *   `/health`: Basic health check.
    *   `/submit_evaluation`: Accepts a single evaluation task (prompt, model response, ideal response, etc.) via POST request, validates it, and queues it for processing by `run_single_task_evaluation_core`. Returns a submission status and evaluation ID.
    *   `/get_evaluation_result/{evaluation_id}`: Retrieves the detailed results of a previously submitted evaluation using its ID via GET request.
*   **`schemas.py`**: Pydantic models defining the structure for API requests (`EvaluationRequest`) and responses (`EvaluationResponse`, `EvaluationResultData`).

### Scripts (`CogniBench/scripts/`)

*   **`run_batch_evaluation.py`**: CLI for end-to-end processing. If batch mode is disabled in config, runs the full synchronous evaluation via `run_batch_evaluation_core`. If batch mode is enabled, triggers ingestion (optional) and then calls `run_batch_evaluation_core` which will *only submit the structuring batch* and save intermediate data.
*   **`prepare_judging_batch.py`**: (New) CLI script used in batch mode. Takes structured output (from `retrieve_batch_results.py --stage structuring`), generates judging requests, and uses `batch_processor` to submit the judging batch job. Logs the judging batch ID.
*   **`retrieve_batch_results.py`**: (New) CLI script used in batch mode. Takes a `--batch-id` and `--stage` (`structuring` or `judging`). Polls for batch completion using `batch_processor`. Downloads and parses results. Loads intermediate data map. If stage is `structuring`, saves combined structured output. If stage is `judging`, calls `workflow.py` functions for post-processing and saves final results via `output_writer.py`.
*   **`run_single_evaluation.py`**: CLI for running synchronous evaluation on a single pre-ingested JSON file.
*   **`ingest_rlhf_data.py`**: Preprocesses raw RLHF JSON data into the ingested format.
*   **`run_structuring.py`**: Runs *only* the synchronous structuring step. (May need update if batch structuring is desired standalone).
*   **`review_flagged_evals.py`**: Utility for reviewing flagged evaluations.
*   **`show_evaluation_data.py`**: Utility for displaying evaluation results.

#### Key Function Contracts (Scripts)

*   **`ingest_rlhf_data` (`ingest_rlhf_data.py`)**
    *   **Purpose:** Loads raw RLHF JSON data from input files. Extracts relevant fields (task ID, prompt, ideal response, model responses, metadata, human evals, final answer). Performs transformations (e.g., boolean conversion for Yes/No scores) and standardizes the structure. Saves the combined, processed data into a single timestamped JSON file suitable for evaluation runners.
    *   **Inputs:** `input_paths` (List[Path]), `output_path` (Path).
    *   **Outputs:** (None) - Writes the ingested data to `output_path` and prints the path to stdout.

### Configuration (`config.yaml` & `core/config.py`)

*   **`config.yaml`**: Central configuration file defining settings for:
    *   LLM clients (API keys, default models).
    *   Input/Output paths.
    *   Evaluation settings (judge model, prompt template path, expected criteria, allowed scores).
    *   Structuring settings (structuring model, prompt template path).
    *   Aggregation rules.
    *   Consistency checks.
    *   **`batch_settings`**: (New) Contains `enabled` flag, `poll_interval_seconds`, `max_poll_attempts`, `intermediate_data_dir`.
*   **`core/config.py`**: Defines Pydantic models (e.g., `AppConfig`, `LLMClientConfig`, `EvaluationSettings`, `BatchSettings`) used to load, validate, and access settings from `config.yaml`.

### Data Structures and Examples

This section illustrates the structure of key data objects used throughout the pipeline.

#### 1. Raw Input JSON (Example for `ingest_rlhf_data.py`)

*Source: `CogniBench/tests/scripts/test_data/valid_input.json`*

```json
{
  "rlhf": [
    {
      "taskId": 101,
      "messages": [
        {
          "role": "user",
          "text": "User prompt 1",
          "prompt_evaluation": [
            {"question": "Subject", "human_input_value": "Algebra"},
            {"question": "Complexity", "human_input_value": "Basic"}
          ]
        },
        {
          "role": "assistant",
          "response_options": [
            {"model_id": "model_a", "text": "Response A"}
          ],
          "signal": {
            "ideal_response": "Ideal 1",
            "human_evals": [
              {
                "model_id": "model_a",
                "evaluation_form": [
                  {"question": "Model Failure", "human_input_value": "No"},
                  {"question": "Failure Comments", "human_input_value": "Looks good."}
                ]
              }
            ],
            "raw_preference_evaluation_form": [ // Example: May contain final answer
              {"question": "Final Answer", "human_input_value": "\\(60\\)"}
            ]
          }
        }
      ]
    }
  ]
}
```

#### 2. Ingested JSON (Example Output of `ingest_rlhf_data.py`, Input to `run_batch_evaluation_core`)

*Source: `Batch-001_20250412_0427/Batch-001_ingested_20250412_0427.json` (showing one task)*

```json
[
  {
    "task_id": 5500, // Standardized ID (could be from taskId, id, or task_id)
    "prompt": "Find the next number of the sequence 9,5,6,10.5,23.",
    "ideal_response": "To solve this step by step...", // Full ideal response text
    "final_answer": "\\(60\\)", // Extracted ground truth answer
    "model_responses": [ // List of model responses for this task
      {
        "model_id": "o1",
        "response_text": "A succinct way to see why the next term is 54..." // Full response text
      },
      // ... other model responses for the same task
    ],
    "human_evaluations": [ // Transformed human evals (optional)
      {
        "model_id": "o1",
        "model_failure": true, // Boolean conversion
        "failure_comments": "The model had wrongly assumed..."
      },
      // ... other human evaluations
    ],
    "metadata": { // Extracted metadata
      "subject": "Algebra",
      "complexity": "Intermediate",
      "system_prompt": "For exact-match questions..." // Full system prompt text
    }
  }
  // ... other ingested tasks
]
```

#### 3. Raw Judge LLM Output (Example Input to `parse_judge_response`)

*Based on test cases in `response_parser.py`*

```
Some introductory text from the LLM.
```json
{
    "evaluation": {
        "Problem Understanding": {
            "score": "Yes",
            "justification": "Model correctly identified the integral.",
            "confidence": 0.9
        },
        "Results Formulae": {
            "Score": "no",
            "Justification": "Final answer was incorrect."
        },
        "Assumptions": {
            "score": "Partial",
            "justification": "Some assumptions were missed."
        }
    },
    "overall_comment": "Good start, but calculation error."
}
```

Some concluding text.

#### 4. Single Task Evaluation Result (Example Line in `_evaluations.jsonl`)

*Source: `Batch-001_20250412_0427/Batch-001_evaluations.jsonl` (Updated Structure)*

```json
{"evaluation_id": "eval_796c7caa-f92a-4ea0-9ef4-e2482362d162", "task_id": 5500, "model_id": "o1", "response_id": null, "ideal_response_id": null, "judge_llm_model": "gpt-4o", "judge_prompt_template_path": "prompts/judging/Math-L1-Judging-v1.0.txt", "raw_judge_output": {"raw_content": "```json\n{\n  \"evaluation\": {\n    \"Problem Understanding\": {\n      \"score\": \"Yes\",\n      \"justification\": \"...\"\n    },\n    ...\n  }\n}\n```"}, "parsed_rubric_scores": {"Rigor and Completeness": {"score": "Yes", "justification": "..."}, "Logical Implications": {"score": "Yes", "justification": "..."}}, "aggregated_score": "Fail", "final_answer_verified": null, "verification_message": "Verification skipped: Extracted answer was None.", "needs_human_review": false, "review_reasons": [], "parsing_error": null, "human_review_status": "Not Required", "human_reviewer_id": null, "human_review_timestamp": null, "human_corrected_scores": null, "human_review_comments": null, "structured_model_response": {"model": "gpt-4o", "response": {"assumptions": "...", "steps": ["..."], "final_answer": "...", ...}}, "structured_ideal_response": {"model": "gpt-4o", "response": {"assumptions": "...", "steps": ["..."], "final_answer": "...", ...}}, "structuring_api_calls": 2, "judging_api_calls": 1, "total_time_seconds": 15.78, "created_at": "2025-04-11T22:57:58.056959Z"}
```

#### 5. Aggregated Final Results (Example Structure of `_final_results.json`)

*Source: `CogniBench/data/Batch-003_Batch-004_20250416_1813/Batch-003_Batch-004_final_results.json` (Updated Structure)*

```json
{
  "summary": {
    "batch_id": "Batch-003_Batch-004", // Cleaned input file stem(s)
    "total_tasks_processed": 199,
    "total_evaluations_processed": 983,
    "total_structuring_api_calls": 1182, // Example value
    "total_judging_api_calls": 983, // Example value
    "total_evaluation_time_seconds": 15450.67, // Example value
    "average_time_per_task_seconds": 77.64, // Example value
    "average_time_per_evaluation_seconds": 15.72, // Example value
    "average_time_per_model_seconds": { // Example values
      "claude-3-5-sonnet-latest": 18.21,
      "deepseek-v3": 14.55,
      "gpt-4o": 16.03,
      "o1": 15.98,
      "o1-mini": 13.83
    },
    "models_evaluated": [
      "claude-3-5-sonnet-latest",
      "deepseek-v3",
      "gpt-4o",
      "o1",
      "o1-mini"
    ]
  },
  "results": [
    {
      "task_id": 5555, // Sourced from ingested data
      "prompt": "Let \\(A\\) be a subset of \\(\\mathbb{R}^3\\) ...", // Sourced from ingested data
      "ideal_response": "\\nWe need to construct a subset \\( A \\) ...", // Sourced from ingested data
      "final_answer": "\\(Uncountable\\)", // Sourced from ingested data (ground truth)
      "metadata": { // Sourced from ingested data
        "subject": "Linear Algebra",
        "complexity": "Intermediate",
        "turing_task_url": "https://rlhf-v3.turing.com/prompt/..."
      },
      "structured_ideal_response": { // Sourced from JSONL (first occurrence for task)
        "model": "gpt-4o",
        "response": {
          "assumptions": "",
          "steps": [
            "Step 1: Construct the subset A = {(1, t, t^2) | t ∈ ℝ} of ℝ^3.",
            // ... other steps ...
          ],
          "intermediate_results": [
            "Matrix determinant: det = (t2 - t1)(t3 - t1)(t3 - t2)"
          ],
          "final_answer": "The maximum possible Cardinality of A ... is uncountable.",
          "format_notes": "Answer is boxed in LaTeX"
        }
      },
      "evaluations": [ // List of evaluations for this task_id
        {
          "model_id": "o1", // Sourced from JSONL
          "model_response": "First, let us restate the problem carefully:...", // Raw text used for evaluation (from structured object)
          "structured_model_response": { // Sourced from JSONL
             "model": "gpt-4o",
             "response": {
               "assumptions": "A is a subset of ℝ³ ...",
               "steps": [
                 "Step 1: Reformulate the condition ...",
                 // ... other steps ...
               ],
               "intermediate_results": [
                 "No three points of A can lie in the same 2-dimensional subspace ...",
                 // ... other results ...
               ],
               "final_answer": "The maximum possible cardinality of A is infinite.",
               "format_notes": ""
             }
          },
          "human_evaluation": { // Sourced from ingested data (if available)
            "model_failure": false,
            "failure_comments": "The answer is correct."
          },
          "judge_evaluation": { // Detailed judge results from JSONL
            "judge_llm_model": "gpt-4o",
            "judge_prompt_template_path": "prompts/judging/Math-L1-Judging-v1.0.txt",
            "parsed_rubric_scores": {
              "results_formulae": { "score": "Partial", "justification": "..." },
              "rigor_and_completeness": { "score": "Partial", "justification": "..." },
              "problem_understanding": { "score": "Yes", "justification": "..." },
              "assumptions": { "score": "Yes", "justification": "..." }
              // ... other criteria ...
            },
            "aggregated_score": "Partial", // Example score
            "final_answer_verified": true, // Example verification
            "verification_message": "Final answer matches.", // Example message
            "needs_human_review": false,
            // ... other judge fields from JSONL ...
            "created_at": "..." // Timestamp
          }
        },
        // ... evaluation object for other models (o1-mini, gpt-4o, etc.) for task_id 5555
      ]
    }
    // ... other task objects in the 'results' array
  ]
}
```

#### 6. API Request/Response Examples

*   **`/submit_evaluation` Request Body:**

    ```json
    {
      "prompt_id": "math_seq_001",
      "prompt_content": "Find the next number of the sequence 9, 5, 6, 10.5, 23.",
      "prompt_metadata": { "subject": "Algebra", "complexity": "Intermediate" },
      "model_response_id": "resp_o1_task5500_run1",
      "model_name": "o1",
      "model_response_text": "A succinct way to see why the next term is 54...",
      "ideal_response_id": "ideal_math_seq_001",
      "ideal_response_text": "To solve this step by step...",
      "correct_answer": "\\(60\\)",
      "judge_llm_model": "o1", // Optional override
      "judge_prompt_version": "Math-L1-v1.0-robust.txt" // Optional override
    }
    ```

*   **`/submit_evaluation` Success Response Body:**

    ```json
    {
      "status": "submitted",
      "message": "Evaluation task accepted and queued.",
      "evaluation_id": "eval_796c7caa-f92a-4ea0-9ef4-e2482362d162"
    }
    ```

*   **`/get_evaluation_result/{evaluation_id}` Response Body:** (Structure mirrors the JSONL line)

    ```json
    {
      "evaluation_id": "eval_796c7caa-f92a-4ea0-9ef4-e2482362d162",
      "response_id": "resp_o1_task5500_run1",
      "ideal_response_id": "ideal_math_seq_001",
      "judge_llm_model": "o1",
      "judge_prompt_template_version": "prompts/Math-L1-v1.0-robust.txt",
      "raw_judge_output": { "raw_content": "```json...```" },
      "parsed_rubric_scores": { "...": { "score": "...", "justification": "..." } },
      "aggregated_score": "Fail",
      "final_answer_verified": null,
      "verification_message": "Verification skipped: Extracted answer was None.",
      "needs_human_review": false,
      "review_reasons": [],
      "parsing_error": null,
      "human_review_status": "Not Required",
      // ... other fields ...
      "created_at": "2025-04-11T22:57:58.056959Z"
    }
    ```

### Batch Output Folder and File Structure (Synchronous Mode)

When `scripts/run_batch_evaluation.py` is executed with **batch mode disabled**, it creates a dedicated, timestamped output directory within the main `CogniBench/data/` directory (e.g., `Batch-XXX_YYYYMMDD_HHMM/`).

Within this directory, the following key files are saved, using the **cleaned stem** of the input file(s) for the filenames:

```
CogniBench/
└── data/
    └── Batch-XXX_YYYYMMDD_HHMM/  <-- Timestamped Output Directory (Sync Mode)
        ├── Batch-XXX_ingested_YYYYMMDD_HHMM.json  (Optional: Preprocessed input)
        ├── Batch-XXX_evaluations.jsonl            (Detailed results per evaluation)
        ├── Batch-XXX_evaluations_formatted.json   (Optional: Prettified JSONL)
        └── Batch-XXX_final_results.json           (Final aggregated results)
```
*(Descriptions as before)*

### Batch Output Folder and File Structure (Batch API Mode)

When running in **Batch API mode**, multiple scripts interact with the filesystem.
1.  `run_batch_evaluation.py` creates the timestamped output directory (`Batch-XXX_YYYYMMDD_HHMM/`) and saves the optional ingested file there. It also creates `batch_intermediate_data/` (location configurable) and saves `intermediate_data_<structuring_batch_id>.json` inside it.
2.  `retrieve_batch_results.py --stage structuring` downloads the raw batch results (temporary file, deleted after parsing) and saves the processed structured output to a user-specified path (e.g., `structured_output.json`).
3.  `prepare_judging_batch.py` reads the structured output file.
4.  `retrieve_batch_results.py --stage judging` downloads the raw judging batch results (temporary file, deleted after parsing), performs post-processing, and saves the final results via `save_evaluation_result` which writes to a JSONL file specified by the user (e.g., `final_results.jsonl`). *Note: Aggregation into a single `_final_results.json` like in sync mode is not currently implemented in the batch retrieve script.*

```
CogniBench/
├── data/
│   └── Batch-XXX_YYYYMMDD_HHMM/  <-- Timestamped Output Directory
│       └── Batch-XXX_ingested_YYYYMMDD_HHMM.json (Optional)
├── batch_intermediate_data/       <-- Intermediate Data Directory (Configurable)
│   └── intermediate_data_<structuring_batch_id>.json
├── structured_output.json         <-- Output of retrieve --stage structuring (User specified path)
└── final_results.jsonl            <-- Output of retrieve --stage judging (User specified path)
```

### Data Flow Summary

*   **Synchronous Evaluation (CLI/UI):**
    1.  Raw Input JSON -> `ingest_rlhf_data.py` (optional) -> Ingested JSON Format
    2.  Ingested JSON -> `run_batch_evaluation_core` -> (Loop) `run_single_task_evaluation_core` -> (Loop) `run_evaluation_workflow` (Sync LLM calls, processing) -> Append to `_evaluations.jsonl`
    3.  `run_batch_evaluation_core` reads `_evaluations.jsonl`, aggregates -> `_final_results.json`.
*   **Batch API Evaluation (CLI - Manual Steps):**
    1.  Raw Input JSON -> `ingest_rlhf_data.py` (optional) -> Ingested JSON Format
    2.  Ingested JSON -> `run_batch_evaluation.py` -> Generate Structuring Requests -> Submit Structuring Batch -> Save `intermediate_data_map.json`.
    3.  **(Wait)**
    4.  `retrieve_batch_results.py --stage structuring` -> Poll Status -> Download/Parse Results -> Load `intermediate_data_map.json` -> Save `structured_output.json`.
    5.  `structured_output.json` -> `prepare_judging_batch.py` -> Generate Judging Requests -> Submit Judging Batch.
    6.  **(Wait)**
    7.  `retrieve_batch_results.py --stage judging` -> Poll Status -> Download/Parse Results -> Load `intermediate_data_map.json` -> Post-process -> Save final results line-by-line to `final_results.jsonl`.
*   **Single Evaluation (API):** (Remains Synchronous) API Call -> `run_single_task_evaluation_core` -> `run_evaluation_workflow` -> Append to `_evaluations.jsonl` -> API Response -> GET retrieves data.

### Key Technologies

*   Python, FastAPI, Uvicorn, OpenAI lib, PyYAML, Pydantic, python-dotenv, TQDM, SymPy, ANTLR, Pytest, Ruff.
