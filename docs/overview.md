# ⚖️ CogniBench: LLM-as-a-Judge System Architecture for Advanced Math & STE Evaluation 🔍

## 1. Introduction & Goals

* **Purpose:** To create an automated system (CogniBench) that evaluates the quality of Large Language Model (LLM) responses to advanced mathematics and STEM prompts.
* **Core Function:** The system takes ingested data containing a `PROMPT`, `MODEL RESPONSE`s (including step-by-step reasoning), an `IDEAL RESPONSE` (expert-written detailed steps), and a ground-truth `FINAL_ANSWER` (extracted during ingestion from human annotations). It uses a configurable Judge LLM and prompt template to evaluate each `MODEL RESPONSE` against a predefined rubric (e.g., the L1 rubric), assessing correctness, rigor, and logical soundness. It outputs detailed, structured evaluation results.
* **Scope:** Focused on the evaluation (`Judge`) component, operating on data preprocessed by `ingest_rlhf_data.py`. Implements L1 diagnostic analysis.
* **Target Users:** Internal R&D teams, potentially adapted for client reporting verification.
* **Key Challenges:** Ensuring consistent and accurate application of nuanced rubrics by the Judge LLM, robust parsing of judge outputs, and accurate verification of complex answers (especially mathematical).

## 2. Key Concepts (from Reference Document)

* **PROMPT:** The original advanced math/STEM problem.
* **MODEL RESPONSE:** The LLM-generated solution, including reasoning steps and a final answer.
* **IDEAL RESPONSE:** The expert-provided, step-by-step correct solution methodology.
* **FINAL_ANSWER:** The ground-truth final answer to the prompt, extracted from human annotations during ingestion.
* **CogniBench:** The system being designed; it leverages a powerful LLM (the "Judge LLM") to perform the evaluation.
* **L1 Rubric Parameters:** Problem Understanding, Assumptions, Logical Implications, Results Formulae, Rigor and Completeness.
* **L2 Subcomponents:** Granular criteria under each L1 parameter.
* **Binary Evaluation (Yes/No):** Each L1 parameter (and potentially L2 subcomponent) is scored as either fully meeting the criteria (Yes) or having a flaw (No).
* **Justification:** The judge must provide reasoning for each Yes/No decision, referencing specific parts of the `MODEL RESPONSE` and `IDEAL RESPONSE`.

## 3. High-Level Architecture

The system follows a pipeline/workflow architecture:

```mermaid
 graph TD

   subgraph CogniBench System
     direction LR

     %% Core Workflow Components
     A[Input Data Intake] --> B(Preprocessing: Normalization);
     B --> C["Evaluation Core (LLM Judge / Batch API)"];
     C --> D[Post-processing & Aggregation];
     D --> E[Output Generation];
     D --> LOG["Enhanced Logging"];
     B --> JSON["Robust JSON Parsing"];
     D --> SYMPY["Improved SymPy Parsing"];

     %% Data Storage & Logging
     F["Data Storage (data/, logs/, batch_intermediate_data/)"] <--> A;
     F <--> E;
     F <--> SB[Submit Structuring Batch];
     F <--> RJ[Retrieve Judging Results];
     LOG["Log Storage (.log)"] <--> G;
     LOG <--> SB;
     LOG <--> RS[Retrieve Structuring Results];
     LOG <--> PJ[Prepare Judging Batch];
     LOG <--> RJ;

     %% Orchestration & UI / Scripts
     subgraph "Execution Modes"
       direction TB
       G["Core Runner (evaluation_runner.py)"] -- Sync Mode --> C;
       G -- Sync Mode --> D;
       G -- Sync Mode --> E;
       I["Streamlit UI (app.py)"] -- Triggers --> G;

       SB["Script: run_batch_evaluation.py"] -- Batch Mode --> C;
       RS["Script: retrieve_batch_results.py (Structuring)"] -- Batch Mode --> C;
       PJ["Script: prepare_judging_batch.py"] -- Batch Mode --> C;
       RJ["Script: retrieve_batch_results.py (Judging)"] -- Batch Mode --> C;
       RJ -- Batch Mode --> D;
       RJ -- Batch Mode --> E;
     end

     I -- Reads Results --> F;
     I -- Uses --> CM[Global COLOR_MAP Constant];

     %% UI Enhancements
     subgraph "Streamlit UI Enhancements"
         direction TB
         I --> UI1[Expandable Sections];
         UI1 --> UI2[Consistent Graph Coloring];
         UI2 --> UI3[Clustered Charts & Filters];
         UI3 --> UI4[Log Expander & Cache Clear];
     end

     %% Evaluation Core Details
     subgraph "Evaluation Core (LLM Judge / Batch API)"
         direction LR
         C1[Prompt Constructor] --> C2["Judge LLM Invocation (Sync / Batch Submit)"];
         C2 --> C3["Response Parser / Batch Result Retrieval"];
         C1 -- Uses --> JP[Judging Prompt];
         C1 -- Uses --> SP[Structuring Prompt];
     end

     %% Optional API Layer (Not used by Streamlit App)
     H[(API Layer)];

     %% Styling
     style F fill:#9575cd,stroke:#333,stroke-width:2px
     style LOG fill:#b0bec5,stroke:#333,stroke-width:2px
     style JSON fill:#ffab91,stroke:#333,stroke-width:2px
     style SYMPY fill:#90caf9,stroke:#333,stroke-width:2px
     style C fill:#64b5f6,stroke:#333,stroke-width:2px
     style H fill:#4db6ac,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
     %% Core Runner & UI
     style G fill:#ffca28,stroke:#333,stroke-width:2px
     style I fill:#a5d6a7,stroke:#333,stroke-width:2px
     style CM fill:#ff7043,stroke:#333,stroke-width:1px
     style UI1 fill:#81c784,stroke:#333,stroke-width:1px
     style UI2 fill:#81c784,stroke:#333,stroke-width:1px
     style UI3 fill:#81c784,stroke:#333,stroke-width:1px
     style UI4 fill:#81c784,stroke:#333,stroke-width:1px
     %% Batch Scripts
     style SB fill:#ffe082,stroke:#333,stroke-width:1px
     style RS fill:#ffe082,stroke:#333,stroke-width:1px
     style PJ fill:#ffe082,stroke:#333,stroke-width:1px
     style RJ fill:#ffe082,stroke:#333,stroke-width:1px
     %% Prompts
     style JP fill:#4fc3f7,stroke:#333,stroke-width:1px
     style SP fill:#4fc3f7,stroke:#333,stroke-width:1px
   end
 ```

*Note: This diagram shows the main components and their relationships. The system supports two main execution modes: Synchronous (via Streamlit UI or `run_batch_evaluation.py` with batch disabled) and Asynchronous Batch API (via sequential execution of scripts as documented in the README).*

## 4. Detailed Workflows

### 4.1 Synchronous Workflow (Streamlit UI or Batch Disabled)

The following diagram illustrates the sequence of operations when using the Streamlit UI or running `run_batch_evaluation.py` with `batch_settings.enabled: false` in `config.yaml`:

```mermaid
 sequenceDiagram
     participant User
     participant UI as Streamlit UI (app.py)
     participant CoreRunner as Core Runner (evaluation_runner.py)
     participant CoreWorkflow as core/workflow.py
     participant LLMClient as LLM Clients
     participant DataStore as Data Storage (data/, logs/)

     User->>+UI: Upload Raw JSON File(s)
     User->>UI: Configure Models & Prompts
     User->>UI: View Prompts & Config
     User->>+UI: Click "Run Evaluation"
     UI->>UI: Generate AppConfig
     UI->>+CoreRunner: Start evaluation_worker thread (pass AppConfig, stop_event)

     CoreRunner->>CoreRunner: Create timestamped output dir in DataStore
     CoreRunner->>CoreRunner: Load Input Data (from AppConfig paths)

     loop For Each Task/Model in Input File(s)
         alt Check Stop Event
             CoreRunner->>CoreRunner: if stop_event.is_set(): break
         end
         CoreRunner->>+CoreWorkflow: run_single_task_evaluation_core(...)
         CoreWorkflow->>DataStore: Log STRUCTURING_CALL
         CoreWorkflow->>LLMClient: Invoke Structuring LLM
         LLMClient-->>CoreWorkflow: Structured Response
         CoreWorkflow->>DataStore: Log JUDGING_CALL
         CoreWorkflow->>LLMClient: Invoke Judging LLM
         LLMClient-->>CoreWorkflow: Judging Response
         CoreWorkflow->>DataStore: Append to _evaluations.jsonl
         CoreWorkflow-->>-CoreRunner: Return task results
     end

     CoreRunner->>CoreRunner: Format _evaluations.jsonl -> _evaluations_formatted.json
     CoreRunner->>+DataStore: Read Input Data (again)
     DataStore-->>-CoreRunner: Return input data
     CoreRunner->>+DataStore: Read _evaluations_formatted.json
     DataStore-->>-CoreRunner: Return formatted evaluations
     CoreRunner->>CoreRunner: Combine Data & Calculate Summary
     CoreRunner->>+DataStore: Write _final_results.json
     DataStore-->>-CoreRunner: Write complete

     CoreRunner-->>UI: Return list of _final_results.json paths (via queue)

     User->>+UI: Click "Stop Processing" (Optional)
     UI->>CoreRunner: stop_event.set()

     UI->>UI: Receive results paths from queue
     UI->>+DataStore: Read _final_results.json (using load_and_process_results)
     DataStore-->>-UI: Return results data
     UI->>UI: Process data (Pandas)
     UI->>UI: Generate Graphs (Plotly) & Tables
     UI-->>-User: Display Results & Visualizations

     User->>+UI: Load Existing Results (Optional)
     UI->>+DataStore: Read selected _final_results.json
     DataStore-->>-UI: Return results data
     UI->>UI: Process & Display

     User->>+UI: Clear Caches (Optional)
     UI->>UI: st.cache_data.clear()
     UI->>LLMClient: clear_openai_cache()
     UI-->>-User: Confirmation
 ```

### 4.2 Asynchronous Batch API Workflow

When `batch_settings.enabled: true` is set in `config.yaml`, the evaluation process requires manual execution of several scripts in sequence, as the OpenAI Batch API calls are asynchronous and can take time (up to 24 hours).

**Step 1: Submit Structuring Batch**
*   **Script:** `scripts/run_batch_evaluation.py`
*   **Action:** Reads input data, generates all structuring requests, submits them as a single batch job to OpenAI.
*   **Output:** Logs a **Structuring Batch ID**. Saves intermediate data mapping request IDs to original data in `batch_intermediate_data/`.

**Step 2: Wait for Structuring Batch Completion**
*   **Action:** User monitors the batch job status via OpenAI's dashboard or waits.

**Step 3: Retrieve Structuring Results & Submit Judging Batch**
*   **Script 1:** `scripts/retrieve_batch_results.py --stage structuring`
*   **Action:** Uses the Structuring Batch ID to check status, download results upon completion, parse results, map them to intermediate data, handle errors, and save the combined structured output (e.g., `structured_output.json`).
*   **Script 2:** `scripts/prepare_judging_batch.py`
*   **Action:** Reads the structured output file, generates all judging requests, submits them as a single batch job to OpenAI.
*   **Output:** Logs a **Judging Batch ID**.

**Step 4: Wait for Judging Batch Completion**
*   **Action:** User monitors the batch job status via OpenAI's dashboard or waits.

**Step 5: Retrieve Judging Results & Finalize**
*   **Script:** `scripts/retrieve_batch_results.py --stage judging`
*   **Action:** Uses the Judging Batch ID to check status, download results upon completion, parse results, map them, perform post-processing (parsing judge output, final answer verification, scoring), and save the final evaluation results (e.g., to `final_results.jsonl`).

*(Refer to `README.md` for detailed command examples).*

## 5. Component Breakdown

* **A. Input Data Intake (Conceptual - Handled by `scripts/run_single_evaluation.py`):**
  * **Function:** Reads pre-ingested data (typically from `*_ingested_*.json` files generated by `ingest_rlhf_data.py`). This data includes `task_id`, `prompt`, `ideal_response`, `final_answer`, `model_responses` (list), `human_evaluations` (list), and `metadata`.
  * **Interface:** Reads a JSON file specified via command-line argument (`--input-data`).
  * **Validation:** Assumes the input file structure is correct as produced by the ingestion script.
* **B. Preprocessing Module:**
  * **Function:** Prepares inputs for the Judge LLM.
  * **Sub-Tasks:**
    * *Format Normalization:* Basic text normalization (Unicode, whitespace). This is the primary preprocessing step applied before structuring.
    * *(Removed)* LaTeX Notation Conversion: This was previously handled here but is now removed.
    * *(Removed)* Model Final Answer Extraction (Regex): Previously done via regex patterns. This step is **removed**; the **Structuring LLM** is now solely responsible for identifying and extracting the `final_answer` from the model's response during the structuring phase.
    * *(Future) Response Segmentation:* (Not currently implemented) Could break down responses into logical sections.
    * *(Future) Sanitization:* (Not currently implemented) Could remove sensitive content if needed.
* **C. Evaluation Core (LLM Judge / Batch API):**
  * **Function:** Performs the core evaluation using a powerful Judge LLM based on the L1 rubric, either via synchronous calls or by preparing/retrieving asynchronous Batch API jobs.
  * **C1. Prompt Constructor:**
    * Dynamically builds the prompt payload (messages, model, parameters) for the Structuring and Judging LLMs.
    * **Key Contents:**
      * **System Message/Instructions:** Define the role ("You are an expert mathematician evaluating an AI's solution..."), the task (evaluate based on the rubric), the required output format (e.g., JSON with Yes/No and Justification for each L1 parameter).
      * **The L1 Rubric:** Embed the full definition of each L1 parameter and its L2 subcomponents, including the Yes/No criteria and examples of failures.
      * **Input Data:** Include the `PROMPT`, the preprocessed `MODEL RESPONSE`, the preprocessed `IDEAL RESPONSE`, and the ground-truth `FINAL_ANSWER`. Clearly label each piece.
      * **Specific Instructions:** Guide the LLM to compare `MODEL RESPONSE` against `IDEAL RESPONSE` and the rubric criteria. Explicitly ask for:
        * A Yes/No judgment for each L1 parameter.
        * Detailed justification for each judgment, citing specific evidence from the `MODEL RESPONSE` (and `IDEAL RESPONSE` where relevant).
        * *(Optional)* Yes/No judgments for L2 subcomponents.
        * Identification of specific errors (calculation, logical fallacy, misinterpreted constraint, etc.).
    * **Strategy:** May require multiple sequential prompts (e.g., one prompt per L1 parameter) or a single complex prompt. Experimentation needed. Single complex prompt is often preferred if context window allows, for better holistic understanding.
  * **C2. Judge LLM Invocation (Sync / Batch Submit):**
    * **Synchronous Mode:** Sends the constructed prompt payload directly to the chosen LLM API (e.g., OpenAI API via `openai_client.py`). Handles retries and errors.
    * **Batch Mode:** Prepares request dictionaries (including `custom_id`, `method`, `url`, `body`) for structuring and judging. Uses `batch_processor.py` to format requests into JSONL, upload the file, and create the batch job via the OpenAI API. Logs the submitted Batch ID.
    * **Logging:** Logs initiation of synchronous structuring/judging calls or batch job submissions.
  * **C3. Response Parser / Batch Result Retrieval:**
    * **Synchronous Mode:** Receives the raw output from the LLM API call. Parses the output (e.g., using `_parse_json_string`, `parse_judge_response`) to extract structured data. Validates against expected format.
    * **Batch Mode:** Uses `batch_processor.py` and `retrieve_batch_results.py` to poll for batch completion, download the result file (JSONL), parse each line, handle per-request errors reported in the results, and map results back using intermediate data.
* **D. Post-processing & Aggregation:**
  * **Function:** Refines and aggregates the raw evaluation results.
  * **Sub-Tasks:**
    * *Final Answer Verification:* Compares the `final_answer` (extracted by the **Structuring LLM** as part of the structured response object) with the ground-truth `final_answer` (from the ingested data). Uses `sympy` (if available) for mathematical/symbolic equivalence checking (including LaTeX parsing), falling back to normalized string comparison otherwise.
    * *Consistency Checks (Optional):* Implement checks, e.g., if `Results Formulae` is 'No' due to final answer mismatch, ensure justification aligns. Could involve rule-based checks or even another LLM call for self-consistency review.
    * *Score Aggregation:* Calculate summary statistics if needed (e.g., total L1 'Yes' count). Determine an overall PASS/FAIL based on predefined rules (e.g., requires 'Yes' on all L1 parameters, or specific combinations). The reference doc suggests *any* 'No' results in overall failure for that component.
    * *Human Review Flags:* Identify evaluations needing human review based on parsing errors, trivial justifications for negative/partial scores, or potentially other configurable rules (e.g., answer mismatches).
* **E. Output Generation:**
  * **Function:** Formats the final evaluation into machine-readable files.
  * **Formats:** JSON Lines (`.jsonl`) for raw judge outputs, formatted JSON (`.json`) for cleaned evaluations, and a final combined JSON (`.json`) aggregating all data per task. All keys in these files are converted to `snake_case`.
  * **Content:** The final combined file (`*_final_results.json`) includes `task_id`, `prompt`, `ideal_response`, `final_answer`, `metadata`, and a list of `evaluations` (one per model), each containing `model_id`, the raw `model_response` text, `human_evaluation` details, and `judge_evaluation` details (rubric scores, justifications, etc.). Stores the structured output in `Data Storage`.
* **F. Data Storage:**
  * **Function:** Persists all relevant data for tracking, analysis, auditing, and potential future fine-tuning of the Judge LLM.
  * **Technology:** Primarily uses JSON Lines (`.jsonl`) for efficient appending of evaluation results during runs, stored on the filesystem. Formatted JSON (`.json`) versions are generated for readability, and a final combined JSON aggregates results. Databases (SQL/NoSQL) could be integrated for more complex querying and management.
  * **Structure (Filesystem):**
    * `data/Batch-XXX_YYYYMMDD_HHMM/`: Timestamped subdirectory created by `run_batch_evaluation.py` for each run.
      * `*_ingested_*.json`: Data prepared for evaluation by `ingest_rlhf_data.py`.
      * `*_evaluations.jsonl`: Detailed judge evaluation results for each model response (append-friendly, `snake_case` keys).
      * `*_evaluations_formatted.json`: Pretty-printed JSON array version of the `.jsonl` file (excluding raw judge output, `snake_case` keys).
      * `*_final_results.json`: Final aggregated output, grouping results by `task_id` and combining ingested data (`prompt`, `ideal_response`, `final_answer`, `metadata`, `human_evaluations`) with judge evaluations for each model (`snake_case` keys).
    * `logs/YYYYMMDD_HHMM/`: Timestamped directory for each run. Contains detailed execution information (DEBUG level and above), including specific logs for structuring and judging call initiation. Console output is typically limited to WARNING level and above, plus `tqdm` progress bars.
      * `backend.log`: Logs from core scripts, API, batch processing, etc.
      * `streamlit.log`: Logs specifically from the Streamlit UI operations.
* **G. Workflow Orchestrator / Scripts:**
  * **Function:** Manages the execution flow depending on the mode.
  * **Technology:**
    * **Synchronous:** `core/evaluation_runner.py` (called by Streamlit UI or `scripts/run_batch_evaluation.py` with batch disabled) manages the A -> B -> C -> D -> E flow directly.
    * **Batch API:** Requires manual execution of sequential scripts: `scripts/run_batch_evaluation.py` (Submit Structuring) -> `scripts/retrieve_batch_results.py` (Retrieve Structuring) -> `scripts/prepare_judging_batch.py` (Submit Judging) -> `scripts/retrieve_batch_results.py` (Retrieve Judging & Finalize).
  * **Features:** Handles file paths, configuration loading/validation, logging setup (`core/log_setup.py`), and calls core logic (`core/workflow.py`, `core/batch_processor.py`, etc.).
* **H. API Layer:**
  * **Function:** Provides an external interface to submit evaluation requests and retrieve results.
  * **Technology:** RESTful API using frameworks like FastAPI, Flask (Python), or Node.js/Express.
  * **Endpoints:**
    * `POST /evaluate`: Submit a new evaluation request (Payload: Prompt, Model Response, Ideal Response, Correct Answer). Returns a job ID.
    * `GET /evaluate/{job_id}`: Check the status and retrieve the results of an evaluation.

## 6. Data Flow

**Synchronous Mode:**
1. User triggers evaluation (via UI or script).
2. `evaluation_runner.py` loads data, calls `workflow.py`.
3. `workflow.py` preprocesses, calls LLM Client for structuring, calls LLM Client for judging, parses results, post-processes, and saves results via `output_writer.py` to `Data Storage`.

**Batch API Mode:**
1. User runs `run_batch_evaluation.py`. Script loads data, generates structuring requests, calls `batch_processor.py` to submit batch job, saves intermediate data map to `Data Storage`. Logs Structuring Batch ID.
2. **(Wait)** User waits for batch completion.
3. User runs `retrieve_batch_results.py --stage structuring`. Script calls `batch_processor.py` to check status, download & parse results. Loads intermediate map from `Data Storage`. Maps results and saves structured output to `Data Storage`.
4. User runs `prepare_judging_batch.py`. Script loads structured output from `Data Storage`, generates judging requests, calls `batch_processor.py` to submit batch job. Logs Judging Batch ID.
5. **(Wait)** User waits for batch completion.
6. User runs `retrieve_batch_results.py --stage judging`. Script calls `batch_processor.py` to check status, download & parse results. Loads intermediate map. Maps results. Calls `workflow.py` functions for post-processing. Calls `output_writer.py` to save final results to `Data Storage`.

## 7. Technology Stack Choices (Example)

* **Programming Language:** Python (due to excellent data science, ML/LLM libraries, and web frameworks).
* **LLM Judge:** Configurable via `config.yaml` (e.g., GPT-4o, Claude 3 Opus, Gemini 1.5 Pro). Choice depends on performance, context window, cost, API features.
* **Math Verification:** `sympy` (optional dependency).
* **Configuration:** `PyYAML`.
* **API Framework:** FastAPI.
* **UI Framework:** Streamlit.
* **Workflow Orchestration:** Python scripts.
* **Data Storage:** Filesystem (JSON, JSONL).
* **Deployment:** Docker (optional), local execution.

## 8. Key Design Considerations & Trade-offs

* **Prompt Engineering:** *Critical Path.* Requires iterative refinement. Prompts must be unambiguous, include the full rubric, and clearly specify the desired output format. Consider using Few-Shot examples within the prompt.
* **Judge LLM Choice:** Stronger models (GPT-4o, Opus) yield better reasoning but are slower/more expensive. Need to balance quality vs. cost/latency.
* **Structured Output:** Relying on LLMs for structured output (JSON) is crucial but not foolproof. Robust parsing, validation (checking required criteria/scores from config), and comprehensive error reporting are implemented. Function Calling/Tool Use could further improve reliability.
* **Consistency:** LLM outputs can vary. Use `temperature=0`. Consider running evaluations multiple times (e.g., 2 out of 3 agreement) for critical assessments, although this increases cost.
* **Human-in-the-Loop:** *Essential.* Implement a review interface for experts to validate/correct Judge LLM outputs, especially during initial calibration and for ambiguous cases. This feedback loop is crucial for improving prompts and potentially fine-tuning the judge model in the future.
* **Scalability:**
   * **Synchronous:** Limited by sequential processing.
   * **Batch API:** Significantly more scalable due to parallel processing by OpenAI and higher rate limits. Handles large datasets efficiently.
* **Asynchronicity (Batch Mode):** Requires users to manage a multi-step process with potentially long waiting periods between steps. Requires careful tracking of Batch IDs.
* **State Management (Batch Mode):** Intermediate data (mapping `custom_id` to original data) must be reliably saved and loaded between script executions. Currently stored in JSON files in `batch_intermediate_data/`.
* **Error Handling:**
   * **Synchronous:** Handled via retries within the LLM client.
   * **Batch API:** Requires handling batch-level failures (e.g., invalid input file) and per-request errors reported in the results file. The `retrieve_batch_results.py` script implements checks for per-request errors.
* **Context Window Limitations:** Applies to both modes. Ensure the chosen LLM has sufficient context window.
* **Cost Management:** Batch API offers significant cost savings (50% discount) compared to standard synchronous calls, making large-scale evaluations more feasible. Caching identical structuring requests (especially for ideal responses) is implemented.
* **Rubric Evolution:** The evaluation rubric (expected criteria, allowed scores) is now defined in `config.yaml`, making it easier to modify without code changes. The prompt template path is also configured. The judging and structuring prompts have been updated to align precisely with these rubric criteria, ensuring consistency and accuracy in evaluations.
* **Security:** If handling sensitive/proprietary prompts or model outputs, ensure appropriate data handling, access controls, and potentially use models with stronger data privacy guarantees (e.g., Azure OpenAI). Applies to both modes.

## 9. Future Enhancements

* **Fine-tuning the Judge LLM:** Use human-corrected evaluations as training data to fine-tune a base LLM for the specific task of rubric-based math evaluation within CogniBench, potentially improving accuracy and consistency.
* **Multi-Judge Consensus:** Employ multiple different Judge LLMs within CogniBench and aggregate their results for higher confidence.
* **Visual Understanding:** Integrate capabilities to analyze diagrams or plots if part of the prompt/response (requires multi-modal models).
* **Advanced Answer Verification:** Implement more sophisticated verification for different answer types (e.g., code execution, set comparison, numerical tolerance).
* **Automated Ideal Response Generation (Research):** Explore using powerful LLMs within the CogniBench workflow to *generate* the `IDEAL RESPONSE` as a starting point for human experts, speeding up the process.
* **Integration with L0:** Tightly integrate the L1 Judge output from CogniBench back into the L0 Golden Prompt Discovery process for richer failure analysis.
* **User Interface (Streamlit):** Further enhance the Streamlit application (`cognibench_agent/app.py`) for better analysis and usability:
  * **Refactoring Complete (April 2025):**
    * UI modularized into functions.
    * Direct integration with `core.evaluation_runner` (no subprocess).
    * Background thread for non-blocking evaluation runs.
    * Detailed log capture and display in UI expander.
    * Graceful evaluation cancellation via "Stop Run" button.
    * Persistent results saving to `data/<BatchName>_YYYYMMDD_HHMM/`.
    * Enhanced charts: Clustered bars, "All Criteria" view, Model filtering.
    * Combined cache clearing button (Streamlit data + LLM cache).
    * Correct parsing of input/output data structures.
    * Renamed UI directory to `cognibench_agent`.
    * Fixed ingestion step within Streamlit workflow.
    * Resolved session state modification error.
  * **Future Ideas:**
    * Interactive Filtering: Allow clicking on graph elements (e.g., bars) to filter the data tables below.
    * Detailed Task Modal: Implement a pop-up or dedicated view to show all details (prompt, responses, full evaluation) for a selected task row.
    * Side-by-Side Comparison: Add a mode to select two models and compare their responses and evaluations directly on the same tasks.
    * Results Export: Add functionality to export the filtered data from the explorers (Task-Level, Human Review) to CSV/Excel.
    * Configuration Presets: Allow saving and loading common LLM Judge configurations.
  * **Historical Run Comparison:** Develop features to load and compare results across different evaluation runs (multiple `_final_results.json` files).
  
  **Recent Enhancements (April 2025):**
  
  - **Graph Regeneration from Existing Data:** Added functionality to regenerate evaluation graphs directly from existing evaluation data without re-running evaluations. Users can select one or more folders containing previous evaluation results (`<BatchName>_final_results.json`) to quickly visualize past results.
  - **Mutually Exclusive Actions:** Implemented a clear UI distinction between "Run Evaluations" and "Recreate Graphs from Existing Data" using a radio button selection. This ensures users explicitly choose one action at a time, preventing confusion and unintended operations.
  - **Folder Sorting by Modification Time:** Enhanced folder selection by sorting available folders based on their modification time, displaying the most recently modified folders at the top for improved usability.
  - **Human-Readable Duration:** The "Total Evaluation Duration" metric in the UI summary is now displayed in a more readable format (e.g., "1 hr, 2 min, 3 s").
