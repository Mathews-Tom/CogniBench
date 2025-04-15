"""
CogniBench Batch Evaluation Orchestration Script (Refactored).

This script serves as the main entry point for running batch evaluations.
It orchestrates the process by:
1. Parsing command-line arguments for input file, config file, and options.
2. Calling the `ingest_rlhf_data.py` script via subprocess to preprocess the raw input data.
3. Calling the `run_batch_evaluation_core` function from the core library
    to perform the actual evaluation workflow using the ingested data and configuration.
4. Managing output directories and logging progress.
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to sys.path to allow absolute import of core modules
APP_DIR = Path(__file__).resolve().parent
COGNIBENCH_ROOT = APP_DIR.parent
if str(COGNIBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(COGNIBENCH_ROOT))

try:
    from core.config import AppConfig  # Keep for loading config path

    # Import the new core batch runner function
    from core.evaluation_runner import run_batch_evaluation_core
    from core.log_setup import setup_logging
except ImportError as e:
    print(f"Error importing core modules: {e}", file=sys.stderr)
    print(
        "Ensure the script is run from the project root or the PYTHONPATH is set correctly.",
        file=sys.stderr,
    )
    sys.exit(1)

# Setup logging
setup_logging(log_level=logging.INFO)  # Default to INFO
logger = logging.getLogger(__name__)


def run_command(command_list: list[str]) -> Optional[subprocess.CompletedProcess]:
    """Runs a command using subprocess and handles errors."""
    logger.debug("Running command: %s", " ".join(command_list))
    try:
        # Use capture_output=True for ingestion to get the output path
        process = subprocess.run(
            command_list, check=True, capture_output=True, text=True, encoding="utf-8"
        )
        logger.info("Command successful.")
        if process.stdout:
            logger.debug("stdout:\n%s", process.stdout)
        if process.stderr:
            logger.warning("stderr:\n%s", process.stderr)  # Log stderr as warning
        return process
    except subprocess.CalledProcessError as e:
        logger.error("Error running command: %s", " ".join(command_list))
        logger.error("Return code: %s", e.returncode)
        if e.stdout:
            logger.error("stdout:\n%s", e.stdout)
        if e.stderr:
            logger.error("stderr:\n%s", e.stderr)
        return None
    except FileNotFoundError:
        logger.error("Command not found (%s)", command_list[0])
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error running command {' '.join(command_list)}: {e}",
            exc_info=True,
        )
        return None


def main() -> None:
    """Parses arguments, runs ingestion, and calls the core batch evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end CogniBench batch evaluation using core logic."
    )
    parser.add_argument(
        "input_batch_files",
        nargs="+",  # Accept one or more arguments
        type=Path,
        help="Paths to one or more input raw RLHF JSON batch files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the CogniBench YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional: Specify a parent directory for the timestamped batch output folder. Defaults to './CogniBench/data/'.",
    )
    parser.add_argument(
        "--use-structured",
        action="store_true",
        help="Use structured responses (ideal and model) if available during evaluation.",
    )
    # Add argument for log level if desired
    # parser.add_argument(
    #     "--log-level",
    #     default="INFO",
    #     choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    #     help="Set the logging level.",
    # )

    args = parser.parse_args()

    # --- Validate Input Paths ---
    input_files_valid = True
    for file_path in args.input_batch_files:
        if not file_path.is_file():
            logger.error(f"Input batch file not found: {file_path}")
            input_files_valid = False
    if not input_files_valid:
        sys.exit(1)
    if not args.config.is_file():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    # --- Load Config (only needed for path validation here, core runner loads full config) ---
    try:
        # We load it here mainly to potentially use config values for output dir logic if needed
        # but the core runner will load and validate it properly.
        config_obj = AppConfig.load_from_path(args.config)
        logger.info(f"Configuration preamble check successful: {args.config}")
    except Exception as e:
        logger.error(
            f"Failed initial config load/validation from {args.config}: {e}",
            exc_info=True,
        )
        sys.exit(1)

    # --- Determine and Create Output Directory ---
    # Combine stems of all input files for the directory name
    batch_stems = [f.stem for f in args.input_batch_files]
    combined_stem = "_".join(
        sorted(batch_stems)
    )  # Sort for consistent naming regardless of order
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")  # Added seconds for uniqueness
    output_subdir_name = f"{combined_stem}_{timestamp}"

    # Default parent is CogniBench/data, allow override
    parent_output_dir = args.output_dir if args.output_dir else COGNIBENCH_ROOT / "data"
    batch_output_dir = parent_output_dir / output_subdir_name

    try:
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created batch output directory: {batch_output_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directory {batch_output_dir}: {e}")
        sys.exit(1)

    # --- Step 1: Ingestion ---
    logger.info("--- Starting Step 1: Ingestion ---")
    ingestion_script_path = APP_DIR / "ingest_rlhf_data.py"
    ingestion_command = [
        sys.executable,  # Use the same python interpreter
        str(ingestion_script_path),
        *[str(f) for f in args.input_batch_files],  # Pass all input files
        "--output-dir",  # Pass the specific batch output directory
        str(batch_output_dir),
    ]

    ingestion_process = run_command(ingestion_command)

    if ingestion_process is None or ingestion_process.returncode != 0:
        logger.error("Ingestion script failed. Aborting.")
        sys.exit(1)

    # Capture the output path from stdout and strip thoroughly
    ingested_file_path_str = ingestion_process.stdout.strip()
    if not ingested_file_path_str:
        logger.error("Ingestion script did not output a file path.")
        sys.exit(1)

    ingested_file_path = Path(ingested_file_path_str)
    if not ingested_file_path.is_file():
        logger.error(f"Ingested file path reported but not found: {ingested_file_path}")
        sys.exit(1)

    logger.info(
        f"--- Finished Step 1: Ingestion successful. Ingested data at: {ingested_file_path} ---"
    )

    # --- Step 2: Run Core Batch Evaluation ---
    logger.info("--- Starting Step 2: Core Batch Evaluation ---")

    # The core runner now handles evaluation, formatting, combining, and saving
    final_results_path = run_batch_evaluation_core(
        ingested_data_path=ingested_file_path,
        config=config_obj,  # Pass the loaded config object
        output_dir=batch_output_dir,  # Pass the dedicated directory for this batch
        use_structured=args.use_structured,
    )

    if final_results_path and final_results_path.is_file():
        logger.info("--- Finished Step 2: Core Batch Evaluation successful. ---")
        logger.info(f"Final combined results saved to: {final_results_path}")
        # Optionally print the path for external tools
        print(f"FINAL_RESULTS_PATH: {final_results_path.resolve()}")
        sys.exit(0)
    else:
        logger.error("--- Step 2: Core Batch Evaluation failed. ---")
        if final_results_path:
            logger.error(
                f"Core runner indicated success but final file not found: {final_results_path}"
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
