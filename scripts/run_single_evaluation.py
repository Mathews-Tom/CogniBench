"""
CogniBench Single Evaluation Script (Refactored).

Provides a command-line interface to run the CogniBench evaluation workflow
on a single input data file using the core evaluation logic. This script
acts as a wrapper around the `run_evaluation_from_file` function.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path to allow absolute import of core modules
APP_DIR = Path(__file__).resolve().parent
COGNIBENCH_ROOT = APP_DIR.parent
if str(COGNIBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(COGNIBENCH_ROOT))

try:
    # Import the new core runner function
    from core.evaluation_runner import run_evaluation_from_file
    from core.log_setup import setup_logging
except ImportError as e:
    print(f"Error importing core modules: {e}", file=sys.stderr)
    print(
        "Ensure the script is run from the project root or the PYTHONPATH is set correctly.",
        file=sys.stderr,
    )
    sys.exit(1)

# Setup logging first
# Consider making log level configurable via CLI argument if needed
setup_logging(log_level=logging.INFO)  # Default to INFO for the script itself
logger = logging.getLogger(__name__)  # Use __name__ for logger


def main() -> None:
    """Parses arguments and calls the core evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Run CogniBench evaluation on a single input data file using core logic."
    )
    parser.add_argument(
        "--input-data",
        type=Path,
        required=True,
        help="Path to the input JSON data file containing evaluation tasks.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        help="Optional path to the output JSONL file for detailed results.",
    )
    parser.add_argument(
        "--use-structured",
        action="store_true",
        help="Use structured responses (ideal and model) if available in the input data and config allows.",
    )
    # Add argument for log level if desired
    # parser.add_argument(
    #     "--log-level",
    #     default="INFO",
    #     choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    #     help="Set the logging level.",
    # )

    args = parser.parse_args()

    # Optional: Reconfigure logging level based on args
    # log_level_numeric = getattr(logging, args.log_level.upper(), logging.INFO)
    # logging.getLogger().setLevel(log_level_numeric) # Set root logger level
    # logger.info(f"Logging level set to {args.log_level}")

    logger.info("Calling core evaluation runner...")
    success = run_evaluation_from_file(
        input_path=args.input_data,
        config_path=args.config,
        output_jsonl_path=args.output_jsonl,
        use_structured=args.use_structured,
    )

    if success:
        logger.info("Evaluation script finished successfully.")
        sys.exit(0)
    else:
        logger.error("Evaluation script finished with errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
