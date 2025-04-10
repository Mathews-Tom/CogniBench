# CogniBench/core/log_setup.py
import logging
import sys
from datetime import datetime  # Import datetime
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"


def setup_logging(log_level=logging.INFO):
    """Configures project-wide logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_filename = f"CogniBench_{timestamp}.log"
    log_file_path = LOG_DIR / log_filename

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create file handler
    file_handler = logging.FileHandler(
        log_file_path, encoding="utf-8"
    )  # Use dynamic path
    file_handler.setFormatter(formatter)

    # Create stream handler (for console output)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    # Set console handler level higher to only show warnings/errors
    stream_handler.setLevel(logging.WARNING)

    # Get the root logger and configure it
    root_logger = logging.getLogger()
    # Set root logger level to the lowest level we want to capture (INFO for file)
    root_logger.setLevel(log_level)  # log_level defaults to INFO

    # Remove existing handlers to avoid duplicates if setup is called multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Add the handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)  # Add console handler

    logging.info("Logging setup complete. Logging to %s", log_file_path)


# Optional: Call setup on import if desired, or call explicitly in main scripts
# setup_logging()
