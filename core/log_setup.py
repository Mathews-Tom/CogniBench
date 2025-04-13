# CogniBench/core/log_setup.py
import logging
import sys
from datetime import datetime  # Import datetime
from pathlib import Path

# Define the base log directory relative to the project root
# Assuming the script is in CogniBench/core, parent.parent gives CogniBench/
BASE_LOG_DIR = Path(__file__).parent.parent / "logs"

def setup_logging(log_level=logging.INFO):
    """Configures project-wide logging."""
    # Create the base log directory if it doesn't exist
    BASE_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Generate timestamp once for the directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    timestamp_log_dir = BASE_LOG_DIR / timestamp
    timestamp_log_dir.mkdir(parents=True, exist_ok=True) # Create timestamped subdir

    # Define specific log file paths
    backend_log_path = timestamp_log_dir / "backend.log"
    streamlit_log_path = timestamp_log_dir / "streamlit.log"
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create file handlers for backend and streamlit logs
    backend_file_handler = logging.FileHandler(backend_log_path, encoding="utf-8")
    backend_file_handler.setFormatter(formatter)
    backend_file_handler.setLevel(log_level)

    streamlit_file_handler = logging.FileHandler(streamlit_log_path, encoding="utf-8")
    streamlit_file_handler.setFormatter(formatter)
    streamlit_file_handler.setLevel(log_level)
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

    # Add only the stream handler to the root logger for console output
    root_logger.addHandler(stream_handler)

    # Get named loggers and add specific file handlers
    backend_logger = logging.getLogger("backend")
    backend_logger.setLevel(log_level)
    backend_logger.addHandler(backend_file_handler)
    # Prevent backend logs from propagating to the root logger's stream handler if desired
    # backend_logger.propagate = False # Keep True to see WARNING/ERROR on console

    streamlit_logger = logging.getLogger("streamlit")
    streamlit_logger.setLevel(log_level)
    streamlit_logger.addHandler(streamlit_file_handler)
    # Prevent streamlit logs from propagating to the root logger's stream handler if desired
    # streamlit_logger.propagate = False # Keep True to see WARNING/ERROR on console

    logging.info(
        "Logging setup complete. Backend logs: %s, Streamlit logs: %s",
        backend_log_path,
        streamlit_log_path,
    )


# Optional: Call setup on import if desired, or call explicitly in main scripts
# setup_logging()
