import json
import subprocess
import sys
from pathlib import Path

import pytest

# Add the script's directory to sys.path to allow importing ingest_rlhf_data
# This assumes the test is run from the workspace root or pytest handles paths correctly.
# A more robust solution might involve package structure or PYTHONPATH adjustments.
SCRIPT_DIR = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR.resolve()))

# Now import the function (assuming the script can be imported)
# If the script relies heavily on __main__, we might need to run it as a subprocess instead.
# Let's try importing first, but fall back to subprocess if needed.
try:
    # Ensure the script doesn't execute __main__ on import
    _original_name = __name__
    __name__ = "_test_import_"
    from ingest_rlhf_data import ingest_rlhf_data

    __name__ = _original_name  # Restore original name
    RUN_AS_SUBPROCESS = False
except ImportError as e:
    print(f"Import failed: {e}. Will run script as subprocess.", file=sys.stderr)
    # If direct import fails (e.g., due to __main__ block complexity or relative imports within script)
    # We'll run the script as a subprocess in tests.
    RUN_AS_SUBPROCESS = True
    INGEST_SCRIPT_PATH = SCRIPT_DIR / "ingest_rlhf_data.py"


TEST_DATA_DIR = Path(__file__).parent / "test_data"
VALID_INPUT_PATH = TEST_DATA_DIR / "valid_input.json"
MISSING_IDEAL_PATH = TEST_DATA_DIR / "missing_ideal.json"
INVALID_SYNTAX_PATH = TEST_DATA_DIR / "invalid_syntax.json"
YES_NO_PATH = TEST_DATA_DIR / "yes_no_conversion.json"
NON_EXISTENT_PATH = TEST_DATA_DIR / "non_existent_file.json"


# Helper function to run script as subprocess
def run_ingest_script_subprocess(input_path):
    """Runs the ingestion script as a subprocess."""
    if not INGEST_SCRIPT_PATH.exists():
        pytest.fail(f"Ingestion script not found at {INGEST_SCRIPT_PATH}")

    command = [
        sys.executable,
        str(INGEST_SCRIPT_PATH),
        str(input_path),
    ]
    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    return process


@pytest.fixture(autouse=True)
def ensure_test_data_dir(tmp_path):
    """Ensure the base directory for test outputs exists."""
    pass  # tmp_path fixture handles temporary directories


# Use parametrize to run tests via direct call and subprocess if possible/needed
# For now, simplifying: use direct call if import worked, otherwise use subprocess.


@pytest.mark.skipif(
    RUN_AS_SUBPROCESS, reason="Direct import failed, cannot run direct call test."
)
def test_successful_ingestion_direct(tmp_path):
    """Tests successful ingestion via direct function call."""
    output_file = tmp_path / "output_valid_direct.json"
    ingest_rlhf_data(VALID_INPUT_PATH, output_file)
    assert output_file.exists()

    with output_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 1
    item = data[0]
    assert item["task_id"] == 101
    assert item["prompt"] == "User prompt 1"
    assert item["ideal_response"] == "Ideal 1"
    assert "model_responses" in item
    assert len(item["model_responses"]) == 1
    assert item["model_responses"][0]["model_id"] == "model_a"
    assert "human_evaluations" in item
    assert len(item["human_evaluations"]) == 1
    eval_details = item["human_evaluations"][0]
    assert eval_details["model_id"] == "model_a"
    assert eval_details["model_failure"] is False
    assert eval_details["failure_comments"] == "Looks good."
    assert "metadata" in item
    assert item["metadata"]["subject"] == "Algebra"
    assert item["metadata"]["complexity"] == "Basic"


@pytest.mark.skipif(
    RUN_AS_SUBPROCESS, reason="Direct import failed, cannot run direct call test."
)
def test_key_standardization_and_boolean_direct(tmp_path):
    """Tests key standardization and boolean conversion via direct call."""
    output_file = tmp_path / "output_yes_no_direct.json"
    ingest_rlhf_data(YES_NO_PATH, output_file)
    assert output_file.exists()

    with output_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 1
    item = data[0]
    assert len(item["human_evaluations"]) == 1
    eval_details = item["human_evaluations"][0]
    assert eval_details["model_id"] == "model_c"
    assert eval_details["is_correct"] is True
    assert eval_details["needs_review"] is False


# --- Subprocess Tests for Error Handling ---

# Modify subprocess helper slightly if needed, or use directly
# These tests assume the script exits with code 1 on error and prints to stderr


@pytest.mark.skipif(
    not RUN_AS_SUBPROCESS,
    reason="Requires running as subprocess to test exit codes/stderr",
)
def test_missing_ideal_response_subprocess(tmp_path):
    """Tests skipping task with missing ideal (subprocess)."""
    process = run_ingest_script_subprocess(MISSING_IDEAL_PATH)
    # Script should succeed overall but warn to stderr
    assert process.returncode == 0
    assert "Warning: Skipping task ID 102" in process.stderr
    # Check output file content (need to parse the dynamic path from stdout)
    output_path_str = process.stdout.strip()
    assert output_path_str, "Script did not print output path to stdout"
    output_path = Path(output_path_str)
    assert output_path.exists(), f"Output file {output_path} not found"
    with output_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data == [], "Output file should be an empty list when task is skipped"


@pytest.mark.skipif(
    not RUN_AS_SUBPROCESS,
    reason="Requires running as subprocess to test exit codes/stderr",
)
def test_file_not_found_subprocess(tmp_path):
    """Tests file not found error (subprocess)."""
    process = run_ingest_script_subprocess(NON_EXISTENT_PATH)
    assert process.returncode == 1
    assert "Error: Input file not found" in process.stderr


@pytest.mark.skipif(
    not RUN_AS_SUBPROCESS,
    reason="Requires running as subprocess to test exit codes/stderr",
)
def test_json_decode_error_subprocess(tmp_path):
    """Tests JSON decode error (subprocess)."""
    process = run_ingest_script_subprocess(INVALID_SYNTAX_PATH)
    assert process.returncode == 1
    assert "Error: Could not decode JSON" in process.stderr


# Optional: Add a subprocess test for the valid case if needed,
# similar to test_missing_ideal_response_subprocess but checking content.
@pytest.mark.skipif(not RUN_AS_SUBPROCESS, reason="Requires running as subprocess")
def test_successful_ingestion_subprocess(tmp_path):
    """Tests successful ingestion via subprocess."""
    process = run_ingest_script_subprocess(VALID_INPUT_PATH)
    assert process.returncode == 0
    assert "Error" not in process.stderr  # Basic check for no errors
    output_path_str = process.stdout.strip()
    assert output_path_str, "Script did not print output path to stdout"
    output_path = Path(output_path_str)
    assert output_path.exists(), f"Output file {output_path} not found"

    # Basic content check
    with output_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["task_id"] == 101
