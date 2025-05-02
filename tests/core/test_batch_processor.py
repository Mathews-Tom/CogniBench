# CogniBench/tests/core/test_batch_processor.py

import json
import os
import tempfile
from unittest.mock import MagicMock, call, mock_open, patch

import pytest

from CogniBench.core.batch_processor import (
    check_batch_status,
    create_batch_job,
    download_batch_result_file,
    format_requests_to_jsonl,
    parse_batch_result_file,
    upload_batch_file,
)

# Assuming OpenAIClient is importable and has the expected methods
# Adjust the import path based on your project structure
from CogniBench.core.llm_clients.openai_client import OpenAIClient

# --- Fixtures ---


@pytest.fixture
def mock_openai_client():
    """Fixture for a mocked OpenAIClient."""
    client = MagicMock(spec=OpenAIClient)
    client.upload_file.return_value = {"id": "file-123", "status": "uploaded"}
    client.create_batch.return_value = {"id": "batch-abc", "status": "validating"}
    client.retrieve_batch.return_value = {
        "id": "batch-abc",
        "status": "completed",
        "output_file_id": "file-456",
        "error_file_id": None,
    }
    client.get_file_content.return_value = '{"custom_id": "req-1", "response": {"body": {"choices": [{"message": {"content": "Result 1"}}]}}}'  # JSONL line

    # Mock download_file_from_url to simulate downloading content
    def mock_download(url, dest_path):
        # Ensure the directory exists before writing the file
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "w") as f:
            f.write(
                '{"custom_id": "req-1", "response": {"body": {"choices": [{"message": {"content": "Result 1"}}]}}}\n'
            )
            f.write(
                '{"custom_id": "req-2", "response": {"body": {"choices": [{"message": {"content": "Result 2"}}]}}}\n'
            )
        return dest_path

    client.download_file_from_url.side_effect = mock_download
    return client


@pytest.fixture
def sample_requests():
    """Sample requests data."""
    return [
        {
            "custom_id": "req-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        },
        {
            "custom_id": "req-2",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "World"}],
            },
        },
    ]


# --- Test Functions ---


# 1. Test format_requests_to_jsonl
def test_format_requests_to_jsonl_success(sample_requests):
    """Test successful formatting of requests to JSONL."""
    expected_jsonl = ""
    for req in sample_requests:
        expected_jsonl += json.dumps(req) + "\n"

    # Use mock_open for file writing within the function
    m = mock_open()
    # Mock NamedTemporaryFile to control the temporary file path
    mock_temp_file = MagicMock()
    mock_temp_file.__enter__.return_value.name = "/tmp/fakefile.jsonl"
    with (
        patch("builtins.open", m),
        patch(
            "tempfile.NamedTemporaryFile", return_value=mock_temp_file
        ) as mock_tempfile_constructor,
    ):
        # Configure the context manager behavior if needed, but name is often enough
        mock_tempfile_constructor.return_value.__enter__.return_value.name = (
            "/tmp/fakefile.jsonl"
        )

        file_path = format_requests_to_jsonl(sample_requests)

        assert file_path == "/tmp/fakefile.jsonl"
        # Check if NamedTemporaryFile was called correctly (delete=False is important)
        mock_tempfile_constructor.assert_called_once_with(
            mode="w", delete=False, suffix=".jsonl", encoding="utf-8"
        )
        # Check if open was called correctly by the function (it uses the temp file handle)
        handle = mock_temp_file.__enter__.return_value
        calls = [call(json.dumps(req) + "\n") for req in sample_requests]
        handle.write.assert_has_calls(calls)


def test_format_requests_to_jsonl_empty():
    """Test formatting with an empty request list."""
    m = mock_open()
    mock_temp_file = MagicMock()
    mock_temp_file.__enter__.return_value.name = "/tmp/empty.jsonl"
    with (
        patch("builtins.open", m),
        patch(
            "tempfile.NamedTemporaryFile", return_value=mock_temp_file
        ) as mock_tempfile_constructor,
    ):
        mock_tempfile_constructor.return_value.__enter__.return_value.name = (
            "/tmp/empty.jsonl"
        )

        file_path = format_requests_to_jsonl([])
        assert file_path == "/tmp/empty.jsonl"
        mock_tempfile_constructor.assert_called_once_with(
            mode="w", delete=False, suffix=".jsonl", encoding="utf-8"
        )
        handle = mock_temp_file.__enter__.return_value
        handle.write.assert_not_called()  # No lines should be written


# 2. Test upload_batch_file
def test_upload_batch_file_success(mock_openai_client):
    """Test successful file upload."""
    file_path = "/path/to/batch_requests.jsonl"
    with patch("os.path.exists", return_value=True):  # Mock file existence
        file_info = upload_batch_file(mock_openai_client, file_path)
        mock_openai_client.upload_file.assert_called_once_with(
            file_path=file_path, purpose="batch"
        )
        assert file_info == {"id": "file-123", "status": "uploaded"}


def test_upload_batch_file_not_found(mock_openai_client):
    """Test upload when file does not exist."""
    file_path = "/path/to/nonexistent.jsonl"
    with patch("os.path.exists", return_value=False):  # Mock file non-existence
        with pytest.raises(FileNotFoundError, match="Batch request file not found"):
            upload_batch_file(mock_openai_client, file_path)
        mock_openai_client.upload_file.assert_not_called()


def test_upload_batch_file_api_error(mock_openai_client):
    """Test upload when OpenAI API call fails."""
    mock_openai_client.upload_file.side_effect = Exception("API Error")
    file_path = "/path/to/batch_requests.jsonl"
    with patch("os.path.exists", return_value=True):
        with pytest.raises(Exception, match="API Error"):
            upload_batch_file(mock_openai_client, file_path)
        mock_openai_client.upload_file.assert_called_once_with(
            file_path=file_path, purpose="batch"
        )


# 3. Test create_batch_job
def test_create_batch_job_success(mock_openai_client):
    """Test successful batch job creation."""
    input_file_id = "file-123"
    endpoint = "/v1/chat/completions"
    completion_window = "24h"

    batch_info = create_batch_job(
        mock_openai_client, input_file_id, endpoint, completion_window
    )

    mock_openai_client.create_batch.assert_called_once_with(
        input_file_id=input_file_id,
        endpoint=endpoint,
        completion_window=completion_window,
    )
    assert batch_info == {"id": "batch-abc", "status": "validating"}


def test_create_batch_job_api_error(mock_openai_client):
    """Test batch creation when OpenAI API call fails."""
    mock_openai_client.create_batch.side_effect = Exception("Batch Creation Failed")
    input_file_id = "file-123"
    endpoint = "/v1/chat/completions"
    completion_window = "24h"

    with pytest.raises(Exception, match="Batch Creation Failed"):
        create_batch_job(mock_openai_client, input_file_id, endpoint, completion_window)

    mock_openai_client.create_batch.assert_called_once_with(
        input_file_id=input_file_id,
        endpoint=endpoint,
        completion_window=completion_window,
    )


# 4. Test check_batch_status
def test_check_batch_status_success(mock_openai_client):
    """Test successful batch status check."""
    batch_id = "batch-abc"
    status_info = check_batch_status(mock_openai_client, batch_id)
    mock_openai_client.retrieve_batch.assert_called_once_with(batch_id=batch_id)
    assert status_info == {
        "id": "batch-abc",
        "status": "completed",
        "output_file_id": "file-456",
        "error_file_id": None,
    }


def test_check_batch_status_api_error(mock_openai_client):
    """Test status check when OpenAI API call fails."""
    mock_openai_client.retrieve_batch.side_effect = Exception("Status Check Failed")
    batch_id = "batch-abc"
    with pytest.raises(Exception, match="Status Check Failed"):
        check_batch_status(mock_openai_client, batch_id)
    mock_openai_client.retrieve_batch.assert_called_once_with(batch_id=batch_id)


# 5. Test download_batch_result_file
@patch("os.makedirs")
@patch("os.path.exists")
def test_download_batch_result_file_success(
    mock_exists, mock_makedirs, mock_openai_client
):
    """Test successful download of batch result file."""
    mock_exists.return_value = False  # Simulate directory doesn't exist initially
    batch_id = "batch-abc"
    output_file_id = "file-456"
    download_dir = "/path/to/results"
    expected_download_path = os.path.join(download_dir, f"{batch_id}_results.jsonl")

    # Create a temporary directory for the mock download to write into
    with tempfile.TemporaryDirectory() as temp_dir:
        # Adjust download_dir and expected_download_path to use the temp_dir
        # This ensures the mock download writes to a real, temporary location
        actual_download_dir = os.path.join(temp_dir, "results")
        actual_expected_path = os.path.join(
            actual_download_dir, f"{batch_id}_results.jsonl"
        )

        # Re-configure the mock download side effect to use the correct temp path
        def mock_download_temp(url, dest_path):
            # Ensure the directory exists before writing the file
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "w") as f:
                f.write(
                    '{"custom_id": "req-1", "response": {"body": {"choices": [{"message": {"content": "Result 1"}}]}}}\n'
                )
                f.write(
                    '{"custom_id": "req-2", "response": {"body": {"choices": [{"message": {"content": "Result 2"}}]}}}\n'
                )
            return dest_path

        mock_openai_client.download_file_from_url.side_effect = mock_download_temp

        downloaded_path = download_batch_result_file(
            mock_openai_client, batch_id, output_file_id, actual_download_dir
        )

        # Assertions using the temporary paths
        mock_exists.assert_called_once_with(actual_download_dir)
        mock_makedirs.assert_called_once_with(actual_download_dir, exist_ok=True)
        mock_openai_client.download_file_from_url.assert_called_once()
        args, kwargs = mock_openai_client.download_file_from_url.call_args
        assert args[1] == actual_expected_path  # Check destination path
        assert downloaded_path == actual_expected_path

        # Verify content was written by the mock download function
        with open(downloaded_path, "r") as f:
            content = f.readlines()
            assert len(content) == 2
            assert "Result 1" in content[0]
            assert "Result 2" in content[1]
        # No need to manually clean up, TemporaryDirectory handles it


@patch("os.makedirs")
@patch("os.path.exists")
def test_download_batch_result_file_dir_exists(
    mock_exists, mock_makedirs, mock_openai_client
):
    """Test download when directory already exists."""
    mock_exists.return_value = True  # Simulate directory exists
    batch_id = "batch-abc"
    output_file_id = "file-456"

    with tempfile.TemporaryDirectory() as temp_dir:
        actual_download_dir = os.path.join(temp_dir, "results")
        actual_expected_path = os.path.join(
            actual_download_dir, f"{batch_id}_results.jsonl"
        )

        # Ensure the mock download uses the temp path
        def mock_download_temp(url, dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "w") as f:
                f.write("{}\n")  # Write dummy content
            return dest_path

        mock_openai_client.download_file_from_url.side_effect = mock_download_temp

        downloaded_path = download_batch_result_file(
            mock_openai_client, batch_id, output_file_id, actual_download_dir
        )

        mock_exists.assert_called_once_with(actual_download_dir)
        mock_makedirs.assert_not_called()  # Should not be called if dir exists
        mock_openai_client.download_file_from_url.assert_called_once()
        args, kwargs = mock_openai_client.download_file_from_url.call_args
        assert args[1] == actual_expected_path
        assert downloaded_path == actual_expected_path


@patch("os.makedirs")
@patch("os.path.exists")
def test_download_batch_result_file_api_error(
    mock_exists, mock_makedirs, mock_openai_client
):
    """Test download when the API download call fails."""
    mock_exists.return_value = True
    mock_openai_client.download_file_from_url.side_effect = Exception("Download Failed")
    batch_id = "batch-abc"
    output_file_id = "file-456"
    download_dir = "/path/to/results"  # Path doesn't need to be real here

    with pytest.raises(Exception, match="Download Failed"):
        download_batch_result_file(
            mock_openai_client, batch_id, output_file_id, download_dir
        )

    mock_openai_client.download_file_from_url.assert_called_once()


# 6. Test parse_batch_result_file
def test_parse_batch_result_file_success():
    """Test successful parsing of a batch result file."""
    jsonl_content = """
{"custom_id": "req-1", "response": {"status_code": 200, "body": {"id": "chatcmpl-1", "choices": [{"message": {"content": "Result 1"}}]}}}
{"custom_id": "req-2", "response": {"status_code": 200, "body": {"id": "chatcmpl-2", "choices": [{"message": {"content": "Result 2"}}]}}}
{"custom_id": "req-3", "error": {"code": "500", "message": "Server Error"}}
"""
    expected_results = {
        "req-1": {
            "status_code": 200,
            "body": {
                "id": "chatcmpl-1",
                "choices": [{"message": {"content": "Result 1"}}],
            },
        },
        "req-2": {
            "status_code": 200,
            "body": {
                "id": "chatcmpl-2",
                "choices": [{"message": {"content": "Result 2"}}],
            },
        },
        # Errors are handled separately
    }
    expected_errors = {"req-3": {"code": "500", "message": "Server Error"}}

    file_path = "/fake/results.jsonl"
    m = mock_open(read_data=jsonl_content)
    with patch("builtins.open", m):
        with patch("os.path.exists", return_value=True):
            results, errors = parse_batch_result_file(file_path)
            m.assert_called_once_with(file_path, "r", encoding="utf-8")
            assert results == expected_results
            assert errors == expected_errors


def test_parse_batch_result_file_not_found():
    """Test parsing when the result file does not exist."""
    file_path = "/fake/nonexistent_results.jsonl"
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="Batch result file not found"):
            parse_batch_result_file(file_path)


def test_parse_batch_result_file_invalid_json(caplog):
    """Test parsing with invalid JSON content."""
    jsonl_content = """
{"custom_id": "req-1", "response": {"body": {"choices": [{"message": {"content": "Result 1"}}]}}}
this is not valid json
{"custom_id": "req-2", "response": {"body": {"choices": [{"message": {"content": "Result 2"}}]}}}
"""
    file_path = "/fake/invalid_results.jsonl"
    m = mock_open(read_data=jsonl_content)
    with (
        patch("builtins.open", m),
        patch("os.path.exists", return_value=True),
        caplog.at_level(logging.ERROR),
    ):  # Capture log messages
        results, errors = parse_batch_result_file(file_path)
        assert "req-1" in results
        assert "req-2" in results
        assert len(results) == 2  # Only valid lines parsed
        assert len(errors) == 0  # No OpenAI errors reported in this case
        # Check that an error was logged for the invalid line
        assert "Failed to parse line" in caplog.text
        assert "this is not valid json" in caplog.text


def test_parse_batch_result_file_missing_custom_id(caplog):
    """Test parsing when a line is missing the custom_id."""
    jsonl_content = """
{"response": {"body": {"choices": [{"message": {"content": "Result 1"}}]}}}
{"custom_id": "req-2", "response": {"body": {"choices": [{"message": {"content": "Result 2"}}]}}}
"""
    file_path = "/fake/missing_id_results.jsonl"
    m = mock_open(read_data=jsonl_content)
    with (
        patch("builtins.open", m),
        patch("os.path.exists", return_value=True),
        caplog.at_level(logging.WARNING),
    ):  # Capture log messages
        results, errors = parse_batch_result_file(file_path)
        assert "req-2" in results
        assert len(results) == 1  # Only line with custom_id parsed
        assert len(errors) == 0
        # Check that a warning was logged
        assert "Missing 'custom_id' in line" in caplog.text


# Add import for logging at the top if not already present
