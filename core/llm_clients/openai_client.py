"""
CogniBench OpenAI LLM Client Implementation.

Provides a concrete implementation of the BaseLLMClient for interacting with
OpenAI models (GPT-3.5, GPT-4, etc.). Includes persistent caching using `shelve`.

Version: 1.0.1
"""

import asyncio
import atexit
import hashlib
import json
import logging
import os
import shelve
import time  # Added for retry delay
from pathlib import Path
from typing import Any, Dict, Optional, Union

import httpx  # Added for batch file download

# Use ABSOLUTE import for base class
from core.llm_clients.base import BaseLLMClient
from dotenv import load_dotenv
from openai import AsyncOpenAI  # Use AsyncOpenAI
from openai import APIConnectionError, OpenAIError, RateLimitError
from openai.types.batch import Batch  # Import Batch type

# Get logger for this module
logger = logging.getLogger("backend")

# --- Cache Setup ---
CACHE_DIR = Path.home() / ".cognibench_cache"
CACHE_FILENAME = CACHE_DIR / "openai_cache"
_cache = None
_cache_dirty = False


def _init_cache():
    global _cache
    if _cache is None:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            _cache = shelve.open(str(CACHE_FILENAME))
            logger.info(f"Opened OpenAI cache file: {CACHE_FILENAME}")
        except Exception as e:
            logger.error(
                f"Failed to open or create cache file {CACHE_FILENAME}: {e}",
                exc_info=True,
            )
            _cache = {}  # Use in-memory dict as fallback


def _close_cache():
    global _cache, _cache_dirty
    if _cache is not None:
        try:
            if _cache_dirty:
                logger.info(f"Syncing and closing OpenAI cache: {CACHE_FILENAME}")
                _cache.sync()  # Ensure writes are flushed
            else:
                logger.info(
                    f"Closing OpenAI cache (no changes detected): {CACHE_FILENAME}"
                )
            _cache.close()
        except Exception as e:
            logger.error(
                f"Error closing cache file {CACHE_FILENAME}: {e}", exc_info=True
            )
        finally:
            _cache = None  # Reset cache variable


# Initialize cache on import and register cleanup
_init_cache()
atexit.register(_close_cache)


def clear_openai_cache():
    """Clears the persistent OpenAI cache."""
    global _cache, _cache_dirty
    logger.warning("Clearing OpenAI cache...")
    if _cache is not None:
        _cache.clear()
        _cache_dirty = True  # Mark as dirty after clearing
        logger.info("OpenAI cache cleared.")
    else:
        logger.warning("Cache not initialized, cannot clear.")


# --- OpenAI Client Class ---


class OpenAIClient(BaseLLMClient):
    """
    LLM Client for interacting with OpenAI models.
    Implements caching and basic retry logic.
    Uses AsyncOpenAI for asynchronous operations.
    """

    def __init__(self, config: Optional[Any] = None, api_key: Optional[str] = None):
        """
        Initializes the AsyncOpenAI client.
        API key is loaded from environment variables (OPENAI_API_KEY) by default,
        but can be overridden by the api_key parameter or config.
        """
        load_dotenv()  # Load .env file if present
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        # TODO: Integrate config object properly if passed
        # if config and hasattr(config, 'llm_client') and hasattr(config.llm_client, 'api_key'):
        #     resolved_api_key = config.llm_client.api_key or resolved_api_key

        if not resolved_api_key:
            logger.warning(
                "OPENAI_API_KEY not found in environment variables or passed directly. "
                "Client initialization might fail if key is required."
            )
            # raise ValueError("OpenAI API key is required.") # Or allow initialization

        try:
            # Use AsyncOpenAI for async methods
            self.client = AsyncOpenAI(api_key=resolved_api_key)
            logger.info("AsyncOpenAI client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize AsyncOpenAI client: {e}", exc_info=True)
            raise

    def _generate_cache_key(self, **kwargs) -> str:
        """Generates a consistent cache key from request parameters."""
        # Create a stable string representation (e.g., sorted JSON)
        try:
            canonical_string = json.dumps(kwargs, sort_keys=True)
            return hashlib.sha256(canonical_string.encode("utf-8")).hexdigest()
        except TypeError as e:
            logger.warning(
                f"Could not generate cache key due to non-serializable args: {e}"
            )
            # Fallback: Use a hash of the repr, less reliable but avoids crashing
            return hashlib.sha256(repr(kwargs).encode("utf-8")).hexdigest()

    async def invoke(
        self,
        prompt: str,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        use_cache: bool = True,
        **kwargs,  # Allow additional parameters like stop sequences etc.
    ) -> Dict[str, Any]:
        """
        Invokes the specified OpenAI model asynchronously.

        Args:
            prompt: The main user prompt.
            model_name: The OpenAI model identifier (e.g., "gpt-4o").
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system_prompt: Optional system message.
            use_cache: Whether to use the persistent cache.
            **kwargs: Additional arguments for the OpenAI API call.

        Returns:
            A dictionary containing 'raw_content' or 'error'.
        """
        global _cache, _cache_dirty
        request_params = {
            "model": model_name,
            "prompt": prompt,  # Include prompt for caching even if messages used
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
            **kwargs,
        }
        cache_key = self._generate_cache_key(**request_params)

        if use_cache and _cache is not None and cache_key in _cache:
            logger.info(
                f"CACHE HIT: Returning cached response for key {cache_key[:8]}..."
            )
            # Ensure cached data structure matches expected return format
            cached_result = _cache[cache_key]
            if isinstance(cached_result, dict) and (
                "raw_content" in cached_result or "error" in cached_result
            ):
                return cached_result
            else:
                logger.warning(
                    f"Invalid data found in cache for key {cache_key[:8]}. Fetching fresh response."
                )
                # Optionally delete invalid cache entry: del _cache[cache_key]; _cache_dirty = True

        logger.info(
            f"CACHE MISS: Invoking model {model_name} for key {cache_key[:8]}..."
        )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        retries = 3
        delay = 1  # Initial delay in seconds

        for attempt in range(retries):
            try:
                response = await self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                raw_content = response.choices[0].message.content
                result = {"raw_content": raw_content}

                if use_cache and _cache is not None:
                    try:
                        _cache[cache_key] = result
                        _cache_dirty = True  # Mark cache as dirty
                        logger.info(
                            f"CACHE STORE: Stored response for key {cache_key[:8]}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to store response in cache for key {cache_key[:8]}: {e}",
                            exc_info=True,
                        )

                return result

            except (APIConnectionError, RateLimitError) as e:
                logger.warning(
                    f"API Error invoking model {model_name} (Attempt {attempt + 1}/{retries}): {type(e).__name__}. Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
            except OpenAIError as e:
                logger.error(
                    f"OpenAI API Error invoking model {model_name}: {e}", exc_info=True
                )
                return {"error": f"OpenAI API Error: {e}"}
            except Exception as e:
                logger.error(
                    f"Unexpected error invoking model {model_name}: {e}", exc_info=True
                )
                return {"error": f"Unexpected error: {e}"}

        # If loop finishes without success
        logger.error(f"Failed to invoke model {model_name} after {retries} attempts.")
        return {"error": f"Failed after {retries} retries."}

    # --- Batch API Helper Methods ---

    async def upload_file(
        self, file_path: str, purpose: str = "batch"
    ) -> Optional[str]:
        """Uploads a file to OpenAI for batch processing."""
        logger.info(f"Uploading file {file_path} for purpose '{purpose}'...")
        try:
            with open(file_path, "rb") as f:
                response = await self.client.files.create(file=f, purpose=purpose)
            logger.info(f"File uploaded successfully. File ID: {response.id}")
            return response.id
        except FileNotFoundError:
            logger.error(f"File not found for upload: {file_path}")
            return None
        except OpenAIError as e:
            logger.error(
                f"OpenAI API error uploading file {file_path}: {e}", exc_info=True
            )
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error uploading file {file_path}: {e}", exc_info=True
            )
            return None

    async def create_batch(
        self,
        file_id: str,
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h",
        metadata: Optional[Dict[str, str]] = None,
    ) -> Optional[Batch]:  # Return Batch object
        """Creates a batch job."""
        logger.info(f"Creating batch job with file ID: {file_id}, endpoint: {endpoint}")
        try:
            batch_job = await self.client.batches.create(
                input_file_id=file_id,
                endpoint=endpoint,
                completion_window=completion_window,
                metadata=metadata,
            )
            logger.info(f"Batch job created successfully. Batch ID: {batch_job.id}")
            return batch_job
        except OpenAIError as e:
            logger.error(
                f"OpenAI API error creating batch job for file {file_id}: {e}",
                exc_info=True,
            )
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error creating batch job for file {file_id}: {e}",
                exc_info=True,
            )
            return None

    async def retrieve_batch(
        self, batch_id: str
    ) -> Optional[Batch]:  # Return Batch object
        """Retrieves the status and details of a batch job."""
        logger.debug(f"Retrieving batch job status for ID: {batch_id}")
        try:
            batch_job = await self.client.batches.retrieve(batch_id)
            logger.debug(
                f"Successfully retrieved batch job {batch_id}. Status: {batch_job.status}"
            )
            return batch_job
        except OpenAIError as e:
            logger.error(
                f"OpenAI API error retrieving batch job {batch_id}: {e}", exc_info=True
            )
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error retrieving batch job {batch_id}: {e}", exc_info=True
            )
            return None

    async def get_file_content(
        self, file_id: str
    ) -> Optional[Any]:  # Return type depends on API/library
        """Retrieves the content object for a file (e.g., batch results)."""
        logger.debug(f"Retrieving content metadata for file ID: {file_id}")
        try:
            # Note: client.files.content() retrieves the file content itself,
            # not metadata with a download URL in recent versions.
            # The download logic needs to handle this.
            # This method might just return the raw bytes or a stream.
            # Let's return the raw response object for now.
            file_content_response = await self.client.files.content(file_id)
            logger.debug(f"Successfully retrieved content object for file {file_id}")
            return file_content_response  # Might be bytes, HttpxResponse etc.
        except OpenAIError as e:
            logger.error(
                f"OpenAI API error retrieving content for file {file_id}: {e}",
                exc_info=True,
            )
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error retrieving content for file {file_id}: {e}",
                exc_info=True,
            )
            return None

    async def download_file_content(self, file_id: str, destination_path: str) -> None:
        """Downloads file content from OpenAI and saves it."""
        logger.info(f"Downloading content for file ID: {file_id} to {destination_path}")
        try:
            # Use client.files.content() which returns file content directly
            response_content = await self.client.files.content(file_id)

            # Assuming response_content is bytes or can be read as bytes
            # Adjust based on actual return type if necessary
            content_bytes = None
            if isinstance(response_content, bytes):
                content_bytes = response_content
            elif hasattr(
                response_content, "read"
            ):  # Handle streaming response if applicable
                content_bytes = await response_content.read()  # Assuming async read
            elif hasattr(
                response_content, "content"
            ):  # Handle httpx response like object
                content_bytes = response_content.content
            else:
                raise TypeError(
                    f"Unexpected content type received from files.content: {type(response_content)}"
                )

            with open(destination_path, "wb") as f:
                f.write(content_bytes)
            logger.info(f"Successfully saved file content to {destination_path}")

        except OpenAIError as e:
            logger.error(
                f"OpenAI API error downloading content for file {file_id}: {e}",
                exc_info=True,
            )
            raise  # Re-raise to indicate failure to the caller
        except Exception as e:
            logger.error(
                f"Unexpected error downloading content for file {file_id}: {e}",
                exc_info=True,
            )
            raise  # Re-raise to indicate failure

            # Note: download_file_from_url is removed as client.files.content is used directly.
            # If a direct URL download is ever needed, httpx can be used as shown previously.
            logger.error(
                f"Unexpected error downloading content for file {file_id}: {e}",
                exc_info=True,
            )
            raise  # Re-raise to indicate failure

    # Note: download_file_from_url is removed as client.files.content is used directly.
    # If a direct URL download is ever needed, httpx can be used as shown previously.
