# CogniBench - OpenAI LLM Client Implementation
# Version: 1.0

import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from .base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """
    Concrete implementation of BaseLLMClient for interacting with OpenAI models.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the OpenAI client.

        Args:
            api_key: OpenAI API key. If None, attempts to read from
                     the OPENAI_API_KEY environment variable.

        Raises:
            ValueError: If the API key is not provided and not found in env vars after attempting to load .env.
            OpenAIError: If there's an issue initializing the underlying client.
        """
        # Load environment variables from .env file if it exists
        load_dotenv()

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "OpenAI API key not provided, not found in environment variables (OPENAI_API_KEY), and not found in .env file."
            )
        try:
            self.client = OpenAI(api_key=resolved_api_key)
            print("OpenAI client initialized successfully.")
        except OpenAIError as e:
            print(f"Error initializing OpenAI client: {e}")
            raise  # Re-raise the exception after logging

    def invoke(
        self,
        prompt: str,
        model_name: str,
        system_prompt: Optional[
            str
        ] = "You are a helpful assistant.",  # Default system prompt
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Invokes the specified OpenAI language model.

        Args:
            prompt: The main user prompt/query for the LLM.
            model_name: The specific OpenAI model identifier (e.g., "gpt-4o").
            system_prompt: An optional system message to guide the LLM's behavior.
            temperature: The sampling temperature for generation (0 for deterministic).
            **kwargs: Additional arguments for the OpenAI API call
                      (e.g., max_tokens, response_format).

        Returns:
            A dictionary containing the LLM's response or an error message.
            Example success: {"raw_content": "The LLM response text..."}
            Example error: {"error": "API connection failed..."}
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            print(f"--- Invoking OpenAI model: {model_name} ---")  # Basic logging
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                **kwargs,  # Pass through other arguments like max_tokens, response_format
            )
            print("--- OpenAI Invocation Complete ---")

            raw_response_content = response.choices[0].message.content
            # You could add more metadata from the response if needed
            return {"raw_content": raw_response_content}

        except OpenAIError as e:
            print(f"Error during OpenAI API call: {e}")
            return {"error": f"OpenAI API Error: {e}"}
        except Exception as e:
            print(f"An unexpected error occurred during OpenAI invocation: {e}")
            return {"error": f"Unexpected Error: {e}"}


# Example usage (for testing - requires API key):
# if __name__ == "__main__":
#     try:
#         client = OpenAIClient()
#         result = client.invoke(
#             prompt="What is 2+2?",
#             model_name="gpt-3.5-turbo" # Use a cheaper model for testing
#         )
#         if "error" in result:
#             print(f"Error: {result['error']}")
#         else:
#             print(f"Response: {result['raw_content']}")
#     except (ValueError, OpenAIError) as e:
#         print(f"Failed to initialize or invoke client: {e}")
