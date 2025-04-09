# CogniBench - Base LLM Client Interface
# Version: 1.0

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseLLMClient(ABC):
    """
    Abstract Base Class for LLM client implementations.
    Defines the standard interface for invoking language models.
    """

    @abstractmethod
    def invoke(
        self,
        prompt: str,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Invokes the specified language model with the given prompt and parameters.

        Args:
            prompt: The main user prompt/query for the LLM.
            model_name: The specific model identifier to use (e.g., "gpt-4o").
            system_prompt: An optional system message to guide the LLM's behavior.
            temperature: The sampling temperature for generation (0 for deterministic).
            **kwargs: Additional provider-specific arguments (e.g., max_tokens, response_format).

        Returns:
            A dictionary containing the LLM's response. Minimally, it should include
            'raw_content' with the text response. It can also include 'error'
            if an issue occurred, or other metadata.
            Example success: {"raw_content": "The LLM response text..."}
            Example error: {"error": "API connection failed..."}
        """
        pass