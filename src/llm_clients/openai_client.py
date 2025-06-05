# Import src package first to ensure path setup

from openai import OpenAI
from typing import List, Dict, Any

from src.configs.config import load_config
from src.llm_clients.base_llm_client import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """Simple wrapper around the OpenAI SDK."""

    def __init__(self, config_path: str = None) -> None:
        self.config = load_config(config_path)
        self.client = OpenAI(api_key=self.config.openai_api_key)

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        """Create a chat completion using the configured model."""
        return self.client.chat.completions.create(
            model=self.config.openai_model,
            messages=messages,
            **kwargs,
        )


__all__ = ["OpenAIClient"]
