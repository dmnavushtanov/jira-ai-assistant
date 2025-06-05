# Import src package first to ensure path setup

from openai import OpenAI
from typing import List, Dict, Any
import logging

from src.configs.config import load_config
from src.llm_clients.base_llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """Simple wrapper around the OpenAI SDK."""

    def __init__(self, config_path: str = None) -> None:
        logger.debug("Initializing OpenAIClient with config_path=%s", config_path)
        self.config = load_config(config_path)
        self.client = OpenAI(api_key=self.config.openai_api_key)

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        """Create a chat completion using the configured model."""
        logger.debug(
            "Creating chat completion with messages=%s kwargs=%s",
            messages,
            kwargs,
        )
        return self.client.chat.completions.create(
            model=self.config.openai_model,
            messages=messages,
            **kwargs,
        )


__all__ = ["OpenAIClient"]
