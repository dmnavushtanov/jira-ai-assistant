"""Placeholder client for Anthropic Claude."""

from typing import List, Dict, Any

from src.configs.config import load_config
from src.llm_clients.base_llm_client import BaseLLMClient


class ClaudeClient(BaseLLMClient):
    """Stub implementation for Anthropic's Claude API."""

    def __init__(self, config_path: str | None = None) -> None:
        self.config = load_config(config_path)
        # Actual client initialization would go here.

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        """Create a chat completion using Claude."""
        raise NotImplementedError("Claude client not implemented")


__all__ = ["ClaudeClient"]

