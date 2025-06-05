"""Placeholder client for Anthropic Claude."""

from typing import List, Dict, Any
import logging

from src.configs.config import load_config
from src.llm_clients.base_llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class ClaudeClient(BaseLLMClient):
    """Stub implementation for Anthropic's Claude API."""

    def __init__(self, config_path: str | None = None) -> None:
        logger.debug("Initializing ClaudeClient with config_path=%s", config_path)
        self.config = load_config(config_path)
        # Actual client initialization would go here.

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        """Create a chat completion using Claude."""
        logger.debug(
            "Creating Claude chat completion with messages=%s kwargs=%s",
            messages,
            kwargs,
        )
        raise NotImplementedError("Claude client not implemented")


__all__ = ["ClaudeClient"]

