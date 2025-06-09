"""LLM client implementations."""

import logging

from typing import Optional

from .openai_client import OpenAIClient
from .claude_client import ClaudeClient
from .base_llm_client import BaseLLMClient
from src.configs.config import load_config

logger = logging.getLogger(__name__)
logger.debug("llm_clients package initialized")

def create_llm_client(config_path: Optional[str] = None) -> BaseLLMClient:
    """Return an LLM client based on the configured provider."""
    config = load_config(config_path)
    llm = config.base_llm.lower()
    logger.info("Selected base LLM: %s", llm)
    if llm == "openai":
        logger.debug("Creating OpenAIClient")
        return OpenAIClient(config_path)
    if llm in {"anthropic", "claude"}:
        logger.debug("Creating ClaudeClient")
        return ClaudeClient(config_path)
    logger.error("Unsupported LLM provider: %s", config.base_llm)
    raise ValueError(f"Unsupported LLM provider: {config.base_llm}")


__all__ = ["OpenAIClient", "ClaudeClient", "create_llm_client"]

