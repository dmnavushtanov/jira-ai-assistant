"""Base agent class providing shared configuration and LLM client setup."""

from __future__ import annotations

import logging
from typing import Any

from src.configs.config import load_config
from src.llm_clients import create_llm_client

logger = logging.getLogger(__name__)
logger.debug("base agent module loaded")


class BaseAgent:
    """Base agent that loads configuration and initializes an LLM client."""

    def __init__(self, config_path: str | None = None) -> None:
        logger.debug("Initializing %s with config_path=%s", self.__class__.__name__, config_path)
        self.config = load_config(config_path)
        self.client = create_llm_client(config_path)

    def ask(self, prompt: str, **kwargs: Any) -> str:
        """Return the LLM response text for ``prompt``."""
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat_completion(messages, **kwargs)
        return self.client.extract_text(response)


__all__ = ["BaseAgent"]
