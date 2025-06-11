"""Factory for LangChain LLMs based on configuration."""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.configs.config import load_config

logger = logging.getLogger(__name__)


def create_langchain_llm(config_path: Optional[str] = None) -> Any:
    """Return a LangChain compatible LLM based on ``base_llm`` setting."""
    config = load_config(config_path)
    provider = config.base_llm.lower()
    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI  # type: ignore

            return ChatOpenAI(model=config.openai_model, api_key=config.openai_api_key)
        if provider in {"anthropic", "claude"}:
            try:
                from langchain_anthropic import ChatAnthropic  # type: ignore
            except Exception:  # pragma: no cover - optional dependency
                logger.warning("langchain-anthropic not installed")
                return None
            return ChatAnthropic(model=config.anthropic_model, api_key=config.anthropic_api_key)
    except Exception:  # pragma: no cover - graceful degradation
        logger.exception("Failed to initialize LangChain LLM")
        return None
    logger.error("Unsupported base LLM: %s", provider)
    return None


__all__ = ["create_langchain_llm"]
