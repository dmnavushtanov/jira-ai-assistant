# Import src package first to ensure path setup

from openai import OpenAI
from typing import List, Dict, Any
import logging
import os

try:
    from langfuse import Langfuse
except Exception:  # pragma: no cover - optional dependency
    Langfuse = None  # type: ignore

from src.configs.config import load_config
from src.llm_clients.base_llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """Simple wrapper around the OpenAI SDK."""

    def __init__(self, config_path: str = None) -> None:
        logger.debug("Initializing OpenAIClient with config_path=%s", config_path)
        self.config = load_config(config_path)
        self.client = OpenAI(api_key=self.config.openai_api_key)
        self.langfuse = None
        if Langfuse is not None:
            public = os.getenv("LANGFUSE_PUBLIC_KEY")
            secret = os.getenv("LANGFUSE_SECRET_KEY")
            host = os.getenv("LANGFUSE_HOST")
            if public and secret:
                try:
                    self.langfuse = Langfuse(
                        public_key=public,
                        secret_key=secret,
                        host=host,
                    )
                    logger.info("Langfuse monitoring enabled")
                except Exception:  # pragma: no cover - fail gracefully
                    logger.exception("Failed to initialize Langfuse")

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        """Create a chat completion using the configured model."""
        logger.debug(
            "Creating chat completion with messages=%s kwargs=%s",
            messages,
            kwargs,
        )
        trace = None
        if self.langfuse:
            try:
                trace = self.langfuse.trace(name="openai.chat_completion")
            except Exception:  # pragma: no cover - monitoring optional
                logger.exception("Failed to start Langfuse trace")
                trace = None
        response = self.client.chat.completions.create(
            model=self.config.openai_model,
            messages=messages,
            **kwargs,
        )
        if trace:
            try:
                prompt = messages[-1].get("content", "") if messages else ""
                completion = (
                    response.choices[0].message.content.strip()
                    if hasattr(response, "choices")
                    else ""
                )
                trace.generation(
                    input=prompt,
                    output=completion,
                    model=self.config.openai_model,
                )
            except Exception:  # pragma: no cover
                logger.exception("Failed to log Langfuse generation")
            finally:
                try:
                    trace.end()
                except Exception:  # pragma: no cover
                    logger.exception("Failed to end Langfuse trace")
        return response


__all__ = ["OpenAIClient"]
