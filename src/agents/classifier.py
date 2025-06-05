"""Simple classifier agent using a configurable LLM provider."""

from typing import List, Dict, Any
import logging

from src.configs.config import load_config
from src.llm_clients.openai_client import OpenAIClient
from src.llm_clients.claude_client import ClaudeClient
from src.services.jira_service import get_issue_by_id_tool

logger = logging.getLogger(__name__)
logger.debug("classifier module loaded")


class ClassifierAgent:
    """Agent that selects an LLM client based on configuration."""

    def __init__(self, config_path: str | None = None) -> None:
        logger.debug("Initializing ClassifierAgent with config_path=%s", config_path)
        self.config = load_config(config_path)
        llm = self.config.base_llm.lower()
        logger.info("Loaded configuration for base LLM: %s", llm)
        if llm == "openai":
            logger.debug("Using OpenAIClient")
            self.client = OpenAIClient(config_path)
        elif llm in {"anthropic", "claude"}:
            logger.debug("Using ClaudeClient")
            self.client = ClaudeClient(config_path)
        else:
            logger.error("Unsupported LLM provider: %s", self.config.base_llm)
            raise ValueError(f"Unsupported LLM provider: {self.config.base_llm}")

        # Tools available to this agent
        self.tools = [get_issue_by_id_tool]

    def classify(self, prompt: str, **kwargs: Any) -> Any:
        """Return the classification result for ``prompt``."""
        logger.info("Classifying prompt")
        logger.debug("Prompt: %s kwargs=%s", prompt, kwargs)
        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
        response = self.client.chat_completion(messages, **kwargs)
        logger.debug("Raw response: %s", response)
        try:
            result = response.choices[0].message.content.strip()
        except Exception:
            try:
                result = response["choices"][0]["message"]["content"].strip()
            except Exception:
                logger.exception("Failed to parse response")
                result = response
        logger.info("Classification result: %s", result)
        return result


__all__ = ["ClassifierAgent"]

