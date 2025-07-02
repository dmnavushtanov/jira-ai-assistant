"""Simple classifier agent using a configurable LLM provider."""

from typing import List, Dict, Any, Optional
import logging

from src.configs.config import load_config
from src.llm_clients import create_llm_client
from src.services.jira_service import get_issue_by_id_tool
from src.utils import JiraContextMemory

logger = logging.getLogger(__name__)
logger.debug("classifier module loaded")


class ClassifierAgent:
    """Agent that selects an LLM client based on configuration."""

    def __init__(
        self,
        config_path: str | None = None,
        memory: Optional[JiraContextMemory] = None,
    ) -> None:
        logger.debug(
            "Initializing ClassifierAgent with config_path=%s", config_path
        )
        self.config = load_config(config_path)
        self.client = create_llm_client(config_path)
        self.memory = memory

        # Tools available to this agent
        self.tools = [get_issue_by_id_tool]

    def classify(self, prompt: str, **kwargs: Any) -> Any:
        """Return the classification result for ``prompt``."""
        logger.info("Classifying prompt")
        logger.debug("Prompt: %s kwargs=%s", prompt, kwargs)
        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
        response = self.client.chat_completion(messages, **kwargs)
        logger.debug("Raw response: %s", response)
        result = self.client.extract_text(response)
        logger.info("Classification result: %s", result)
        return result


__all__ = ["ClassifierAgent"]

