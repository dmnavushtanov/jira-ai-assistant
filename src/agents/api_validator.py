"""API Validator agent for Jira AI Assistant."""

from __future__ import annotations

import logging
from typing import Any, Dict

import yaml

from src.prompts import load_prompt
from src.utils import extract_plain_text, safe_format
from src.configs.config import load_config
from src.llm_clients.openai_client import OpenAIClient
from src.llm_clients.claude_client import ClaudeClient

logger = logging.getLogger(__name__)
logger.debug("api_validator module loaded")


def _load_status_prompts() -> Dict[str, str]:
    """Return a mapping of Jira statuses to prompt templates."""
    text = load_prompt("api_validator.txt")
    if not text:
        logger.warning("api_validator.txt not found or empty")
        return {}
    try:
        prompts = yaml.safe_load(text) or {}
        if not isinstance(prompts, dict):
            logger.error("api_validator.txt must define a mapping of status to prompt")
            return {}
        logger.debug("Loaded %d status prompts", len(prompts))
        return {str(k): str(v) for k, v in prompts.items()}
    except yaml.YAMLError:
        logger.exception("Failed to parse api_validator.txt as YAML")
        return {}



class ApiValidatorAgent:
    """Agent that validates Jira issues based on their status."""

    def __init__(self, config_path: str | None = None) -> None:
        logger.debug("Initializing ApiValidatorAgent with config_path=%s", config_path)
        self.config = load_config(config_path)
        llm = self.config.base_llm.lower()
        if llm == "openai":
            logger.debug("Using OpenAIClient")
            self.client = OpenAIClient(config_path)
        elif llm in {"anthropic", "claude"}:
            logger.debug("Using ClaudeClient")
            self.client = ClaudeClient(config_path)
        else:
            logger.error("Unsupported LLM provider: %s", self.config.base_llm)
            raise ValueError(f"Unsupported LLM provider: {self.config.base_llm}")

        self.prompts = _load_status_prompts()

    def validate(self, issue: Dict[str, Any], **kwargs: Any) -> Any:
        """Validate ``issue`` according to its status using the configured LLM."""
        fields = issue.get("fields", {})
        status = fields.get("status", {}).get("name", "").replace(" ", "")
        logger.info("Validating issue %s with status %s", issue.get("key"), status)

        template = self.prompts.get(status)
        if not template:
            logger.warning("No prompt found for status %s", status)
            return ""

        values = {
            "key": issue.get("key", ""),
            "summary": extract_plain_text(fields.get("summary")),
            "description": extract_plain_text(fields.get("description")),
            "status": status,
            "body_instructions": (
                "Include full request and response bodies" if self.config.include_whole_api_body
                else "Do not include full bodies. Provide boolean fields `request_body_exists`, `request_body_valid`, `response_body_exists`, and `response_body_valid` and set the example fields to null."
            ),
        }
        try:
            prompt = safe_format(template, values)
        except Exception:
            logger.exception("Failed to format prompt")
            return ""
        logger.debug("Prompt for validation: %s", prompt)
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat_completion(messages, **kwargs)
        try:
            result = response.choices[0].message.content.strip()
        except Exception:  # pragma: no cover - handle unexpected structures
            try:
                result = response["choices"][0]["message"]["content"].strip()
            except Exception:
                logger.exception("Failed to parse response")
                result = response
        logger.info("Validation result: %s", result)
        return result


__all__ = ["ApiValidatorAgent"]


