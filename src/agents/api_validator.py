"""API Validator agent for Jira AI Assistant."""

from __future__ import annotations

import logging
from typing import Any, Dict

from pathlib import Path

from src.prompts import load_prompt, PROMPTS_DIR
from src.utils import extract_plain_text, safe_format
from src.configs.config import load_config
from src.llm_clients import create_llm_client

logger = logging.getLogger(__name__)
logger.debug("api_validator module loaded")


def _load_status_prompts(directory: str) -> Dict[str, str]:
    """Return a mapping of Jira statuses to prompt templates."""
    prompts: Dict[str, str] = {}
    folder = PROMPTS_DIR / directory
    if not folder.exists():
        logger.warning("Validation prompts directory %s not found", folder)
        return prompts
    for file in folder.glob("validate_*.txt"):
        status = file.stem.replace("validate_", "").lower()
        try:
            prompts[status] = file.read_text(encoding="utf-8")
            logger.debug("Loaded validation prompt for %s", status)
        except Exception:
            logger.exception("Failed to read prompt file %s", file)
    return prompts



class ApiValidatorAgent:
    """Agent that validates Jira issues based on their status."""

    def __init__(self, config_path: str | None = None) -> None:
        logger.debug("Initializing ApiValidatorAgent with config_path=%s", config_path)
        self.config = load_config(config_path)
        self.client = create_llm_client(config_path)

        self.prompts = _load_status_prompts(self.config.validation_prompts_dir)
        self.general_prompt = load_prompt(
            str(Path(self.config.validation_prompts_dir) / "general.txt")
        )

    def validate(self, issue: Dict[str, Any], **kwargs: Any) -> Any:
        """Validate ``issue`` according to its status using the configured LLM."""
        fields = issue.get("fields", {})
        status = fields.get("status", {}).get("name", "").replace(" ", "").lower()
        logger.info("Validating issue %s with status %s", issue.get("key"), status)

        template = self.prompts.get(status)
        if not template:
            logger.warning("No prompt found for status %s", status)
            template = self.general_prompt
            if not template:
                return ""
        elif self.general_prompt:
            template = f"{self.general_prompt}\n{template}"

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


