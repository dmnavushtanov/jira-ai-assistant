"""Agent for generating test cases and exercising APIs."""

from __future__ import annotations

import logging
from typing import Any

from src.configs.config import load_config
from src.llm_clients import create_llm_client
from src.prompts import load_prompt
from src.utils import safe_format

logger = logging.getLogger(__name__)
logger.debug("test_agent module loaded")


class TestAgent:
    """Agent that generates test cases from validation results."""

    def __init__(self, config_path: str | None = None) -> None:
        logger.debug("Initializing TestAgent with config_path=%s", config_path)
        self.config = load_config(config_path)
        self.client = create_llm_client(config_path)
        self.prompt_template = load_prompt("tests/testCasesGeneration.txt")

    def create_test_cases(self, validation_result: str, **kwargs: Any) -> str:
        """Return test cases based on ``validation_result``."""
        template = self.prompt_template or (
            "Generate test cases based on the following validation summary:\n{summary}"
        )
        prompt = safe_format(template, {"summary": validation_result})
        logger.info("Generating test cases from validation result")
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat_completion(messages, **kwargs)
        try:
            result = response.choices[0].message.content.strip()
        except Exception:
            try:
                result = response["choices"][0]["message"]["content"].strip()
            except Exception:  # pragma: no cover - handle unexpected structure
                logger.exception("Failed to parse response")
                result = str(response)
        logger.debug("Generated test cases: %s", result)
        return result


__all__ = ["TestAgent"]
