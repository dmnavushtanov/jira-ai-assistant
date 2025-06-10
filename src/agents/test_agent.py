"""Agent for generating test cases and exercising APIs."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import json
import re

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
        self.prompts = {
            "GET": load_prompt("tests/get_test_cases.txt"),
            "POST": load_prompt("tests/post_test_cases.txt"),
            "PUT": load_prompt("tests/put_test_cases.txt"),
            "DELETE": load_prompt("tests/delete_test_cases.txt"),
        }
        self.default_prompt = load_prompt("tests/testCasesGeneration.txt")

    def _extract_method(self, validation_result: str) -> Optional[str]:
        """Return HTTP method found in ``validation_result`` if any."""
        try:
            data: Dict[str, Any] = json.loads(validation_result)
        except Exception:
            from src.utils import parse_json_block

            data = parse_json_block(validation_result) or {}
        if isinstance(data, dict):
            parsed = data.get("parsed") or {}
            method = parsed.get("method")
            if method:
                return str(method).upper()
        match = re.search(r"\b(GET|POST|PUT|DELETE)\b", validation_result, re.I)
        return match.group(1).upper() if match else None

    def create_test_cases(
        self, validation_result: str, method: Optional[str] = None, **kwargs: Any
    ) -> str:
        """Return test cases based on ``validation_result`` and HTTP ``method``."""
        method = (method or self._extract_method(validation_result) or "GET").upper()
        template = self.prompts.get(method) or self.default_prompt or (
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
