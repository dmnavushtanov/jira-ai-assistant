"""Agent for generating multi-step plans from user requests."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from src.configs.config import load_config
from src.llm_clients import create_llm_client
from src.prompts import load_prompt
from src.utils import safe_format, parse_json_block, JiraContextMemory
from src.models import SharedContext

logger = logging.getLogger(__name__)
logger.debug("planning module loaded")


class PlanningAgent:
    """Generate a structured execution plan for Jira operations."""

    def __init__(
        self,
        config_path: str | None = None,
        memory: Optional[JiraContextMemory] = None,
        context: Optional[SharedContext] = None,
    ) -> None:
        logger.debug(
            "Initializing PlanningAgent with config_path=%s", config_path
        )
        self.config = load_config(config_path)
        self.client = create_llm_client(config_path)
        self.memory = memory
        self.context = context
        self.plan_prompt = load_prompt("operations_plan.txt")

    def generate_plan(
        self,
        user_request: str,
        issue_context: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Return a plan describing the steps to satisfy ``user_request``."""
        if not self.plan_prompt:
            raise RuntimeError("Plan prompt not found")

        context_json = json.dumps(issue_context or {}, indent=2)
        prompt = safe_format(
            self.plan_prompt,
            {"user_request": user_request, "issue_context": context_json},
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat_completion(messages, **kwargs)
        text = self.client.extract_text(response)
        plan = parse_json_block(text)
        if isinstance(plan, dict) and isinstance(plan.get("plan"), list):
            logger.debug("Generated plan: %s", plan)
            return plan
        logger.warning("Invalid plan structure received: %s", text)
        return {"plan": []}


__all__ = ["PlanningAgent"]
