"""Agent for planning and creating Jira issues of any type."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from src.agents.jira_operations import JiraOperationsAgent
from src.configs.config import load_config
from src.llm_clients import create_llm_client
from src.prompts import load_prompt
from src.models import SharedContext
from src.utils import safe_format, parse_json_block, JiraContextMemory

logger = logging.getLogger(__name__)
logger.debug("issue_creator module loaded")


class IssueCreatorAgent:
    """Agent that extracts issue details and creates Jira tickets."""

    def __init__(
        self,
        config_path: str | None = None,
        memory: Optional[JiraContextMemory] = None,
        context: Optional[SharedContext] = None,
    ) -> None:
        logger.debug(
            "Initializing IssueCreatorAgent with config_path=%s", config_path
        )
        self.config = load_config(config_path)
        self.client = create_llm_client(config_path)
        self.memory = memory
        self.context = context
        self.operations = JiraOperationsAgent(
            config_path, memory=memory, context=context
        )
        self.plan_prompt = load_prompt("issue_plan.txt")

    def plan_issue(self, request: str, history: str = "", **kwargs: Any) -> dict[str, Any]:
        """Return structured issue info extracted from ``request``."""
        if not self.plan_prompt:
            raise RuntimeError("Issue planning prompt not found")
        template = self.plan_prompt
        prompt = safe_format(template, {"request": request})
        if history:
            prompt = f"Previous conversation:\n{history}\n\nCurrent request:\n{prompt}"
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat_completion(messages, **kwargs)
        text = self.client.extract_text(response)
        plan = parse_json_block(text)
        if isinstance(plan, dict):
            return plan
        logger.debug("Plan not JSON: %s", text)
        return {}

    def create_issue(
        self, request: str, project_key: str, history: str = "", **kwargs: Any
    ) -> str:
        """Create a Jira issue using details from ``request``."""
        plan = self.plan_issue(request, history=history, **kwargs)
        summary = plan.get("summary") or "New Issue"
        description = plan.get("description", "")
        issue_type = str(plan.get("issue_type", "Task"))
        issue_type_norm = issue_type.replace(" ", "-").title()
        if issue_type_norm.lower() in {"sub-task", "subtask"}:
            issue_type_norm = "Sub-task"
        parent = plan.get("parent")
        if issue_type_norm == "Sub-task" and not parent:
            return "Sure, I can create a sub-task. Which parent issue should it be under?"
        result = self.operations.create_issue(
            summary,
            description,
            project_key,
            issue_type_norm,
            parent_key=parent,
            **kwargs,
        )
        return json.dumps(result)


__all__ = ["IssueCreatorAgent"]
