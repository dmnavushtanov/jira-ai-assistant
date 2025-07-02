"""Agent providing insights about Jira issues."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional
from datetime import datetime

from src.configs.config import load_config
from src.llm_clients import create_llm_client
from src.prompts import load_prompt
from src.models import SharedContext
from src.utils import safe_format, JiraContextMemory
from src.services.jira_service import (
    get_issue_by_id_tool,
    get_issue_history_tool,
    get_related_issues_tool,
    get_issue_comments_tool,
)

logger = logging.getLogger(__name__)
logger.debug("issue_insights module loaded")


class IssueInsightsAgent:
    """Agent that answers general questions about Jira issues."""

    def __init__(
        self,
        config_path: str | None = None,
        memory: Optional[JiraContextMemory] = None,
        context: Optional[SharedContext] = None,
    ) -> None:
        logger.debug(
            "Initializing IssueInsightsAgent with config_path=%s", config_path
        )
        self.config = load_config(config_path)
        self.client = create_llm_client(config_path)
        self.memory = memory
        self.context = context

        # Tools available to this agent
        self.tools = [get_issue_by_id_tool, get_issue_history_tool]

        self.summary_prompt = load_prompt("issue_summary.txt")
        self.insights_prompt = load_prompt("issue_insights.txt")

    def _summarize_related(self, issues: list[dict[str, Any]], **kwargs: Any) -> str:
        """Return summaries for a list of Jira issues."""
        parts = []
        for issue in issues:
            fields = issue.get("fields", {})
            comments = "\n".join(c.get("body", "") for c in issue.get("comments", []))
            summary = self._summarize(
                fields.get("summary", ""),
                fields.get("description", ""),
                comments,
                **kwargs,
            )
            parts.append(f"{issue.get('key')}: {summary}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _summarize(self, summary: str, description: str, comments: str = "", **kwargs: Any) -> str:
        """Return a short summary of the issue using the configured LLM.

        The ``comments`` field allows providing additional context from the
        discussion on the ticket.
        """
        if not self.summary_prompt:
            raise RuntimeError("Summary prompt template missing")
        template = self.summary_prompt
        prompt = safe_format(
            template,
            {"summary": summary, "description": description, "comments": comments},
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat_completion(messages, **kwargs)
        return self.client.extract_text(response)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ask(
        self,
        issue_id: str,
        question: str,
        include_history: bool = True,
        history: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> str:
        """Answer ``question`` about ``issue_id`` using the configured LLM."""
        logger.info("Answering question for issue %s", issue_id)
        issue_json = get_issue_by_id_tool.run(issue_id)
        history_json = get_issue_history_tool.run(issue_id) if include_history else ""
        related_summary = ""
        if self.config.follow_related_jiras:
            related_json = get_related_issues_tool.run(issue_id)
            related = json.loads(related_json)
            all_related = related.get("subtasks", []) + related.get("linked_issues", [])
            related_summary = self._summarize_related(all_related, **kwargs) if all_related else ""

        if not self.insights_prompt:
            raise RuntimeError("Insights prompt template missing")
        if include_history:
            prompt_template = self.insights_prompt
        else:
            prompt_template = self.insights_prompt
        values = {"issue": issue_json, "history": history_json, "question": question, "related": related_summary}
        prompt = safe_format(prompt_template, values)
        system_msg = {
            "role": "system",
            "content": f"Current date and time: {datetime.now().isoformat()}",
        }
        messages = [system_msg] + (history or []) + [{"role": "user", "content": prompt}]
        response = self.client.chat_completion(messages, **kwargs)
        return self.client.extract_text(response)

    def summarize(self, issue_id: str, **kwargs: Any) -> str:
        """Return a short summary for ``issue_id``."""
        logger.info("Summarizing issue %s", issue_id)
        issue_json = get_issue_by_id_tool.run(issue_id)
        issue = json.loads(issue_json)
        fields = issue.get("fields", {})
        comments_json = get_issue_comments_tool.run(issue_id)
        comments_list = json.loads(comments_json)
        comments = "\n".join(c.get("body", "") for c in comments_list)
        return self._summarize(
            fields.get("summary", ""),
            fields.get("description", ""),
            comments,
            **kwargs,
        )


__all__ = ["IssueInsightsAgent"]
