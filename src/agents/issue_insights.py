"""Agent providing insights about Jira issues."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from src.configs.config import load_config
from src.llm_clients.openai_client import OpenAIClient
from src.llm_clients.claude_client import ClaudeClient
from src.prompts import load_prompt
from src.utils import safe_format
from src.services.jira_service import (
    get_issue_by_id_tool,
    get_issue_history_tool,
)

logger = logging.getLogger(__name__)
logger.debug("issue_insights module loaded")


class IssueInsightsAgent:
    """Agent that answers general questions about Jira issues."""

    def __init__(self, config_path: str | None = None) -> None:
        logger.debug("Initializing IssueInsightsAgent with config_path=%s", config_path)
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

        # Tools available to this agent
        self.tools = [get_issue_by_id_tool, get_issue_history_tool]

        self.summary_prompt = load_prompt("issue_summary.txt")
        self.insights_prompt = load_prompt("issue_insights.txt")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _summarize(self, summary: str, description: str, **kwargs: Any) -> str:
        """Return a short summary of the issue using the configured LLM."""
        template = self.summary_prompt or (
            "Provide a short, one or two sentence summary of the following Jira issue.\n"
            "Summary: {summary}\nDescription: {description}"
        )
        prompt = safe_format(template, {"summary": summary, "description": description})
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat_completion(messages, **kwargs)
        try:
            return response.choices[0].message.content.strip()
        except Exception:
            try:
                return response["choices"][0]["message"]["content"].strip()
            except Exception:  # pragma: no cover
                logger.exception("Failed to parse summary response")
                return str(response)

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

        if include_history:
            prompt_template = self.insights_prompt or (
                "You are a Jira assistant. Given the issue details and change history below, answer the user's question.\n"
                "Issue JSON:\n{issue}\n\nHistory JSON:\n{history}\n\nQuestion: {question}"
            )
        else:
            prompt_template = (
                self.insights_prompt
                or (
                    "You are a Jira assistant. Given the issue details below, answer the user's question.\n"
                    "Issue JSON:\n{issue}\n\nQuestion: {question}"
                )
            )
        values = {"issue": issue_json, "history": history_json, "question": question}
        prompt = safe_format(prompt_template, values)
        messages = (history or []) + [{"role": "user", "content": prompt}]
        response = self.client.chat_completion(messages, **kwargs)
        try:
            return response.choices[0].message.content.strip()
        except Exception:
            try:
                return response["choices"][0]["message"]["content"].strip()
            except Exception:  # pragma: no cover
                logger.exception("Failed to parse response")
                return str(response)

    def summarize(self, issue_id: str, **kwargs: Any) -> str:
        """Return a short summary for ``issue_id``."""
        logger.info("Summarizing issue %s", issue_id)
        issue_json = get_issue_by_id_tool.run(issue_id)
        issue = json.loads(issue_json)
        fields = issue.get("fields", {})
        return self._summarize(fields.get("summary", ""), fields.get("description", ""), **kwargs)


__all__ = ["IssueInsightsAgent"]
