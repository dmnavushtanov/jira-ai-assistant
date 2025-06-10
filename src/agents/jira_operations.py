"""Agent for performing write operations on Jira issues."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.services.jira_service import (
    add_comment_to_issue_tool,
    create_jira_issue_tool,
    fill_field_by_label_tool,
    update_issue_fields_tool,
)
from src.configs.config import load_config

logger = logging.getLogger(__name__)
logger.debug("jira_operations module loaded")


class JiraOperationsAgent:
    """Agent that performs modifications on Jira issues."""

    def __init__(self, config_path: str | None = None) -> None:
        logger.debug("Initializing JiraOperationsAgent with config_path=%s", config_path)
        self.config = load_config(config_path)

    def add_comment(self, issue_id: str, comment: str, **kwargs: Any) -> str:
        """Add ``comment`` to ``issue_id`` using the Jira API."""
        logger.info("Adding comment to %s", issue_id)
        payload = json.dumps({"issue_id": issue_id, "comment": comment})
        result_json = add_comment_to_issue_tool.run(payload)
        try:
            parsed = json.loads(result_json)
        except Exception:
            logger.debug("Failed to parse add_comment response")
            parsed = result_json
        logger.info("Comment added to %s", issue_id)
        return parsed

    def create_issue(
        self,
        summary: str,
        description: str,
        project_key: str,
        issue_type: str = "Task",
        **kwargs: Any,
    ) -> str:
        """Create a new Jira issue."""
        logger.info("Creating issue in %s", project_key)
        result_json = create_jira_issue_tool.run(summary, description, project_key, issue_type)
        try:
            return json.loads(result_json)
        except Exception:
            logger.debug("Failed to parse create_issue response")
            return result_json

    def update_fields(self, issue_id: str, fields_json: str, **kwargs: Any) -> str:
        """Update fields on ``issue_id``."""
        logger.info("Updating issue %s", issue_id)
        result_json = update_issue_fields_tool.run(issue_id, fields_json)
        try:
            return json.loads(result_json)
        except Exception:
            logger.debug("Failed to parse update_fields response")
            return result_json

    def fill_field_by_label(
        self, issue_id: str, field_label: str, value: str, **kwargs: Any
    ) -> str:
        """Set an issue field value using the field's label."""
        logger.info("Setting %s on %s", field_label, issue_id)
        result_json = fill_field_by_label_tool.run(issue_id, field_label, value)
        try:
            return json.loads(result_json)
        except Exception:
            logger.debug("Failed to parse fill_field_by_label response")
            return result_json


__all__ = ["JiraOperationsAgent"]
