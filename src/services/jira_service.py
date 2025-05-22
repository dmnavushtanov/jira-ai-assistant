from __future__ import annotations

"""Service layer for Jira interactions."""

from typing import Optional, List

from src.adapters.jira_api import JiraAPI, JiraConfig
from src.models.jira_models import Comment, Issue
import config


class JiraService:
    """High level Jira operations using :class:`JiraAPI`."""

    def __init__(self) -> None:
        self._api = JiraAPI(
            JiraConfig(config.JIRA_URL, config.JIRA_USERNAME, config.JIRA_API_TOKEN)
        )

    def fetch_issue(self, issue_key: str) -> Issue:
        """Return an :class:`Issue` model populated from Jira."""
        data = self._api.get_issue(issue_key)
        fields = data.get("fields", {})
        comments: List[Comment] = []
        comment_data = fields.get("comment", {}).get("comments", [])
        for c in comment_data:
            author = c.get("author", {}).get("displayName", "")
            body = c.get("body", "")
            comments.append(Comment(author=author, body=body))

        return Issue(
            key=data.get("key", issue_key),
            summary=fields.get("summary", ""),
            description=fields.get("description"),
            comments=comments,
        )

    def get_issue_description(self, issue_key: str) -> Optional[str]:
        """Convenience method to return just the issue description."""
        issue = self.fetch_issue(issue_key)
        return issue.description
