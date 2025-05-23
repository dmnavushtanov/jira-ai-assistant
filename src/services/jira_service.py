from __future__ import annotations

"""Service layer for Jira interactions."""

from typing import Any, Optional, List

from src.adapters.jira_api import JiraAPI, JiraConfig
import requests
from src.models.jira_models import Comment, Issue
import config


def _extract_plain_text(data: Any) -> str:
    """Return plain text from Atlassian document format structures."""
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        return "".join(_extract_plain_text(d) for d in data)
    if isinstance(data, dict):
        text = data.get("text", "")
        if "content" in data:
            text += "".join(_extract_plain_text(c) for c in data["content"])
        return text
    return ""


class JiraService:
    """High level Jira operations using :class:`JiraAPI`."""

    def __init__(self) -> None:
        self._api = JiraAPI(
            JiraConfig(config.JIRA_URL, config.JIRA_USERNAME, config.JIRA_API_TOKEN)
        )

    def fetch_issue(self, issue_key: str) -> Issue:
        """Return an :class:`Issue` model populated from Jira."""
        try:
            data = self._api.get_issue(issue_key)
        except requests.HTTPError as exc:
            # Provide a more friendly error message when authentication fails
            raise RuntimeError(
                f"Failed to fetch issue {issue_key}: {exc}"
            ) from exc
        fields = data.get("fields", {})
        comments: List[Comment] = []
        comment_data = fields.get("comment", {}).get("comments", [])
        for c in comment_data:
            author = c.get("author", {}).get("displayName", "")
            body = _extract_plain_text(c.get("body", ""))
            comments.append(Comment(author=author, body=body))

        return Issue(
            key=data.get("key", issue_key),
            summary=fields.get("summary", ""),
            description=_extract_plain_text(fields.get("description")),
            comments=comments,
        )

    def get_issue_description(self, issue_key: str) -> Optional[str]:
        """Convenience method to return just the issue description."""
        issue = self.fetch_issue(issue_key)
        return issue.description
