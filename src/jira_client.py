from typing import Any, Dict, List

from jira import JIRA


class JiraClient:
    """Wrapper around the official ``jira`` client exposing common helpers."""

    def __init__(self, base_url: str, email: str, api_token: str) -> None:
        self._jira = JIRA(server=base_url.rstrip('/'), basic_auth=(email, api_token))

    def get_issue(self, issue_key: str, expand: str | None = None) -> Dict[str, Any]:
        """Retrieve an issue by its key."""
        issue = self._jira.issue(issue_key, expand=expand)
        return issue.raw

    def create_issue(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new issue and return its raw representation."""
        issue = self._jira.create_issue(fields=fields)
        return issue.raw

    def add_comment(self, issue_key: str, comment: str) -> Dict[str, Any]:
        """Add a comment to an issue and return the created comment."""
        cmt = self._jira.add_comment(issue_key, comment)
        return cmt.raw

    def get_comments(self, issue_key: str) -> List[Dict[str, Any]]:
        """Return comments for an issue."""
        comments = self._jira.comments(issue_key)
        return [c.raw for c in comments]

    def update_issue(self, issue_key: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Update fields of an issue."""
        issue = self._jira.issue(issue_key)
        issue.update(fields=fields)
        return issue.raw

    def get_changelog(self, issue_key: str) -> Dict[str, Any]:
        """Return changelog for an issue."""
        issue = self._jira.issue(issue_key, expand="changelog")
        return issue.raw.get("changelog", {})

    def search_issues(self, jql: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """Run a JQL query and return the matching issues."""
        issues = self._jira.search_issues(jql, **kwargs)
        return [iss.raw for iss in issues]
