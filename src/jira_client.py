from typing import Any, Dict, List

from src.utils import JiraUtils, strip_unused_jira_data, strip_nulls

from jira import JIRA
import logging

logger = logging.getLogger(__name__)


class JiraClient:
    """Wrapper around the official ``jira`` client exposing common helpers."""

    def __init__(self, base_url: str, email: str, api_token: str, *, strip_unused_payload: bool = False) -> None:
        logger.debug("Initializing JiraClient with base_url=%s email=%s", base_url, email)
        self._jira = JIRA(server=base_url.rstrip('/'), basic_auth=(email, api_token))
        self._strip_unused = strip_unused_payload

    def get_issue(self, issue_key: str, expand: str | None = None) -> Dict[str, Any]:
        """Retrieve an issue by its key."""
        logger.debug("Fetching issue %s", issue_key)
        issue = self._jira.issue(issue_key, expand=expand)
        return JiraUtils.clean_issue(issue.raw, strip_unused=self._strip_unused)

    def create_issue(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new issue and return its raw representation."""
        logger.debug("Creating issue with fields: %s", fields)
        issue = self._jira.create_issue(fields=fields)
        return JiraUtils.clean_issue(issue.raw, strip_unused=self._strip_unused)

    def add_comment(self, issue_key: str, comment: str) -> Dict[str, Any]:
        """Add a comment to an issue and return the created comment."""
        logger.debug("Adding comment to issue %s", issue_key)
        cmt = self._jira.add_comment(issue_key, comment)
        data = strip_nulls(cmt.raw)
        if self._strip_unused:
            data = strip_unused_jira_data(data)
        return data

    def get_comments(self, issue_key: str) -> List[Dict[str, Any]]:
        """Return comments for an issue."""
        logger.debug("Fetching comments for issue %s", issue_key)
        comments = self._jira.comments(issue_key)
        cleaned = [strip_nulls(c.raw) for c in comments]
        if self._strip_unused:
            cleaned = [strip_unused_jira_data(c) for c in cleaned]
        return cleaned

    def update_issue(self, issue_key: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Update fields of an issue."""
        logger.debug("Updating issue %s with fields: %s", issue_key, fields)
        issue = self._jira.issue(issue_key)
        issue.update(fields=fields)
        return JiraUtils.clean_issue(issue.raw, strip_unused=self._strip_unused)

    def get_changelog(self, issue_key: str) -> Dict[str, Any]:
        """Return cleaned changelog for an issue."""
        logger.debug("Fetching changelog for issue %s", issue_key)
        issue = self._jira.issue(issue_key, expand="changelog")
        history = issue.raw.get("changelog", {})
        return JiraUtils.clean_history(history, strip_unused=self._strip_unused)

    def search_issues(self, jql: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """Run a JQL query and return the matching issues."""
        logger.debug("Searching issues with JQL: %s", jql)
        issues = self._jira.search_issues(jql, **kwargs)
        return [JiraUtils.clean_issue(iss.raw, strip_unused=self._strip_unused) for iss in issues]
