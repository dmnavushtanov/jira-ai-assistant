from typing import Any, Dict, List

from src.utils import JiraUtils, strip_unused_jira_data, strip_nulls

from jira import JIRA
import logging

logger = logging.getLogger(__name__)


class JiraClient:
    """Wrapper around the official ``jira`` client exposing common helpers."""

    def __init__(
        self,
        base_url: str,
        email: str,
        api_token: str,
        *,
        strip_unused_payload: bool = False,
        log_payloads: bool = False,
    ) -> None:
        logger.debug(
            "Initializing JiraClient with base_url=%s email=%s", base_url, email
        )
        self._jira = JIRA(server=base_url.rstrip("/"), basic_auth=(email, api_token))
        self._strip_unused = strip_unused_payload
        self._log_payloads = log_payloads

    def get_issue(self, issue_key: str, expand: str | None = None) -> Dict[str, Any]:
        """Retrieve an issue by its key."""
        logger.debug("Fetching issue %s", issue_key)
        issue = self._jira.issue(issue_key, expand=expand)
        cleaned = JiraUtils.clean_issue(issue.raw, strip_unused=self._strip_unused)
        if self._log_payloads:
            logger.debug("Issue data for %s: %s", issue_key, cleaned)
        return cleaned

    def create_issue(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new issue and return its raw representation."""
        logger.debug("Creating issue with fields: %s", fields)
        issue = self._jira.create_issue(fields=fields)
        cleaned = JiraUtils.clean_issue(issue.raw, strip_unused=self._strip_unused)
        if self._log_payloads:
            logger.debug("Created issue payload: %s", cleaned)
        return cleaned

    def add_comment(self, issue_key: str, comment: str) -> Dict[str, Any]:
        """Add a comment to an issue and return the created comment."""
        logger.debug("Adding comment to issue %s", issue_key)
        cmt = self._jira.add_comment(issue_key, comment)
        data = strip_nulls(cmt.raw)
        if self._strip_unused:
            data = strip_unused_jira_data(data)
        if self._log_payloads:
            logger.debug("Comment payload for %s: %s", issue_key, data)
        return data

    def get_comments(self, issue_key: str) -> List[Dict[str, Any]]:
        """Return comments for an issue."""
        logger.debug("Fetching comments for issue %s", issue_key)
        comments = self._jira.comments(issue_key)
        cleaned = [strip_nulls(c.raw) for c in comments]
        if self._strip_unused:
            cleaned = [strip_unused_jira_data(c) for c in cleaned]
        if self._log_payloads:
            logger.debug("Comments payload for %s: %s", issue_key, cleaned)
        return cleaned

    def update_issue(self, issue_key: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Update fields of an issue."""
        logger.debug("Updating issue %s with fields: %s", issue_key, fields)
        issue = self._jira.issue(issue_key)
        issue.update(fields=fields)
        cleaned = JiraUtils.clean_issue(issue.raw, strip_unused=self._strip_unused)
        if self._log_payloads:
            logger.debug("Updated issue payload for %s: %s", issue_key, cleaned)
        return cleaned

    def get_changelog(self, issue_key: str) -> Dict[str, Any]:
        """Return cleaned changelog for an issue."""
        logger.debug("Fetching changelog for issue %s", issue_key)
        issue = self._jira.issue(issue_key, expand="changelog")
        history = issue.raw.get("changelog", {})
        cleaned = JiraUtils.clean_history(history, strip_unused=self._strip_unused)
        if self._log_payloads:
            logger.debug("Changelog payload for %s: %s", issue_key, cleaned)
        return cleaned

    def search_issues(self, jql: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """Run a JQL query and return the matching issues."""
        logger.debug("Searching issues with JQL: %s", jql)
        issues = self._jira.search_issues(jql, **kwargs)
        cleaned = [
            JiraUtils.clean_issue(iss.raw, strip_unused=self._strip_unused)
            for iss in issues
        ]
        if self._log_payloads:
            logger.debug("Search result payload: %s", cleaned)
        return cleaned

    def get_related_issues(self, issue_key: str) -> Dict[str, List[Dict[str, Any]]]:
        """Return subtasks and linked issues for the given ticket.

        Each returned issue also includes its comments under a ``comments`` key
        since context from discussions can be important.
        """
        logger.debug("Fetching related issues for %s", issue_key)
        issue = self._jira.issue(issue_key, expand="issuelinks,subtasks")
        subtasks: List[Dict[str, Any]] = []
        for sub in getattr(issue.fields, "subtasks", []):
            try:
                sub_issue = self._jira.issue(sub.key)
                cleaned = JiraUtils.clean_issue(
                    sub_issue.raw, strip_unused=self._strip_unused
                )
                cleaned["comments"] = self.get_comments(sub.key)
                subtasks.append(cleaned)
            except Exception:
                logger.exception("Failed to fetch subtask %s", getattr(sub, "key", ""))
        linked: List[Dict[str, Any]] = []
        for link in getattr(issue.fields, "issuelinks", []):
            other = getattr(link, "outwardIssue", None) or getattr(
                link, "inwardIssue", None
            )
            if not other:
                continue
            try:
                other_issue = self._jira.issue(other.key)
                cleaned = JiraUtils.clean_issue(
                    other_issue.raw, strip_unused=self._strip_unused
                )
                cleaned["comments"] = self.get_comments(other.key)
                linked.append(cleaned)
            except Exception:
                logger.exception(
                    "Failed to fetch linked issue %s", getattr(other, "key", "")
                )
        result = {"subtasks": subtasks, "linked_issues": linked}
        if self._log_payloads:
            logger.debug("Related issues payload for %s: %s", issue_key, result)
        return result

    def set_field_by_label(
        self, issue_key: str, field_label: str, value: Any
    ) -> Dict[str, Any]:
        """Set a field's value using its display label.

        ``field_label`` should match the human readable name shown in Jira's UI
        (e.g. ``"Definition Of Done"``). The helper will look up the
        corresponding field ID before performing the update.
        """
        logger.debug(
            "Setting field labelled %s on %s to %s", field_label, issue_key, value
        )
        field_id = None
        built_in_match = None
        for field in self._jira.fields():
            if field.get("name") != field_label:
                continue
            fid = field.get("id")
            # Prefer standard fields over custom ones when names collide
            if not fid.startswith("customfield_"):
                built_in_match = fid
                break
            if field_id is None:
                field_id = fid

        field_id = built_in_match or field_id
        if not field_id:
            raise ValueError(f"Field label '{field_label}' not found")

        issue = self._jira.issue(issue_key)
        issue.update(fields={field_id: value})
        cleaned = JiraUtils.clean_issue(issue.raw, strip_unused=self._strip_unused)
        if self._log_payloads:
            logger.debug("Field update payload for %s: %s", issue_key, cleaned)
        return cleaned

    def get_transitions(self, issue_key: str) -> List[Dict[str, Any]]:
        """Return the workflow transitions available for ``issue_key``."""
        logger.debug("Fetching transitions for %s", issue_key)
        transitions = self._jira.transitions(issue_key)
        cleaned = [strip_nulls(t) for t in transitions]
        if self._strip_unused:
            cleaned = [strip_unused_jira_data(t) for t in cleaned]
        if self._log_payloads:
            logger.debug("Transitions for %s: %s", issue_key, cleaned)
        return cleaned

    def transition_issue(self, issue_key: str, transition: str) -> Dict[str, Any]:
        """Move ``issue_key`` to the workflow state ``transition``."""
        logger.debug("Transitioning %s using %s", issue_key, transition)
        transitions = self._jira.transitions(issue_key)
        transition_id = None
        for t in transitions:
            tid = str(t.get("id"))
            name = str(t.get("name", ""))
            if transition.lower() in (tid.lower(), name.lower()):
                transition_id = tid
                break
        if not transition_id:
            raise ValueError(f"Transition '{transition}' not available for {issue_key}")
        self._jira.transition_issue(issue_key, transition_id)
        issue = self._jira.issue(issue_key)
        cleaned = JiraUtils.clean_issue(issue.raw, strip_unused=self._strip_unused)
        if self._log_payloads:
            logger.debug("Transitioned issue payload for %s: %s", issue_key, cleaned)
        return cleaned
