"""Jira-related helper utilities."""
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


def extract_plain_text(content: Any) -> str:
    """Return plain text from Jira fields that may use Atlassian Document Format."""
    if isinstance(content, str):
        logger.debug("Content already plain text")
        return content

    parts: List[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            text = node.get("text")
            if text:
                parts.append(str(text))
            for child in node.get("content", []):
                _walk(child)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    if content is not None:
        logger.debug("Walking structured content to extract text")
        _walk(content)
    return " ".join(parts).strip()



class JiraUtils:
    """Helper methods for cleaning and parsing Jira issues."""

    CUSTOM_FIELD_PREFIX = "customfield_"

    @classmethod
    def clean_fields(cls, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Return ``fields`` without ``None`` custom fields."""
        if not isinstance(fields, dict):
            logger.debug("Fields is not a dict; returning as-is")
            return fields
        logger.debug("Cleaning custom fields")
        cleaned = {
            k: v
            for k, v in fields.items()
            if not (k.startswith(cls.CUSTOM_FIELD_PREFIX) and v is None)
        }
        return cleaned

    @classmethod
    def clean_issue(cls, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Return ``issue`` with ``None`` custom fields removed."""
        fields = issue.get("fields")
        if isinstance(fields, dict):
            logger.debug("Cleaning issue fields for %s", issue.get("key"))
            cleaned = cls.clean_fields(fields)
            if cleaned is not fields:
                issue = dict(issue)
                issue["fields"] = cleaned
        return issue


__all__ = ["extract_plain_text", "JiraUtils"]
