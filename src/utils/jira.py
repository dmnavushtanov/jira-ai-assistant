"""Jira-related helper utilities."""
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


def normalize_newlines(text: str | None) -> str | None:
    """Return ``text`` with literal ``\\n`` sequences replaced."""
    if isinstance(text, str):
        return text.replace("\\n", "\n")
    return text


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



def strip_nulls(obj: Any) -> Any:
    """Recursively remove keys with ``None`` values from dictionaries and lists."""
    if isinstance(obj, dict):
        return {k: strip_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [strip_nulls(v) for v in obj if v is not None]
    return obj


def strip_unused_jira_data(obj: Any) -> Any:
    """Recursively remove keys that add noise to Jira payloads."""
    if isinstance(obj, dict):
        # If this looks like a user object keep only email and display name
        if "emailAddress" in obj and "displayName" in obj:
            return {
                "emailAddress": obj.get("emailAddress"),
                "displayName": obj.get("displayName"),
            }

        cleaned: Dict[str, Any] = {}
        for k, v in obj.items():
            key_lower = k.lower()
            if "avatar" in key_lower:
                continue
            if key_lower == "id" or key_lower.endswith("id") or key_lower.endswith("_id"):
                continue
            cleaned[k] = strip_unused_jira_data(v)
        return cleaned
    if isinstance(obj, list):
        return [strip_unused_jira_data(v) for v in obj]
    return obj


class JiraUtils:
    """Helper methods for cleaning and parsing Jira issues."""

    CUSTOM_FIELD_PREFIX = "customfield_"

    @classmethod
    def clean_fields(cls, fields: Dict[str, Any], *, strip_unused: bool = False) -> Dict[str, Any]:
        """Return ``fields`` without ``None`` values."""
        if not isinstance(fields, dict):
            logger.debug("Fields is not a dict; returning as-is")
            return fields
        logger.debug("Cleaning fields and stripping nulls")
        cleaned = {
            k: v
            for k, v in fields.items()
            if not k.startswith(cls.CUSTOM_FIELD_PREFIX)
        }
        cleaned = strip_nulls(cleaned)
        if strip_unused:
            cleaned = strip_unused_jira_data(cleaned)
        return cleaned

    @classmethod
    def clean_issue(cls, issue: Dict[str, Any], *, strip_unused: bool = False) -> Dict[str, Any]:
        """Return ``issue`` with ``None`` values removed."""
        fields = issue.get("fields")
        if isinstance(fields, dict):
            logger.debug("Cleaning issue fields for %s", issue.get("key"))
            cleaned = cls.clean_fields(fields, strip_unused=strip_unused)
            if cleaned is not fields:
                issue = dict(issue)
                issue["fields"] = cleaned
        cleaned_issue = strip_nulls(issue)
        if strip_unused:
            cleaned_issue = strip_unused_jira_data(cleaned_issue)
        return cleaned_issue

    @classmethod
    def clean_history(cls, history: Dict[str, Any], *, strip_unused: bool = False) -> Dict[str, Any]:
        """Return ``history`` with ``None`` values removed."""
        logger.debug("Cleaning changelog")
        cleaned_history = strip_nulls(history)
        if strip_unused:
            cleaned_history = strip_unused_jira_data(cleaned_history)
        return cleaned_history


__all__ = [
    "normalize_newlines",
    "extract_plain_text",
    "strip_nulls",
    "strip_unused_jira_data",
    "JiraUtils",
]
