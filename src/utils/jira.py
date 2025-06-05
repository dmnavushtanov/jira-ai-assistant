"""Jira-related helper utilities."""
from typing import Any, List


def extract_plain_text(content: Any) -> str:
    """Return plain text from Jira fields that may use Atlassian Document Format."""
    if isinstance(content, str):
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
        _walk(content)
    return " ".join(parts).strip()


__all__ = ["extract_plain_text"]
