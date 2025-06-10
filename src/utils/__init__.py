"""Utility helpers for the Jira AI Assistant."""

from .jira import extract_plain_text, JiraUtils, strip_nulls, strip_unused_jira_data
from .prompt import safe_format
from .rich_logger import RichLogger
from .context_memory import JiraContextMemory
from .http_client import SimpleHttpClient
from .json_utils import parse_json_block
from src.configs.config import load_config


def confirm_action(message: str) -> bool:
    """Return ``True`` if the user confirms the suggested Jira action."""

    cfg = load_config()
    if not cfg.ask_for_confirmation:
        return True
    ans = input(f"{message} [y/N]: ").strip().lower()
    return ans.startswith("y")

__all__ = [
    "extract_plain_text",
    "strip_nulls",
    "strip_unused_jira_data",
    "JiraUtils",
    "safe_format",
    "RichLogger",
    "JiraContextMemory",
    "SimpleHttpClient",
    "parse_json_block",
    "confirm_action",
]
