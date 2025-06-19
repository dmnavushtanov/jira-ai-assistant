"""Utility helpers for the Jira AI Assistant."""

from .jira import (
    extract_plain_text,
    JiraUtils,
    strip_nulls,
    strip_unused_jira_data,
    normalize_newlines,
)
from .prompt import safe_format
from .rich_logger import RichLogger
from .context_memory import JiraContextMemory
from .http_client import SimpleHttpClient
from .json_utils import parse_json_block
import logging

from src.configs.config import load_config


logger = logging.getLogger(__name__)


def confirm_action(message: str) -> bool:
    """Return ``True`` without blocking for user input."""

    cfg = load_config()
    if cfg.ask_for_confirmation:
        logger.info("Confirmation requested: %s", message)
    # In non-interactive contexts we can't block for input
    return True

__all__ = [
    "extract_plain_text",
    "strip_nulls",
    "strip_unused_jira_data",
    "JiraUtils",
    "normalize_newlines",
    "safe_format",
    "RichLogger",
    "JiraContextMemory",
    "SimpleHttpClient",
    "parse_json_block",
    "confirm_action",
]
