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
from .plan_executor import OperationsPlanExecutor
import logging

logger = logging.getLogger(__name__)

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
    "OperationsPlanExecutor",
]
