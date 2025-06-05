"""
Prompts module for the Jira AI Assistant.

This module contains prompt templates and text resources for AI interactions.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Get the directory path for loading prompt files
PROMPTS_DIR = Path(__file__).parent

def load_prompt(filename: str) -> str:
    """Load a prompt template from file."""
    filepath = PROMPTS_DIR / filename
    if filepath.exists():
        logger.debug("Loading prompt file: %s", filepath)
        try:
            content = filepath.read_text(encoding="utf-8")
            logger.info("Loaded prompt %s", filename)
            return content
        except Exception:  # pragma: no cover - file read errors
            logger.exception("Failed to read prompt file %s", filepath)
            return ""
    logger.warning("Prompt file %s not found", filepath)
    return ""

__all__ = ["load_prompt", "PROMPTS_DIR"] 