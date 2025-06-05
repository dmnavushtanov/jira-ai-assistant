"""
Prompts module for the Jira AI Assistant.

This module contains prompt templates and text resources for AI interactions.
"""

import os
from pathlib import Path

# Get the directory path for loading prompt files
PROMPTS_DIR = Path(__file__).parent

def load_prompt(filename: str) -> str:
    """Load a prompt template from file."""
    filepath = PROMPTS_DIR / filename
    if filepath.exists():
        return filepath.read_text(encoding='utf-8')
    return ""

__all__ = ["load_prompt", "PROMPTS_DIR"] 