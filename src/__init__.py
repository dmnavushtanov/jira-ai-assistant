"""
Jira AI Assistant - Main Package

This package contains utilities for interacting with Jira and OpenAI services.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path to ensure imports work from anywhere
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

__version__ = "1.0.0"
__author__ = "Jira AI Assistant"

# Main package exports
__all__ = [
    "configs",
    "services", 
    "llm_clients",
    "agents",
    "ui"
] 