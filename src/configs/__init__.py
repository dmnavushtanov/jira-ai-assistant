"""
Configuration module for the Jira AI Assistant.

This module handles loading configuration from YAML files and environment variables.
"""

from .config import Config, load_config

__all__ = ["Config", "load_config"] 