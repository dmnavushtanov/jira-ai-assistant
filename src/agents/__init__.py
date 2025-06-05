"""Agents for the Jira AI Assistant."""

import logging

logger = logging.getLogger(__name__)
logger.debug("agents package loaded")
logger.info("ClassifierAgent will be exported")

from .classifier import ClassifierAgent

__all__ = ["ClassifierAgent"]

