"""Service layer for the Jira AI Assistant."""

import logging

logger = logging.getLogger(__name__)
logger.debug("services package initialized")

from .openai_service import OpenAIService

__all__ = ["OpenAIService"]

