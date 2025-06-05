"""Abstract base class for LLM clients."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)
logger.debug("BaseLLMClient module loaded")


class BaseLLMClient(ABC):
    """Interface that all LLM clients must implement."""

    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        """Return a chat completion response."""
        raise NotImplementedError


__all__ = ["BaseLLMClient"]

