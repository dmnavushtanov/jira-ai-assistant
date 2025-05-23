"""Simple wrapper over :class:`OpenAIService` used by the agent."""

from typing import Any, List, Optional

from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

from .openai_service import get_service


class LLMWrapper(LLM):
    """Thin wrapper exposing the OpenAI service via the LangChain ``LLM`` API."""

    def __init__(self, **_: Any) -> None:
        super().__init__()
        self.service = get_service()

    @property
    def _llm_type(self) -> str:  # pragma: no cover - simple metadata
        return "openai-wrapper"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Execute the prompt against the underlying service."""
        return self.service(prompt)


def get_llm(**kwargs: Any) -> LLMWrapper:
    """Return the configured LLM instance."""
    return LLMWrapper(**kwargs)
