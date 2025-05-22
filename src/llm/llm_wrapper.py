"""Simple wrapper over :class:`OpenAIService` used by the agent."""

from typing import Any

from .openai_service import get_service


class LLMWrapper:
    """Thin wrapper exposing the OpenAI service via a simple ``__call__``."""

    def __init__(self, **_: Any) -> None:
        self.service = get_service()

    def __call__(self, prompt: str) -> str:
        return self.service(prompt)


def get_llm(**kwargs: Any):
    """Return the configured LLM instance."""
    return LLMWrapper(**kwargs)
