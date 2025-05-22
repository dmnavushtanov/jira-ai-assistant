"""Simple wrapper to allow pluggable LLMs used by the agent."""

from typing import Any, Dict

from langchain.llms import OpenAI

import config


class LLMWrapper:
    """Thin wrapper around the LangChain LLM interface."""

    def __init__(self, **kwargs: Any) -> None:
        params: Dict[str, Any] = {"api_key": config.OPENAI_API_KEY}
        params.update(kwargs)
        self.llm = OpenAI(**params)

    def __call__(self, prompt: str) -> str:
        return self.llm(prompt)


def get_llm(**kwargs: Any):
    """Return the configured LLM instance."""
    return LLMWrapper(**kwargs)
