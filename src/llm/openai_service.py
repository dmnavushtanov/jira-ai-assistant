import openai
from typing import Any, Dict, List

import config


class OpenAIService:
    """Service wrapper around the OpenAI API."""

    def __init__(self, model: str | None = None) -> None:
        openai.api_key = config.OPENAI_API_KEY
        self.model = model or config.OPENAI_MODEL

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Send a list of messages to the OpenAI Responses API."""
        response = openai.Responses.create(model=self.model, messages=messages, **kwargs)
        return response["choices"][0]["message"]["content"]

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """Call the model with a single prompt string."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)


_service: OpenAIService | None = None


def get_service() -> OpenAIService:
    """Return a singleton instance of :class:`OpenAIService`."""
    global _service
    if _service is None:
        _service = OpenAIService()
    return _service

