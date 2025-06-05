"""Simple classifier agent using a configurable LLM provider."""

from typing import List, Dict, Any

from src.configs.config import load_config
from src.llm_clients.openai_client import OpenAIClient
from src.llm_clients.claude_client import ClaudeClient


class ClassifierAgent:
    """Agent that selects an LLM client based on configuration."""

    def __init__(self, config_path: str | None = None) -> None:
        self.config = load_config(config_path)
        llm = self.config.base_llm.lower()
        if llm == "openai":
            self.client = OpenAIClient(config_path)
        elif llm in {"anthropic", "claude"}:
            self.client = ClaudeClient(config_path)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.base_llm}")

    def classify(self, prompt: str, **kwargs: Any) -> Any:
        """Return the classification result for ``prompt``."""
        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
        response = self.client.chat_completion(messages, **kwargs)
        try:
            return response.choices[0].message.content.strip()
        except Exception:
            try:
                return response["choices"][0]["message"]["content"].strip()
            except Exception:
                return response


__all__ = ["ClassifierAgent"]

