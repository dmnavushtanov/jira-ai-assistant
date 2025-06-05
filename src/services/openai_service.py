from typing import Any, List, Dict

from src.llm_clients.openai_client import OpenAIClient


class OpenAIService:
    """High level service exposing simple question answering using OpenAI."""

    def __init__(self, config_path: str = None) -> None:
        self.client = OpenAIClient(config_path)

    def ask_question(self, question: str, **kwargs: Any) -> str:
        """Return the assistant answer for ``question`` using OpenAI chat API."""
        messages: List[Dict[str, str]] = [{"role": "user", "content": question}]
        response = self.client.chat_completion(messages, **kwargs)
        return response["choices"][0]["message"]["content"].strip()


__all__ = ["OpenAIService"]


if __name__ == "__main__":
    import sys

    prompt = " ".join(sys.argv[1:]) or input("Ask a question: ")
    service = OpenAIService()
    answer = service.ask_question(prompt)
    print(answer)
