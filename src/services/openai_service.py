from typing import Any, List, Dict

from src.llm_clients.openai_client import OpenAIClient
import logging
from src.configs import load_config, setup_logging

logger = logging.getLogger(__name__)


class OpenAIService:
    """High level service exposing simple question answering using OpenAI."""

    def __init__(self, config_path: str = None) -> None:
        logger.debug("Initializing OpenAIService with config_path=%s", config_path)
        self.client = OpenAIClient(config_path)

    def ask_question(self, question: str, **kwargs: Any) -> str:
        """Return the assistant answer for ``question`` using OpenAI chat API."""
        logger.debug("Asking question: %s", question)
        messages: List[Dict[str, str]] = [{"role": "user", "content": question}]
        response = self.client.chat_completion(messages, **kwargs)
        answer = response.choices[0].message.content.strip()
        logger.info("Received response from OpenAI")
        return answer


__all__ = ["OpenAIService"]


if __name__ == "__main__":
    import sys

    cfg = load_config()
    setup_logging(cfg)

    prompt = " ".join(sys.argv[1:]) or input("Ask a question: ")
    service = OpenAIService()
    answer = service.ask_question(prompt)
    print(answer)
