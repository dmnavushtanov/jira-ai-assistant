from typing import List

from src.services.jira_service import jira_tools
from src.services.openai_service import OpenAIService


class JiraAIAgent:
    """Simple agent exposing Jira tools and OpenAI question answering."""

    def __init__(self, service: OpenAIService | None = None) -> None:
        self.service = service or OpenAIService()
        # expose available tools so integrators can wire them as needed
        self.tools: List = jira_tools

    def ask(self, question: str) -> str:
        """Return OpenAI's answer for the provided question."""
        return self.service.ask_question(question)


__all__ = ["JiraAIAgent"]


if __name__ == "__main__":
    import sys

    prompt = " ".join(sys.argv[1:]) or input("Ask a question: ")
    agent = JiraAIAgent()
    print(agent.ask(prompt))
