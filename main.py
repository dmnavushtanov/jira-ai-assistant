"""Simple interactive entry point for Jira AI Assistant."""

import logging

from src.agents.jira_agent import get_agent


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    question = input("Ask a Jira question: ").strip()
    if not question:
        print("No question provided.")
        return

    agent = get_agent()
    logging.info("Sending question to agent: %s", question)
    answer = agent.run(question)
    print(answer)


if __name__ == "__main__":
    main()
