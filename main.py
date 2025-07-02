"""CLI entry point for the Jira AI assistant.

This script starts a simple interaction loop using ``RouterAgent``. The router
decides whether to validate APIs, perform operations, or provide insights based
on the user's question. Comment posting and test case generation are handled by
the router according to ``config.yml`` options.
"""

from dotenv import load_dotenv

from src.agents.router_agent import RouterAgent
from src.configs import load_config, setup_logging
import logging
import langchain

logger = logging.getLogger(__name__)

# Load environment variables from .env file (force reload)
load_dotenv(override=True)

# Load application configuration and configure logging
config = load_config()
setup_logging(config)

def main() -> None:
    logger.info("Starting main interaction loop")

    logger.debug("Instantiating RouterAgent")
    router = RouterAgent()
    logger.info("LangChain available - advanced routing enabled")

    while True:
        question = input("Enter your question (type 'exit' to quit): ").strip()
        if question.lower() == "exit":
            logger.info("Exiting interaction loop")
            break

        try:
            answer = router.ask(question)
            logger.info("Agent response: %s", answer)
            print(answer)
        except Exception:
            logger.exception("Error processing question")


if __name__ == "__main__":
    main()
