import os
from dotenv import load_dotenv
from src.jira_client import JiraClient

from src.agents.router_agent import RouterAgent
from src.configs import load_config, setup_logging
import logging

try:
    import langchain
except Exception:  # pragma: no cover - langchain optional
    langchain = None

logger = logging.getLogger(__name__)

# Load environment variables from .env file (force reload)
load_dotenv(override=True)

# Load application configuration and configure logging
config = load_config()
setup_logging(config)
if langchain:
    langchain.debug = config.debug


def get_jira_client():
    """Return a JiraClient instance using environment variables from .env file."""
    base_url = os.getenv("JIRA_BASE_URL")
    email = os.getenv("JIRA_EMAIL")
    token = os.getenv("JIRA_API_TOKEN")
    
    if not all([base_url, email, token]):
        raise RuntimeError(
            "JIRA_BASE_URL, JIRA_EMAIL and JIRA_API_TOKEN must be set in .env file"
        )
    
    logger.debug("Creating JiraClient for %s", base_url)
    return JiraClient(base_url, email, token)


def main() -> None:
    logger.info("Starting main interaction loop")

    logger.debug("Instantiating RouterAgent")
    router = RouterAgent()

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
