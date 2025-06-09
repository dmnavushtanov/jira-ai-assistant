import os
from dotenv import load_dotenv
from src.jira_client import JiraClient

from src.agents.router_agent import RouterAgent
from src.configs import load_config, setup_logging
import logging

logger = logging.getLogger(__name__)

# Load environment variables from .env file (force reload)
load_dotenv(override=True)

# Load application configuration and configure logging
config = load_config()
setup_logging(config)


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
    question = input("Enter your question: ").strip()

    logger.debug("Instantiating RouterAgent")
    router = RouterAgent()

    try:
        answer = router.ask(question)
        logger.info("Agent response: %s", answer)
    except Exception:
        logger.exception("Error processing question")


if __name__ == "__main__":
    main()
