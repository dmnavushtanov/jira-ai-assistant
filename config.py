"""Configuration for Jira AI Assistant."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Determine the .env file path (look for it in the project root)
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# If .env wasn't found in the project root, also try the current working directory
if not env_path.exists():
    load_dotenv()  # Will look for .env in current working directory

# Load environment variables with default values
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

JIRA_URL = os.getenv("JIRA_URL", "https://fadata.atlassian.net/")
JIRA_USERNAME = os.getenv("JIRA_USERNAME", "dimitar.navushtanov@fadata.eu")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN", "")

# Alias for backwards compatibility
JIRA_API_KEY = JIRA_API_TOKEN

# Validate configuration
def validate_config() -> None:
    """Validate that required configuration values are set."""
    missing: list[str] = []
    
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    
    if not JIRA_API_TOKEN:
        missing.append("JIRA_API_TOKEN")
    
    if missing:
        print(f"WARNING: Missing required configuration: {', '.join(missing)}")
        print("Please set these environment variables or add them to a .env file.")

# Run validation on import
validate_config()
