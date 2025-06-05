import os
from dotenv import load_dotenv
from src.jira_client import JiraClient
from src.agents.classifier import ClassifierAgent
from src.prompts import load_prompt
from src.configs import load_config, setup_logging

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
    
    return JiraClient(base_url, email, token)


def main() -> None:
    print("Initializing Jira client...")
    
    try:
        client = get_jira_client()
        print("Jira client initialized successfully")
    except Exception as exc:
        print(f"Failed to initialize Jira client: {exc}")
        return

    issue_id = input("Enter Jira issue ID: ").strip()

    try:
        issue = client.get_issue(issue_id)
        print("\nIssue found:")
        print(f"Key: {issue['key']}")
        print(
            f"Project: {issue['fields']['project']['key']} - {issue['fields']['project']['name']}"
        )
        print(f"Summary: {issue['fields']['summary']}")
        print(f"Status: {issue['fields']['status']['name']}")
        assignee = issue['fields']['assignee']['displayName'] if issue['fields']['assignee'] else 'Unassigned'
        print(f"Assignee: {assignee}")
        print(f"Reporter: {issue['fields']['reporter']['displayName']}")
        print(f"Created: {issue['fields']['created']}")
        print(f"Updated: {issue['fields']['updated']}")

        # Classify issue using the LLM
        prompt_template = load_prompt("classifier.txt")
        prompt = prompt_template.format(
            summary=issue['fields'].get('summary', ''),
            description=issue['fields'].get('description', '')
        )
        agent = ClassifierAgent()
        classification = agent.classify(prompt)
        print(f"\nClassification: {classification}")

    except Exception as exc:
        print(f"Failed to fetch issue {issue_id}: {exc}")


if __name__ == "__main__":
    main()
