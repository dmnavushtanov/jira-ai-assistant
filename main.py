import os
from dotenv import load_dotenv
from src.jira_client import JiraClient
import json
from langchain.schema.runnable import RunnableLambda, RunnableSequence

from src.agents.classifier import ClassifierAgent
from src.agents.api_validator import ApiValidatorAgent
from src.prompts import load_prompt
from src.utils import safe_format
from src.configs import load_config, setup_logging
from src.services.jira_service import get_issue_by_id_tool

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
    issue_id = input("Enter Jira issue ID: ").strip()

    classifier = ClassifierAgent()
    validator = ApiValidatorAgent()

    def classify_step(iid: str) -> dict:
        issue_json = get_issue_by_id_tool.run(iid)
        issue = json.loads(issue_json)

        print("\nIssue found:")
        print(f"Key: {issue.get('key')}")
        fields = issue.get('fields', {})
        project = fields.get('project', {})
        print(f"Project: {project.get('key')} - {project.get('name')}")
        print(f"Summary: {fields.get('summary')}")
        print(f"Status: {fields.get('status', {}).get('name')}")

        prompt = safe_format(
            load_prompt("classifier.txt"),
            {
                "summary": fields.get('summary', ''),
                "description": fields.get('description', ''),
            },
        )
        classification = classifier.classify(prompt)
        print(f"\nClassification: {classification}")
        return {"issue": issue, "classification": classification}

    def validate_step(data: dict) -> str:
        if str(data.get("classification", "")).upper().startswith("API"):
            return validator.validate(data["issue"])
        return "Validation skipped (not API related)"

    sequence = RunnableSequence(
        first=RunnableLambda(classify_step),
        last=RunnableLambda(validate_step),
    )

    try:
        result = sequence.invoke(issue_id)
        print(f"Validation result: {result}")
    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
