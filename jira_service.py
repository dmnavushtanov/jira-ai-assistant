import json
import os
from typing import Dict, Any

from langchain.tools import Tool

from src.jira_client import JiraClient


def _get_jira_client() -> JiraClient:
    """Instantiate a :class:`JiraClient` using environment variables."""
    base_url = os.getenv("JIRA_BASE_URL")
    email = os.getenv("JIRA_EMAIL")
    token = os.getenv("JIRA_API_TOKEN")
    if not all([base_url, email, token]):
        raise ValueError(
            "JIRA_BASE_URL, JIRA_EMAIL and JIRA_API_TOKEN environment variables must be set"
        )
    return JiraClient(base_url, email, token)


# --- Tool for getting issue details by ID ---

def get_issue_by_id_func(issue_id: str) -> str:
    """Fetch details for a given Jira issue ID."""
    client = _get_jira_client()
    issue = client.get_issue(issue_id)
    return json.dumps(issue)


get_issue_by_id_tool = Tool(
    name="get_issue_by_id",
    func=get_issue_by_id_func,
    description=(
        "Useful for when you need to get the details of a specific Jira issue. "
        "Input should be the Jira issue ID (e.g., 'PROJ-123')."
    ),
)


# --- Tool for creating a new Jira issue ---

def create_jira_issue_func(
    summary: str, description: str, project_key: str, issue_type: str = "Task"
) -> str:
    """Create a new Jira issue."""
    client = _get_jira_client()
    fields: Dict[str, Any] = {
        "project": {"key": project_key},
        "summary": summary,
        "description": description,
        "issuetype": {"name": issue_type},
    }
    issue = client.create_issue(fields)
    return json.dumps(issue)


create_jira_issue_tool = Tool(
    name="create_jira_issue",
    func=create_jira_issue_func,
    description=(
        "Useful for when you need to create a new Jira issue. Requires summary, "
        "description and project_key. Optionally an issue_type can be provided "
        "(default is 'Task')."
    ),
)


# --- Tool for fetching comments of an issue ---

def get_issue_comments_func(issue_id: str) -> str:
    """Fetch comments for a given Jira issue ID."""
    client = _get_jira_client()
    comments = client.get_comments(issue_id)
    return json.dumps(comments)


get_issue_comments_tool = Tool(
    name="get_issue_comments",
    func=get_issue_comments_func,
    description=(
        "Useful for when you need to get the comments for a specific Jira issue. "
        "Input should be the Jira issue ID (e.g., 'PROJ-123')."
    ),
)


# --- Tool for fetching the changelog/history of an issue ---

def get_issue_history_func(issue_id: str) -> str:
    """Fetch the changelog for a given Jira issue ID."""
    client = _get_jira_client()
    changelog = client.get_changelog(issue_id)
    return json.dumps(changelog)


get_issue_history_tool = Tool(
    name="get_issue_history",
    func=get_issue_history_func,
    description=(
        "Useful for when you need to get the change history for a specific Jira "
        "issue. Input should be the Jira issue ID (e.g., 'PROJ-123')."
    ),
)

# --- Tool for adding a comment to an issue ---

def add_comment_to_issue_func(issue_id: str, comment: str) -> str:
    """Add a comment to the specified issue."""
    client = _get_jira_client()
    result = client.add_comment(issue_id, comment)
    return json.dumps(result)

add_comment_to_issue_tool = Tool(
    name="add_comment_to_issue",
    func=add_comment_to_issue_func,
    description=(
        "Useful for adding a text comment to an existing Jira issue."
        " Input requires the issue ID and the comment body."
    ),
)

# --- Tool for updating fields of an issue ---

def update_issue_fields_func(issue_id: str, fields_json: str) -> str:
    """Update one or more fields on the given issue.

    ``fields_json`` should be a JSON string mapping field names to new values.
    """
    client = _get_jira_client()
    fields: Dict[str, Any] = json.loads(fields_json)
    updated = client.update_issue(issue_id, fields)
    return json.dumps(updated)

update_issue_fields_tool = Tool(
    name="update_issue_fields",
    func=update_issue_fields_func,
    description=(
        "Update fields of an existing Jira issue."
        " Input should be the issue ID followed by a JSON string of fields to update."
    ),
)


# Collection of available Jira tools
jira_tools = [
    get_issue_by_id_tool,
    create_jira_issue_tool,
    get_issue_comments_tool,
    get_issue_history_tool,
    add_comment_to_issue_tool,
    update_issue_fields_tool,
]

__all__ = [
    "get_issue_by_id_tool",
    "create_jira_issue_tool",
    "get_issue_comments_tool",
    "get_issue_history_tool",
    "add_comment_to_issue_tool",
    "update_issue_fields_tool",
    "jira_tools",
]
