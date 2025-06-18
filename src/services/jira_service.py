import json
import os
from typing import Dict, Any

from langchain.tools import Tool

from src.jira_client import JiraClient
from src.configs.config import load_config
import logging

logger = logging.getLogger(__name__)


def _get_jira_client() -> JiraClient:
    """Instantiate a :class:`JiraClient` using environment variables."""
    cfg = load_config()
    base_url = os.getenv("JIRA_BASE_URL")
    email = os.getenv("JIRA_EMAIL")
    token = os.getenv("JIRA_API_TOKEN")
    if not all([base_url, email, token]):
        raise ValueError(
            "JIRA_BASE_URL, JIRA_EMAIL and JIRA_API_TOKEN environment variables must be set"
        )
    logger.debug("Creating JiraClient for base_url=%s email=%s", base_url, email)
    logger.info("JiraClient initialized for %s", base_url)
    return JiraClient(
        base_url,
        email,
        token,
        strip_unused_payload=cfg.strip_unused_jira_data,
        log_payloads=cfg.log_jira_payloads,
    )


# --- Tool for getting issue details by ID ---


def get_issue_by_id_func(issue_id: str) -> str:
    """Fetch details for a given Jira issue ID."""
    logger.debug("Getting issue by ID: %s", issue_id)
    client = _get_jira_client()
    issue = client.get_issue(issue_id)
    cfg = load_config()
    if cfg.log_jira_payloads:
        logger.debug("Issue payload: %s", issue)
    logger.info("Fetched issue %s", issue_id)
    return json.dumps(issue)


get_issue_by_id_tool = Tool(
    name="get_issue_by_id",
    func=get_issue_by_id_func,
    description=(
        "Return the JSON details for a Jira issue. "
        "Input is the issue key such as 'PROJ-123'."
    ),
)


# --- Tool for creating a new Jira issue ---


def create_jira_issue_func(
    summary: str,
    description: str,
    project_key: str,
    issue_type: str = "Task",
    parent_key: str | None = None,
) -> str:
    """Create a new Jira issue.

    ``parent_key`` sets the parent when creating a sub-task.
    """
    logger.debug(
        "Creating Jira issue in project %s with summary %s", project_key, summary
    )
    from src.utils import confirm_action

    if not confirm_action(f"Create issue in {project_key} with summary '{summary}'?"):
        logger.info("User declined to create issue in %s", project_key)
        return "Creation cancelled by user"
    client = _get_jira_client()
    fields: Dict[str, Any] = {
        "project": {"key": project_key},
        "summary": summary,
        "description": description,
        "issuetype": {"name": issue_type},
    }
    if parent_key:
        fields["parent"] = {"key": parent_key}
    issue = client.create_issue(fields)
    cfg = load_config()
    if cfg.log_jira_payloads:
        logger.debug("Created issue payload: %s", issue)
    logger.info(
        "Created issue %s in project %s",
        issue.get("key", "unknown"),
        project_key,
    )
    return json.dumps(issue)


create_jira_issue_tool = Tool(
    name="create_jira_issue",
    func=create_jira_issue_func,
    description=(
        "Create a new Jira issue using summary, description and project_key. "
        "Optionally specify issue_type (default 'Task') and parent_key when "
        "creating a sub-task. Returns the created issue JSON."
    ),
)


# --- Tool for fetching comments of an issue ---


def get_issue_comments_func(issue_id: str) -> str:
    """Fetch comments for a given Jira issue ID."""
    logger.debug("Fetching comments for issue %s", issue_id)
    client = _get_jira_client()
    comments = client.get_comments(issue_id)
    cfg = load_config()
    if cfg.log_jira_payloads:
        logger.debug("Comments payload: %s", comments)
    logger.info("Retrieved %d comments for issue %s", len(comments), issue_id)
    return json.dumps(comments)


get_issue_comments_tool = Tool(
    name="get_issue_comments",
    func=get_issue_comments_func,
    description=("Return all comments for a Jira issue key as JSON."),
)


# --- Tool for fetching the changelog/history of an issue ---


def get_issue_history_func(issue_id: str) -> str:
    """Fetch the changelog for a given Jira issue ID."""
    logger.debug("Fetching changelog for issue %s", issue_id)
    client = _get_jira_client()
    changelog = client.get_changelog(issue_id)
    cfg = load_config()
    if cfg.log_jira_payloads:
        logger.debug("Changelog payload: %s", changelog)
    logger.info("Fetched changelog for issue %s", issue_id)
    return json.dumps(changelog)


get_issue_history_tool = Tool(
    name="get_issue_history",
    func=get_issue_history_func,
    description=("Return the change history (changelog) for a Jira issue key as JSON."),
)

# --- Tool for fetching subtasks and linked issues ---


def get_related_issues_func(issue_id: str) -> str:
    """Return subtasks and linked issues for the given ticket.

    Each related issue will include its comments since they often provide
    valuable context.
    """
    logger.debug("Fetching related issues for %s", issue_id)
    client = _get_jira_client()
    related = client.get_related_issues(issue_id)
    cfg = load_config()
    if cfg.log_jira_payloads:
        logger.debug("Related issues payload: %s", related)
    logger.info("Fetched related issues for %s", issue_id)
    return json.dumps(related)


get_related_issues_tool = Tool(
    name="get_related_issues",
    func=get_related_issues_func,
    description=(
        "Return linked issues and subtasks for a Jira issue key as JSON. "
        "Comments from those issues are included."
    ),
)

# --- Tool for adding a comment to an issue ---


def add_comment_to_issue_func(issue_id: str, comment: str) -> str:
    """Add a comment to the specified issue."""
    logger.debug("Adding comment to issue %s", issue_id)
    from src.utils import confirm_action

    if not confirm_action(f"Add the proposed comment to {issue_id}?"):
        logger.info("User declined to add comment to %s", issue_id)
        return "Comment creation cancelled by user"
    client = _get_jira_client()
    result = client.add_comment(issue_id, comment)
    cfg = load_config()
    if cfg.log_jira_payloads:
        logger.debug("Add comment payload: %s", result)
    logger.info("Added comment to issue %s", issue_id)
    return json.dumps(result)


def _add_comment_wrapper(*args: str) -> str:
    """Wrapper allowing single or dual argument invocation."""
    if len(args) == 1:
        text = args[0]
        try:
            data = json.loads(text)
            issue_id = data["issue_id"]
            comment = data["comment"]
        except Exception:
            if "|" in text:
                issue_id, comment = text.split("|", 1)
            else:
                raise TypeError(
                    "add_comment_to_issue requires 'issue_id|comment' or JSON"
                )
    elif len(args) == 2:
        issue_id, comment = args
    else:
        raise TypeError("add_comment_to_issue expects issue_id and comment")

    return add_comment_to_issue_func(issue_id.strip(), comment.strip())


add_comment_to_issue_tool = Tool(
    name="add_comment_to_issue",
    func=_add_comment_wrapper,
    description=(
        "Add a text comment to an existing Jira issue. "
        "Provide the issue key and comment. Returns the created comment JSON."
    ),
)

# --- Tool for updating fields of an issue ---


def update_issue_fields_func(issue_id: str, fields_json: str) -> str:
    """Update one or more fields on the given issue.

    ``fields_json`` should be a JSON string mapping field names to new values.
    """
    logger.debug("Updating issue %s with fields %s", issue_id, fields_json)
    from src.utils import confirm_action

    if not confirm_action(f"Apply the provided updates to {issue_id}?"):
        logger.info("User declined to update %s", issue_id)
        return "Update cancelled by user"
    client = _get_jira_client()
    fields: Dict[str, Any] = json.loads(fields_json)
    updated = client.update_issue(issue_id, fields)
    cfg = load_config()
    if cfg.log_jira_payloads:
        logger.debug("Update fields payload: %s", updated)
    logger.info("Updated issue %s", issue_id)
    return json.dumps(updated)


def fill_field_by_label_func(issue_id: str, field_label: str, value: str) -> str:
    """Update an issue field using its display label."""
    logger.debug("Filling field %s on %s with value %s", field_label, issue_id, value)
    from src.utils import confirm_action

    if not confirm_action(f"Set '{field_label}' on {issue_id} to '{value}'?"):
        logger.info("User declined to update %s", issue_id)
        return "Field update cancelled by user"
    client = _get_jira_client()
    updated = client.set_field_by_label(issue_id, field_label, value)
    cfg = load_config()
    if cfg.log_jira_payloads:
        logger.debug("Fill field payload: %s", updated)
    logger.info("Updated %s on %s", field_label, issue_id)
    return json.dumps(updated)


def _fill_field_by_label_wrapper(*args: str) -> str:
    """Wrapper allowing single or triple argument invocation."""
    if len(args) == 1:
        text = args[0]
        try:
            data = json.loads(text)
            issue_id = data["issue_id"]
            field_label = data["field_label"]
            value = data["value"]
        except Exception:
            if "|" in text:
                issue_id, field_label, value = text.split("|", 2)
            else:
                raise TypeError(
                    "fill_field_by_label requires 'issue_id|field_label|value' or JSON"
                )
    elif len(args) == 3:
        issue_id, field_label, value = args
    else:
        raise TypeError("fill_field_by_label expects issue_id, field_label and value")

    return fill_field_by_label_func(
        issue_id.strip(), field_label.strip(), value.strip()
    )


fill_field_by_label_tool = Tool(
    name="fill_field_by_label",
    func=_fill_field_by_label_wrapper,
    description=(
        "Set a field's value using the human readable label. Provide the issue key,"
        " the field label such as 'Definition Of Done' and the new value."
        " Returns the updated issue JSON."
    ),
)

update_issue_fields_tool = Tool(
    name="update_issue_fields",
    func=update_issue_fields_func,
    description=(
        "Update one or more fields on a Jira issue. "
        "Provide the issue key and a JSON string of field-value pairs. "
        "Returns the updated issue JSON."
    ),
)


# Collection of available Jira tools
jira_tools = [
    get_issue_by_id_tool,
    create_jira_issue_tool,
    get_issue_comments_tool,
    get_issue_history_tool,
    get_related_issues_tool,
    add_comment_to_issue_tool,
    fill_field_by_label_tool,
    update_issue_fields_tool,
]

__all__ = [
    "get_issue_by_id_tool",
    "create_jira_issue_tool",
    "get_issue_comments_tool",
    "get_issue_history_tool",
    "get_related_issues_tool",
    "add_comment_to_issue_tool",
    "fill_field_by_label_tool",
    "update_issue_fields_tool",
    "jira_tools",
]
