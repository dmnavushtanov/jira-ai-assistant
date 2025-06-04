from langchain.tools import Tool

# --- Tool for getting issue details by ID ---
def get_issue_by_id_func(issue_id: str) -> str:
    """Fetches details for a given Jira issue ID."""
    # Call your Jira API here
    # For example, using the mcp_Atlassian_MCP_via_CLI_jira_get_issue tool
    # response = mcp_Atlassian_MCP_via_CLI_jira_get_issue(issue_key=issue_id)
    # return response 
    return f"Details for issue {issue_id}"

get_issue_by_id_tool = Tool(
    name="get_issue_by_id",
    func=get_issue_by_id_func,
    description="Useful for when you need to get the details of a specific Jira issue. Input should be the Jira issue ID (e.g., 'PROJ-123')."
)

# --- Tool for creating a new Jira issue ---
def create_jira_issue_func(summary: str, description: str, project_key: str, issue_type: str = 'Task') -> str:
    """Creates a new Jira issue."""
    # Call Jira POST /issue
    # For example, using the mcp_Atlassian_MCP_via_CLI_jira_create_issue tool
    # response = mcp_Atlassian_MCP_via_CLI_jira_create_issue(
    #     project_key=project_key,
    #     summary=summary,
    #     description=description,
    #     issue_type=issue_type
    # )
    # return response
    return f"Issue '{summary}' created in project {project_key} with description: {description}"

create_jira_issue_tool = Tool(
    name="create_jira_issue",
    func=create_jira_issue_func,
    description="Useful for when you need to create a new Jira issue. Requires summary, description, and project_key. Optionally, an issue_type can be provided (default is 'Task')."
)

# --- Placeholder for get comments ---
def get_issue_comments_func(issue_id: str) -> str:
    """Fetches comments for a given Jira issue ID."""
    # Call your Jira API here to get comments
    # For example, using mcp_Atlassian_MCP_via_CLI_jira_get_issue with expand='comment'
    # or a more specific comment fetching tool if available.
    return f"Comments for issue {issue_id}"

get_issue_comments_tool = Tool(
    name="get_issue_comments",
    func=get_issue_comments_func,
    description="Useful for when you need to get the comments for a specific Jira issue. Input should be the Jira issue ID (e.g., 'PROJ-123')."
)

# --- Placeholder for get history/changelog ---
def get_issue_history_func(issue_id: str) -> str:
    """Fetches the history (changelog) for a given Jira issue ID."""
    # Call your Jira API here to get issue history
    # For example, mcp_Atlassian_MCP_via_CLI_jira_get_issue with expand='changelog'
    # or mcp_Atlassian_MCP_via_CLI_jira_batch_get_changelogs
    return f"History for issue {issue_id}"

get_issue_history_tool = Tool(
    name="get_issue_history",
    func=get_issue_history_func,
    description="Useful for when you need to get the change history for a specific Jira issue. Input should be the Jira issue ID (e.g., 'PROJ-123')."
)

# You can add more tools here following the same pattern

# To make these tools usable, you might want to collect them in a list:
jira_tools = [
    get_issue_by_id_tool,
    create_jira_issue_tool,
    get_issue_comments_tool,
    get_issue_history_tool,
]

# Example of how you might use one (for testing purposes, not part of the service itself)
if __name__ == '__main__':
    print(get_issue_by_id_tool.run("PROJ-123"))
    print(create_jira_issue_tool.run({"summary": "New Task", "description": "Detailed description here", "project_key": "PROJ"}))
    print(get_issue_comments_tool.run("PROJ-124"))
    print(get_issue_history_tool.run("PROJ-125")) 