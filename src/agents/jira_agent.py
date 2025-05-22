"""LangChain agent placeholder."""

from langchain.agents import Tool
from langchain.agents import initialize_agent

from src.adapters.jira_api import JiraAPI, JiraConfig
from llm.llm_wrapper import get_llm
import config

jira_client = JiraAPI(JiraConfig(config.JIRA_URL, config.JIRA_USERNAME, config.JIRA_API_TOKEN))


def describe_issue(issue_key: str) -> str:
    issue = jira_client.get_issue(issue_key)
    fields = issue.get("fields", {})
    summary = fields.get("summary", "")
    description = fields.get("description", "")
    return f"Summary: {summary}\nDescription: {description}"


def search_issues(jql: str) -> str:
    issues = jira_client.search_issues(jql)
    keys = [issue.get("key") for issue in issues]
    return ", ".join(keys)


def transition_issue(issue_key: str, transition_id: str) -> str:
    jira_client.transition_issue(issue_key, transition_id)
    return "Transition performed"


def get_agent():
    llm = get_llm()
    tools = [
        Tool(name="Describe Issue", func=describe_issue, description="Describe Jira issue"),
        Tool(name="Search Issues", func=search_issues, description="Search issues by JQL"),
        Tool(name="Transition Issue", func=transition_issue, description="Perform a transition on an issue"),
    ]
    return initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
