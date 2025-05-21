"""LangChain agent placeholder."""

from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

from adapters.jira_api import JiraAPI, JiraConfig
import config

jira_client = JiraAPI(JiraConfig(config.JIRA_URL, config.JIRA_USERNAME, config.JIRA_API_TOKEN))


def describe_issue(issue_key: str) -> str:
    issue = jira_client.get_issue(issue_key)
    fields = issue.get("fields", {})
    summary = fields.get("summary", "")
    description = fields.get("description", "")
    return f"Summary: {summary}\nDescription: {description}"


def get_agent():
    llm = OpenAI(api_key=config.OPENAI_API_KEY)
    tools = [
        Tool(name="Describe Issue", func=describe_issue, description="Describe Jira issue")
    ]
    return initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
