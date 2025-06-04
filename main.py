import json
import os
from dotenv import load_dotenv
from src.jira_client import JiraClient

# Load environment variables from .env file (force reload)
load_dotenv(override=True)


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
        
        print("\nSearching for recent issues you can access...")
        issues = client.search_issues("ORDER BY updated DESC", maxResults=10)
        
        if issues:
            print(f"Found {len(issues)} accessible issues:")
            projects_found = set()
            for issue in issues:
                project_key = issue['fields']['project']['key']
                projects_found.add(project_key)
                print(f"  {issue['key']} ({project_key}): {issue['fields']['summary'][:50]}...")
            
            print(f"\nAccessible projects: {', '.join(sorted(projects_found))}")
        else:
            print("No issues found")
        
    except Exception as exc:
        print(f"Failed to search issues: {exc}")
        return
    
    print("\n" + "-" * 60)
    issue_id = input("Enter Jira issue ID: ").strip()
    
    try:
        issue = client.get_issue(issue_id)
        print("\nIssue found:")
        print(f"Key: {issue['key']}")
        print(f"Project: {issue['fields']['project']['key']} - {issue['fields']['project']['name']}")
        print(f"Summary: {issue['fields']['summary']}")
        print(f"Status: {issue['fields']['status']['name']}")
        print(f"Assignee: {issue['fields']['assignee']['displayName'] if issue['fields']['assignee'] else 'Unassigned'}")
        print(f"Reporter: {issue['fields']['reporter']['displayName']}")
        print(f"Created: {issue['fields']['created']}")
        print(f"Updated: {issue['fields']['updated']}")
    except Exception as exc:
        print(f"Failed to fetch issue {issue_id}: {exc}")


if __name__ == "__main__":
    main()
