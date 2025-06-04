import json
import os
import requests


def get_jira_client():
    """Return a simple Jira REST client using environment variables."""
    base_url = os.getenv("JIRA_BASE_URL")
    email = os.getenv("JIRA_EMAIL")
    token = os.getenv("JIRA_API_TOKEN")
    if not all([base_url, email, token]):
        raise RuntimeError(
            "JIRA_BASE_URL, JIRA_EMAIL and JIRA_API_TOKEN must be set in the environment"
        )
    return base_url.rstrip("/"), (email, token)


def get_issue(issue_key: str) -> dict:
    base_url, auth = get_jira_client()
    url = f"{base_url}/rest/api/3/issue/{issue_key}"
    response = requests.get(url, auth=auth)
    response.raise_for_status()
    return response.json()


def main() -> None:
    issue_id = input("Enter Jira issue ID: ").strip()
    try:
        issue = get_issue(issue_id)
    except Exception as exc:
        print(f"Failed to fetch issue {issue_id}: {exc}")
        return
    print(json.dumps(issue, indent=2))


if __name__ == "__main__":
    main()
