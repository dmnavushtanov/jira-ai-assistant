import requests
from typing import Any, Dict, Optional


class JiraClient:
    """Simple Jira API client using API token authentication."""

    def __init__(self, base_url: str, email: str, api_token: str) -> None:
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.auth = (email, api_token)
        self.session.headers.update({"Accept": "application/json", "Content-Type": "application/json"})

    def request(self, method: str, endpoint: str, **kwargs: Any) -> requests.Response:
        """Send an HTTP request to the Jira API."""
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    def get_issue(self, issue_key: str) -> Dict[str, Any]:
        """Retrieve an issue by its key."""
        resp = self.request("GET", f"/rest/api/3/issue/{issue_key}")
        return resp.json()

    def create_issue(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new issue given a payload."""
        resp = self.request("POST", "/rest/api/3/issue", json=payload)
        return resp.json()

    def add_comment(self, issue_key: str, comment: str) -> Dict[str, Any]:
        """Add a comment to an issue."""
        resp = self.request(
            "POST",
            f"/rest/api/3/issue/{issue_key}/comment",
            json={"body": comment},
        )
        return resp.json()

    def search_issues(self, jql: str, **params: Any) -> Dict[str, Any]:
        """Search for issues using JQL."""
        payload = {"jql": jql}
        payload.update(params)
        resp = self.request("POST", "/rest/api/3/search", json=payload)
        return resp.json()
