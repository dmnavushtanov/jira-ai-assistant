"""Minimal Jira REST client adapter."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import base64
import requests

logger = logging.getLogger(__name__)


def _build_auth_headers(username: str, api_token: str) -> Dict[str, str]:
    """Return headers for Jira basic auth using an API token."""
    token = base64.b64encode(f"{username}:{api_token}".encode()).decode()
    return {
        "Authorization": f"Basic {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


@dataclass
class JiraConfig:
    base_url: str
    username: str
    api_token: str


class JiraAPI:
    """Simple wrapper around the Jira REST API."""

    def __init__(self, config: JiraConfig):
        self.config = config
        self.session = requests.Session()
        # Normalize base URL to avoid double slashes
        self.config.base_url = self.config.base_url.rstrip("/")
        self.session.headers.update(
            _build_auth_headers(config.username, config.api_token)
        )

    def get_issue(self, issue_key: str) -> Dict[str, Any]:
        url = f"{self.config.base_url}/rest/api/3/issue/{issue_key}"
        logger.debug("Fetching issue %s", issue_key)
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.json()

    def transition_issue(self, issue_key: str, transition_id: str) -> None:
        url = f"{self.config.base_url}/rest/api/3/issue/{issue_key}/transitions"
        payload = {"transition": {"id": transition_id}}
        logger.debug("Transitioning issue %s with payload %s", issue_key, payload)
        resp = self.session.post(url, json=payload)
        resp.raise_for_status()

    def search_issues(self, jql: str) -> List[Dict[str, Any]]:
        """Return issues matching the JQL query."""
        url = f"{self.config.base_url}/rest/api/3/search"
        logger.debug("Searching issues with JQL: %s", jql)
        resp = self.session.get(url, params={"jql": jql})
        resp.raise_for_status()
        data = resp.json()
        return data.get("issues", [])

    def get_transitions(self, issue_key: str) -> List[Dict[str, Any]]:
        """Fetch available transitions for an issue."""
        url = f"{self.config.base_url}/rest/api/3/issue/{issue_key}/transitions"
        logger.debug("Fetching transitions for issue %s", issue_key)
        resp = self.session.get(url)
        resp.raise_for_status()
        data = resp.json()
        return data.get("transitions", [])

