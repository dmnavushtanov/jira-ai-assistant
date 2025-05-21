"""Minimal Jira REST client adapter."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


def _build_auth_headers(api_token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_token}", "Accept": "application/json", "Content-Type": "application/json"}


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
        self.session.headers.update(_build_auth_headers(config.api_token))

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

