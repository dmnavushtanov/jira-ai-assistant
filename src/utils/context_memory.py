from __future__ import annotations

"""Lightweight memory for tracking the current Jira issue in a conversation."""

from typing import Dict, Any, Optional, List
import re

try:
    from langchain.memory import BaseMemory
except Exception:  # pragma: no cover - langchain optional
    BaseMemory = object  # type: ignore[misc, assignment]


class JiraContextMemory(BaseMemory):
    """Simple conversation memory keeping track of a Jira issue id."""

    def __init__(self) -> None:
        self.current_issue: Optional[str] = None
        self.chat_history: List[str] = []

    # ------------------------------------------------------------------
    # BaseMemory API
    # ------------------------------------------------------------------
    @property
    def memory_variables(self) -> list[str]:
        return ["current_issue", "chat_history"]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return stored variables for prompt formatting."""
        return {
            "current_issue": self.current_issue or "",
            "chat_history": "\n".join(self.chat_history),
        }

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Update memory based on the latest interaction."""
        input_text = inputs.get("input", "")
        output_text = outputs.get("output", "")

        found_key = self.extract_jira_key(input_text)
        if found_key:
            self.current_issue = found_key

        if "forget" in input_text.lower():
            self.current_issue = None

        self.chat_history.append(f"Human: {input_text}")
        self.chat_history.append(f"AI: {output_text}")

    def clear(self) -> None:
        """Reset memory state."""
        self.current_issue = None
        self.chat_history = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def extract_jira_key(text: str) -> Optional[str]:
        match = re.search(r"\b[A-Z]+-\d+\b", text)
        return match.group(0) if match else None


__all__ = ["JiraContextMemory"]
