"""Agent for performing write operations on Jira issues."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.services.jira_service import (
    add_comment_to_issue_tool,
    create_jira_issue_tool,
    fill_field_by_label_tool,
    get_issue_transitions_tool,
    update_issue_fields_tool,
    transition_issue_tool,
)
from src.configs.config import load_config
from src.llm_clients import create_llm_client
from src.prompts import load_prompt
from src.utils import safe_format, parse_json_block

logger = logging.getLogger(__name__)
logger.debug("jira_operations module loaded")


class JiraOperationsAgent:
    """Agent that performs modifications on Jira issues."""

    def __init__(self, config_path: str | None = None) -> None:
        logger.debug(
            "Initializing JiraOperationsAgent with config_path=%s", config_path
        )
        self.config = load_config(config_path)
        self.client = create_llm_client(config_path)

        # Tools available to this agent
        self.tools = [
            add_comment_to_issue_tool,
            create_jira_issue_tool,
            fill_field_by_label_tool,
            get_issue_transitions_tool,
            update_issue_fields_tool,
            transition_issue_tool,
        ]

        self.plan_prompt = load_prompt("jira_operations.txt")
        self.transition_prompt = load_prompt("transition_choice.txt")

    def add_comment(self, issue_id: str, comment: str, **kwargs: Any) -> str:
        """Add ``comment`` to ``issue_id`` using the Jira API."""
        logger.info("Adding comment to %s", issue_id)
        payload = json.dumps({"issue_id": issue_id, "comment": comment})
        result_json = add_comment_to_issue_tool.run(payload)
        try:
            parsed = json.loads(result_json)
        except Exception:
            logger.debug("Failed to parse add_comment response")
            parsed = result_json
        logger.info("Comment added to %s", issue_id)
        return parsed

    def create_issue(
        self,
        summary: str,
        description: str,
        project_key: str,
        issue_type: str = "Task",
        *,
        parent_key: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Create a new Jira issue.

        ``parent_key`` is used when ``issue_type`` is ``Sub-task``.
        """
        logger.info("Creating issue in %s", project_key)
        result_json = create_jira_issue_tool.run(
            summary, description, project_key, issue_type, parent_key
        )
        try:
            return json.loads(result_json)
        except Exception:
            logger.debug("Failed to parse create_issue response")
            return result_json

    def update_fields(self, issue_id: str, fields_json: str, **kwargs: Any) -> str:
        """Update fields on ``issue_id``."""
        logger.info("Updating issue %s", issue_id)
        result_json = update_issue_fields_tool.run(issue_id, fields_json)
        try:
            return json.loads(result_json)
        except Exception:
            logger.debug("Failed to parse update_fields response")
            return result_json

    def fill_field_by_label(
        self, issue_id: str, field_label: str, value: str, **kwargs: Any
    ) -> str:
        """Set an issue field value using the field's label."""
        logger.info("Setting %s on %s", field_label, issue_id)
        payload = json.dumps(
            {
                "issue_id": issue_id,
                "field_label": field_label,
                "value": value,
            }
        )
        result_json = fill_field_by_label_tool.run(payload)
        try:
            return json.loads(result_json)
        except Exception:
            logger.debug("Failed to parse fill_field_by_label response")
            return result_json

    def _choose_transition(
        self, requested: str, transitions: list[dict[str, Any]], **kwargs: Any
    ) -> str | None:
        """Return the best matching transition using the LLM."""
        if not transitions or not self.transition_prompt:
            return None

        options = []
        for t in transitions:
            name = t.get("name") or t.get("to", {}).get("name") or t.get("id")
            if name:
                options.append(str(name))

        if not options:
            return None

        prompt = safe_format(
            self.transition_prompt,
            {"requested": requested, "options": ", ".join(options)},
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat_completion(messages, **kwargs)
        try:
            text = response.choices[0].message.content.strip()
        except Exception:
            try:
                text = response["choices"][0]["message"]["content"].strip()
            except Exception:
                logger.exception("Failed to parse transition selection response")
                return None

        if not text or text.upper() == "NONE":
            return None

        return text

    def transition_issue(
        self,
        issue_id: str,
        transition: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Move ``issue_id`` to a new workflow status.

        ``transition`` can be provided as a positional argument or via the
        ``transition_name`` keyword in ``kwargs`` to handle variations in plan
        parameters.

        The available transitions are retrieved first and ``transition`` is
        compared case-insensitively against the available names. If no direct
        match is found, the user is prompted with the available options so they
        can decide which status to use. The LLM may suggest a close match, but
        the issue is not transitioned automatically when the requested status is
        unavailable.
        """

        if transition is None:
            transition = kwargs.pop("transition_name", None)
        if not transition:
            raise TypeError(
                "transition_issue requires 'transition' or 'transition_name'"
            )

        logger.info("Transitioning %s using %s", issue_id, transition)

        transitions_json = get_issue_transitions_tool.run(issue_id)
        try:
            transitions = json.loads(transitions_json)
        except Exception:
            logger.debug("Failed to parse transitions response")
            transitions = []

        desired = transition.strip().lower()
        match = None

        def _normalize(name: str | None) -> str:
            return str(name or "").lower()

        # exact id or name match
        for t in transitions:
            tid = str(t.get("id"))
            name = t.get("name") or t.get("to", {}).get("name")
            if desired in {_normalize(tid), _normalize(name)}:
                match = t
                break

        if match is None:
            suggestion = self._choose_transition(transition, transitions, **kwargs)
            available = [
                str(t.get("name") or t.get("to", {}).get("name"))
                for t in transitions
                if t.get("name") or t.get("to", {}).get("name")
            ]
            logger.warning("Transition '%s' not available for %s", transition, issue_id)
            message = (
                f"Error: Transition '{transition}' is not available for {issue_id}. "
                f"Available statuses: {', '.join(available)}. "
            )
            if suggestion:
                message += (
                    f"Did you mean '{suggestion}'? Please specify the status to use."
                )
            else:
                message += "Which status should I use?"
            return message

        matched_name = str(match.get("name") or match.get("to", {}).get("name"))
        payload = json.dumps({"issue_id": issue_id, "transition": matched_name})
        result_json = transition_issue_tool.run(payload)
        try:
            return json.loads(result_json)
        except Exception:
            logger.debug("Failed to parse transition_issue response")
            return result_json

    # ------------------------------------------------------------------
    # Natural language operation handling
    # ------------------------------------------------------------------
    def _plan_operation(
        self, question: str, issue_id: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Return an action plan dict for ``question`` using the LLM."""
        if not self.plan_prompt:
            raise RuntimeError("Jira operations prompt not found")
        template = self.plan_prompt
        prompt = safe_format(
            template, {"question": question, "issue_id": issue_id or ""}
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat_completion(messages, **kwargs)
        try:
            text = response.choices[0].message.content.strip()
        except Exception:
            try:
                text = response["choices"][0]["message"]["content"].strip()
            except Exception:
                logger.exception("Failed to parse planning response")
                return {"action": "unknown"}
        plan = parse_json_block(text)
        if isinstance(plan, dict):
            return plan
        logger.debug("Plan not JSON: %s", text)
        return {"action": "unknown"}

    def operate(self, question: str, issue_id: str | None = None, **kwargs: Any) -> str:
        """Execute an operation described by ``question``."""
        plan = self._plan_operation(question, issue_id=issue_id, **kwargs)
        action = str(plan.get("action", "unknown")).lower()
        try:
            if action == "add_comment":
                issue = plan.get("issue_id") or issue_id
                comment = plan.get("comment")
                if not issue or not comment:
                    return "Missing issue_id or comment for add_comment"
                result = self.add_comment(str(issue), str(comment), **kwargs)
                return json.dumps(result)
            if action == "create_issue":
                summary = plan.get("summary")
                description = plan.get("description")
                project_key = plan.get("project_key")
                issue_type = plan.get("issue_type", "Task")
                if not summary or not description or not project_key:
                    return "Missing parameters for create_issue"
                result = self.create_issue(
                    str(summary),
                    str(description),
                    str(project_key),
                    str(issue_type),
                    **kwargs,
                )
                return json.dumps(result)
            if action == "update_fields":
                issue = plan.get("issue_id") or issue_id
                fields = plan.get("fields")
                if not issue or not fields:
                    return "Missing issue_id or fields for update_fields"
                fields_json = json.dumps(fields)
                result = self.update_fields(str(issue), fields_json, **kwargs)
                return json.dumps(result)
            if action == "fill_field_by_label":
                issue = plan.get("issue_id") or issue_id
                label = plan.get("field_label")
                value = plan.get("value")
                if not issue or not label:
                    return "Missing issue_id or field_label for fill_field_by_label"
                result = self.fill_field_by_label(
                    str(issue), str(label), str(value), **kwargs
                )
                return json.dumps(result)
            if action == "transition_issue":
                issue = plan.get("issue_id") or issue_id
                transition = plan.get("transition")
                if not issue or not transition:
                    return "Missing issue_id or transition for transition_issue"
                result = self.transition_issue(str(issue), str(transition), **kwargs)
                return json.dumps(result)
        except Exception:
            logger.exception("Failed to execute operation")
            return "Error performing Jira operation"
        return "Unknown operation"


__all__ = ["JiraOperationsAgent"]
