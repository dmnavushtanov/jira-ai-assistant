"""Utility for executing multi-step Jira operations plans."""

from __future__ import annotations

from typing import Any, Dict
import json
import logging

from src.services.jira_service import get_issue_by_id_tool

logger = logging.getLogger(__name__)


class OperationsPlanExecutor:
    """Execute a plan of Jira operations sequentially."""

    def __init__(
        self,
        operations_agent: Any,
        validator_agent: Any | None = None,
        test_agent: Any | None = None,
        insights_agent: Any | None = None,
    ) -> None:
        self.operations = operations_agent
        self.validator = validator_agent
        self.tester = test_agent
        self.insights = insights_agent

    def _lookup(self, value: str, results: Dict[str, Any]) -> Any:
        """Resolve "$stepN" or "$stepN.field" references."""
        if not value.startswith("$step"):
            return value
        parts = value[1:].split(".")  # drop leading '$'
        step_key = parts[0]
        data = results.get(step_key)
        if data is None:
            return None
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                if len(parts) == 1:
                    return data
                return None
        for part in parts[1:]:
            if isinstance(data, dict):
                data = data.get(part)
            else:
                return None
        return data

    def execute(self, plan: Dict[str, Any], issue_key: str, **kwargs: Any) -> Dict[str, Any]:
        """Run ``plan`` against ``issue_key`` and return step results."""
        steps = plan.get("plan") or []
        if not isinstance(steps, list):
            return {"error": "Plan did not contain actionable steps"}

        logger.info("Executing %d-step plan for issue %s", len(steps), issue_key)
        results: Dict[str, Any] = {}

        for i, step in enumerate(steps, 1):
            step_name = f"step_{i}"
            agent = step.get("agent")
            action = step.get("action")
            params = step.get("parameters") or {}

            if isinstance(params, dict):
                resolved = {}
                for k, v in params.items():
                    if isinstance(v, str):
                        resolved[k] = self._lookup(v, results)
                    else:
                        resolved[k] = v
                params = resolved

            logger.info("Step %d/%d: %s.%s with params %s", i, len(steps), agent, action, params)

            if agent == "jira_operations":
                target = self.operations
                call_args = (issue_key,)
            elif agent == "api_validator" and self.validator is not None:
                target = self.validator
                issue_json = get_issue_by_id_tool.run(issue_key)
                try:
                    issue_data = json.loads(issue_json)
                except Exception:
                    issue_data = issue_json
                call_args = (issue_data,)
            elif agent == "test_agent" and self.tester is not None:
                target = self.tester
                issue_json = get_issue_by_id_tool.run(issue_key)
                try:
                    issue = json.loads(issue_json)
                except Exception:
                    issue = {}
                fields = issue.get("fields", {}) if isinstance(issue, dict) else {}
                summary = fields.get("summary", "")
                description = fields.get("description", "")
                question = params.pop("question", "")
                text = f"{summary}\n{description}\n{question}"
                call_args = (text, None)
            elif agent == "issue_insights" and self.insights is not None:
                target = self.insights
                question = params.pop("question", "Tell me about this issue")
                call_args = (issue_key, question)
            else:
                results[step_name] = f"Unknown agent {agent}"
                continue

            func = getattr(target, action, None)
            if not callable(func):
                results[step_name] = f"Unknown action {action}"
                continue

            try:
                result = func(*call_args, **params, **kwargs)
                results[step_name] = result
                if isinstance(result, str) and result.startswith("Error"):
                    logger.warning("Step %d failed: %s", i, result)
                else:
                    logger.info("Step %d completed successfully", i)
            except Exception as exc:
                logger.exception("Failed step %s", action)
                results[step_name] = f"Failed {action}: {exc}"

        return results


__all__ = ["OperationsPlanExecutor"]
