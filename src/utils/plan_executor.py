"""Utility for executing multi-step Jira operations plans."""

from __future__ import annotations

from typing import Any, Dict
import json
import logging

logger = logging.getLogger(__name__)


class OperationsPlanExecutor:
    """Execute a plan of Jira operations sequentially."""

    def __init__(self, operations_agent: Any) -> None:
        self.operations = operations_agent

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

            if agent != "jira_operations":
                results[step_name] = f"Unknown agent {agent}"
                continue

            func = getattr(self.operations, action, None)
            if not callable(func):
                results[step_name] = f"Unknown action {action}"
                continue

            try:
                result = func(issue_key, **params, **kwargs)
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
