"""Router agent that directs questions to the appropriate workflow."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

try:
    from langchain.tools import Tool  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Tool = None

try:
    from langchain.memory import (
        ConversationBufferWindowMemory,
        ConversationSummaryMemory,
        CombinedMemory,
    )
    from langchain.agents import initialize_agent, AgentType  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ConversationBufferWindowMemory = None
    ConversationSummaryMemory = None
    CombinedMemory = None
    initialize_agent = None  # type: ignore
    AgentType = None  # type: ignore

from src.configs.config import load_config
from src.llm_clients import create_langchain_llm
from src.agents.classifier import ClassifierAgent
from src.agents.issue_insights import IssueInsightsAgent
from src.agents.api_validator import ApiValidatorAgent
from src.agents.jira_operations import JiraOperationsAgent
from src.agents.test_agent import TestAgent, EXISTING_TESTS_MSG
from src.agents.issue_creator import IssueCreatorAgent
from src.agents.planning import PlanningAgent
from src.prompts import load_prompt
from src.services.jira_service import get_issue_by_id_tool
from src.utils import (
    safe_format,
    JiraContextMemory,
    parse_json_block,
    normalize_newlines,
)

from jira import JIRAError
from openai import OpenAIError

logger = logging.getLogger(__name__)
logger.debug("router_agent module loaded")


class RouterAgent:
    """Agent that routes questions to insights or validation workflows."""

    def __init__(self, config_path: str | None = None) -> None:
        logger.debug("Initializing RouterAgent with config_path=%s", config_path)
        self.config = load_config(config_path)
        self.classifier = ClassifierAgent(config_path)
        self.validator = ApiValidatorAgent(config_path)
        self.insights = IssueInsightsAgent(config_path)
        self.operations = JiraOperationsAgent(config_path)
        self.tester = TestAgent(config_path)
        self.creator = IssueCreatorAgent(config_path)
        self.planner = PlanningAgent(config_path)
        self.use_memory = self.config.conversation_memory
        # global conversation context is enabled when memory is enabled
        self.global_memory_enabled = self.use_memory
        self.max_history = self.config.max_questions_to_remember
        self._pending_confirmation: str | None = None
        self._confirm_issue: str | None = None
        self._confirm_comment: str | None = None
        if self.use_memory:
            if ConversationBufferWindowMemory is None:
                logger.warning("LangChain not installed; conversation memory disabled")
                self.use_memory = False
                self.memory = None
            else:
                base_mem = ConversationBufferWindowMemory(
                    k=self.max_history,
                    return_messages=True,
                )
                if (
                    self.max_history > 3
                    and CombinedMemory is not None
                    and ConversationSummaryMemory is not None
                ):
                    llm = create_langchain_llm(config_path)
                    summary_mem = ConversationSummaryMemory(
                        llm=llm,
                        return_messages=True,
                    )
                    self.memory = CombinedMemory(memories=[base_mem, summary_mem])
                else:
                    self.memory = base_mem
        else:
            self.memory = None
        self.session_memory = JiraContextMemory()
        if self.config.projects:
            pattern = "|".join(re.escape(p) for p in self.config.projects)
        else:
            pattern = r"[A-Za-z][A-Za-z0-9]+"
        self.issue_re = re.compile(rf"(?:{pattern})-\d+", re.IGNORECASE)
        self.history_prompt = load_prompt("needs_history.txt")
        self.intent_prompt = load_prompt("intent_classifier.txt")

        # expose helper agents for external orchestration
        intent_tool = None
        if Tool is not None:
            intent_tool = Tool(
                name="classify_intent",
                func=self._classify_intent,
                description=(
                    "Classify user intent as VALIDATE, OPERATE, INSIGHT, TEST, CREATE, or UNKNOWN"
                ),
            )

        self.tools = {
            "classifier": self.classifier,
            "validator": self.validator,
            "insights": self.insights,
            "operations": self.operations,
            "tester": self.tester,
            "creator": self.creator,
        }
        if intent_tool:
            self.tools["classify_intent"] = intent_tool

        # setup optional LangChain executor for the insight workflow
        self.insight_executor = None
        if (
            self.use_memory
            and self.memory is not None
            and initialize_agent is not None
            and AgentType is not None
        ):
            llm = create_langchain_llm(config_path)
            self.insight_executor = initialize_agent(
                tools=list(self.tools.values()),
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                memory=self.memory,
                handle_parsing_errors=True,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_issue_id(self, text: str) -> str | None:
        match = self.issue_re.search(text)
        if match:
            issue_id = match.group(0).upper()
            logger.debug("Extracted issue id %s from text", issue_id)
            return issue_id
        logger.warning("No issue id found in text: %s", text)
        return None

    def _extract_project_key(self, text: str) -> str | None:
        """Return the first configured project key found in ``text``."""
        if self.config.projects:
            for key in self.config.projects:
                pattern = rf"\b{re.escape(key)}\b"
                if re.search(pattern, text, re.IGNORECASE):
                    logger.debug("Extracted project key %s from text", key)
                    return key.upper()
        logger.warning("No project key found in text: %s", text)
        return None

    def _classify_intent(self, question: str, **kwargs: Any) -> str:
        """Return the intent label for ``question`` using the classifier agent."""
        if not self.intent_prompt:
            raise RuntimeError("Intent classification prompt not found")
        prompt = safe_format(self.intent_prompt, {"question": question})
        label = self.classifier.classify(prompt, **kwargs)
        result = str(label).strip().upper()
        logger.debug("Intent classification result: %s", result)
        return result

    def _classify_intent_with_context(self, question: str, **kwargs: Any) -> str:
        """Classify intent with conversation history for better context."""
        if not self.intent_prompt:
            raise RuntimeError("Intent classification prompt not found")

        if len(question.strip().split()) <= self.max_history:
            history = self.prepare_conversation_history(limit=self.max_history)
            if history:
                enhanced_question = f"Conversation context:\n{history}\n\nCurrent input: {question}"
                prompt = safe_format(self.intent_prompt, {"question": enhanced_question})
            else:
                prompt = safe_format(self.intent_prompt, {"question": question})
        else:
            prompt = safe_format(self.intent_prompt, {"question": question})

        label = self.classifier.classify(prompt, **kwargs)
        result = str(label).strip().upper()
        logger.debug("Intent classification result: %s", result)
        return result

    def _should_validate(self, question: str, **kwargs: Any) -> bool:
        """Return ``True`` if the detected intent is VALIDATE."""
        return self._classify_intent(question, **kwargs).startswith("VALIDATE")

    def _needs_history(self, question: str, **kwargs: Any) -> bool:
        """Return True if the LLM determines the changelog is required."""
        if not self.history_prompt:
            raise RuntimeError("History check prompt not found")
        prompt = safe_format(self.history_prompt, {"question": question})
        label = self.classifier.classify(prompt, **kwargs)
        result = str(label).strip().upper().startswith("HISTORY")
        logger.debug("Needs history: %s (label=%s)", result, label)
        return result

    def _check_history_limit(self) -> str | None:
        """Reset chat history when it grows too large.

        When LangChain memory isn't available the CLI prompt normally asks the
        user to start a new conversation once ``max_history`` is exceeded. In a
        non-interactive setting there is no opportunity for user input. This
        helper automatically clears :class:`JiraContextMemory` and returns a
        notice to the caller when the limit has been reached.
        """
        if self.use_memory or self.session_memory is None:
            return None
        if len(self.session_memory.chat_history) >= 2 * self.max_history:
            logger.info("History limit reached; resetting session memory")
            self.session_memory.clear()
            return "Starting a new conversation due to length."
        return None

    def prepare_conversation_history(self, limit: int | None = None) -> str:
        """Return recent conversation history for context injection."""
        if not self.global_memory_enabled:
            return ""

        limit = limit or self.max_history

        if self._try_primary_memory(limit):
            return self._format_memory_messages(limit)

        if self._try_session_memory(limit):
            return self._format_session_history(limit)

        return ""

    def _try_primary_memory(self, limit: int) -> bool:
        if not (self.use_memory and self.memory is not None):
            return False
        try:
            _ = self.memory.chat_memory.messages[-1]
            return True
        except Exception:
            logger.exception("Failed to access primary memory")
            return False

    def _try_session_memory(self, limit: int) -> bool:
        if not (self.session_memory and hasattr(self.session_memory, "chat_history")):
            return False
        return bool(self.session_memory.chat_history)

    def _format_memory_messages(self, limit: int) -> str:
        try:
            messages = self.memory.chat_memory.messages[-limit:]
        except Exception:
            logger.exception("Failed to read conversation memory")
            return ""
        return "\n".join(
            f"{'User' if msg.type == 'human' else 'Assistant'}: {msg.content}"
            for msg in messages
        )

    def _format_session_history(self, limit: int) -> str:
        try:
            recent = self.session_memory.chat_history[-limit:]
        except Exception:
            logger.exception("Failed to read session history")
            return ""
        history_parts = []
        for i, content in enumerate(recent):
            role = "User" if i % 2 == 0 else "Assistant"
            history_parts.append(f"{role}: {content}")
        return "\n".join(history_parts)

    def _classify_and_validate(self, issue_id: str, history: str = "", **kwargs: Any) -> str:
        logger.info("Running classification/validation flow for %s", issue_id)
        issue_json = get_issue_by_id_tool.run(issue_id)
        issue = json.loads(issue_json)
        fields = issue.get("fields", {})
        base_prompt = safe_format(
            load_prompt("classifier.txt"),
            {
                "summary": fields.get("summary", ""),
                "description": fields.get("description", ""),
            },
        )

        prompt = (
            f"Conversation context:\n{history}\n\n{base_prompt}" if history else base_prompt
        )
        label = self.classifier.classify(prompt, **kwargs)
        logger.debug("Classifier label: %s", label)
        if str(label).upper().startswith("API"):
            return self.validator.validate(issue, history=history, **kwargs)
        return "Issue not API related"

    def _handle_validation_result(self, issue_id: str, result: str) -> bool:
        """Post validation results to Jira if a comment is present.

        Returns ``True`` if a comment was successfully posted.
        """
        try:
            data = json.loads(result)
        except Exception:
            data = parse_json_block(result)
        if data is None:
            logger.debug("Validation result not JSON")
            comment = None
        else:
            comment = data.get("jira_comment") if isinstance(data, dict) else None
        if comment:
            comment = normalize_newlines(comment)
            if not self.config.write_comments_to_jira:
                logger.info(
                    "write_comments_to_jira disabled; skipping comment to %s",
                    issue_id,
                )
                return False
            if self.config.ask_for_confirmation:
                self._confirm_issue = issue_id
                self._confirm_comment = comment
                self._pending_confirmation = (
                    f"Post validation comment to {issue_id}? (yes/no)"
                )
                logger.info("Awaiting user confirmation to comment on %s", issue_id)
                return False
            try:
                self.operations.add_comment(issue_id, comment)
                logger.info("Posted validation comment to %s", issue_id)
                return True
            except Exception:
                logger.exception("Failed to add validation comment to %s", issue_id)
        return False

    def _generate_test_cases(
        self, issue_id: str, question: str, history: str = "", **kwargs: Any
    ) -> str | None:
        """Return test cases string generated from Jira ``issue_id``.

        ``None`` or a short message is returned when test cases are already
        present on the issue.
        """
        try:
            issue_json = get_issue_by_id_tool.run(issue_id)
            issue = json.loads(issue_json)
            fields = issue.get("fields", {})
            summary = fields.get("summary", "") or ""
            description = fields.get("description", "") or ""
            text = f"{summary}\n{description}\n{question}"
            tests = self.tester.create_test_cases(text, None, history=history, **kwargs)
            return tests
        except JIRAError:
            logger.exception("Jira error while fetching issue %s", issue_id)
            return f"Jira issue {issue_id} does not exist or you do not have permission to access it."
        except Exception:
            logger.exception("Failed to generate test cases")
            return "Not enough information to generate test cases."

    def _add_tests_to_description(self, issue_id: str, tests: str) -> bool:
        """Append ``tests`` to the Description field of ``issue_id``."""
        try:
            tests = normalize_newlines(tests)
            issue_json = get_issue_by_id_tool.run(issue_id)
            issue = json.loads(issue_json)
            fields = issue.get("fields", {})
            desc = fields.get("description", "") or ""
            new_desc = desc + ("\n\n" if desc else "") + tests
            self.operations.fill_field_by_label(issue_id, "Description", new_desc)
            logger.info("Updated description for %s", issue_id)
            return True
        except Exception:
            logger.exception("Failed to update description for %s", issue_id)
            return False

    def _generate_tests(self, issue_id: str, question: str, history: str = "", **kwargs: Any) -> str:
        """Return generated test cases and update Jira when possible."""
        enhanced_question = f"{history}\n\nCurrent request: {question}" if history else question
        tests = self._generate_test_cases(issue_id, enhanced_question, history=history, **kwargs)

        if tests is None or tests == EXISTING_TESTS_MSG:
            return EXISTING_TESTS_MSG
          
        cleaned = normalize_newlines(tests)
        if cleaned and not cleaned.lower().startswith("not enough"):
            if self._add_tests_to_description(issue_id, cleaned):
                cleaned += "\n\nDescription updated with generated tests."
        return cleaned

    def _execute_operations_plan(self, plan: dict[str, Any], history: str = "", **kwargs: Any) -> str:
        """Execute a multi-step Jira operations plan."""
        issue_key = plan.get("issue_key") or self.session_memory.current_issue
        if not issue_key:
            return (
                "I'm sorry, I couldn't determine which Jira issue to use. "
                "Could you specify the issue key?"
            )
        steps = plan.get("plan") or []
        if not isinstance(steps, list):
            return "Plan did not contain actionable steps"
        results = []
        for step in steps:
            agent = step.get("agent")
            action = step.get("action")
            params = step.get("parameters") or {}
            if agent != "jira_operations":
                results.append(f"Unknown agent {agent}")
                continue
            func = getattr(self.operations, action, None)
            if not callable(func):
                results.append(f"Unknown action {action}")
                continue
            try:
                result = func(issue_key, **params, **kwargs)
                if isinstance(result, str) and result.startswith("Error"):
                    results.append(f"Failed {action}: {result}")
                else:
                    results.append(f"Executed {action} successfully")
            except Exception as exc:
                logger.exception("Failed step %s", action)
                results.append(f"Failed {action}: {exc}")
        return "\n".join(results)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ask(self, question: str, **kwargs: Any) -> str:
        """Route ``question`` to the appropriate workflow."""
        logger.info("Router received question: %s", question)
        used_executor = False

        notice = self._check_history_limit()

        if self._pending_confirmation:
            user_reply = question.strip().lower()
            issue = self._confirm_issue
            comment = self._confirm_comment
            self._pending_confirmation = None
            self._confirm_issue = None
            self._confirm_comment = None
            if user_reply in ("y", "yes") and issue and comment:
                try:
                    self.operations.add_comment(issue, comment)
                    answer = "✅ Comment posted."
                except Exception:
                    logger.exception("Failed to post confirmation comment")
                    answer = "Failed to add comment."
            else:
                answer = "🚫 Operation cancelled."
            if self.use_memory and self.memory is not None:
                self.memory.save_context({"input": question}, {"output": answer})
            self.session_memory.save_context({"input": question}, {"output": answer})
            return answer

        issue_id = self._extract_issue_id(question)
        if issue_id:
            self.session_memory.current_issue = issue_id
        else:
            issue_id = self.session_memory.current_issue

        conversation_history = self.prepare_conversation_history()

        try:
            intent = self._classify_intent_with_context(question, **kwargs)
            if intent.startswith("OPERATE"):
                logger.info("Routing to operations workflow")
                plan = self.planner.generate_plan(
                    question, {"issue_key": issue_id or ""}, **kwargs
                )
                if plan.get("plan"):
                    answer = self._execute_operations_plan(plan, history=conversation_history, **kwargs)
                else:
                    answer = self.operations.operate(
                        question, issue_id=issue_id, history=conversation_history, **kwargs
                    )
            elif intent.startswith("CREATE"):
                logger.info("Routing to issue creation workflow")
                project_key = self._extract_project_key(question)
                if not project_key and self.config.projects:
                    project_key = self.config.projects[0]
                if not project_key:
                    answer = "Please specify a Jira project key"
                else:
                    answer = self.creator.create_issue(question, project_key, history=conversation_history, **kwargs)
            else:
                if not issue_id:
                    answer = (
                        "I'm sorry, I didn't catch an issue key in your question. "
                        "Could you specify which Jira issue you mean?"
                    )
                    if self.use_memory and self.memory is not None:
                        self.memory.save_context(
                            {"input": question}, {"output": answer}
                        )
                    self.session_memory.save_context(
                        {"input": question}, {"output": answer}
                    )
                    return answer
                if intent.startswith("VALIDATE"):
                    logger.info("Routing to validation workflow")
                    answer = self._classify_and_validate(issue_id, history=conversation_history, **kwargs)
                    comment_posted = self._handle_validation_result(issue_id, answer)
                    if self._pending_confirmation:
                        answer = self._pending_confirmation
                        if self.use_memory and self.memory is not None:
                            self.memory.save_context(
                                {"input": question}, {"output": answer}
                            )
                        self.session_memory.save_context(
                            {"input": question}, {"output": answer}
                        )
                        return answer
                    if comment_posted:
                        answer += "\n\nValidation summary posted as a Jira comment."
                elif intent.startswith("TEST"):
                    logger.info("Routing to test generation workflow")
                    answer = self._generate_tests(issue_id, question, history=conversation_history, **kwargs)
                else:
                    logger.info("Routing to general insights workflow")
                    if self.insight_executor is not None:
                        answer = self.insight_executor.run(question)
                        used_executor = True
                    else:
                        answer = self.insights.ask(issue_id, question, **kwargs)
        except JIRAError:
            logger.exception("Jira error while fetching issue %s", issue_id)
            answer = f"Sorry, I couldn't find the Jira issue {issue_id}. Please check the key and try again."
        except OpenAIError:
            logger.exception("OpenAI API error")
            answer = (
                "I'm having trouble communicating with the language model right now. "
                "Please try again later."
            )
        except RuntimeError as exc:
            logger.exception("Runtime error while processing question")
            msg = str(exc)
            if "validation prompt" in msg.lower():
                answer = f"Sorry, {msg}"
            else:
                answer = msg
        except ValueError as exc:
            logger.exception("Value error while processing question")
            answer = str(exc)
        except Exception:
            logger.exception("Unexpected error while processing question")
            answer = "Sorry, something went wrong while handling your request."
        if self.use_memory and self.memory is not None and not used_executor:
            self.memory.save_context({"input": question}, {"output": answer})
        if notice:
            answer = f"{notice}\n\n{answer}"
        self.session_memory.save_context({"input": question}, {"output": answer})
        return answer


__all__ = ["RouterAgent"]
