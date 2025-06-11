"""Router agent that directs questions to the appropriate workflow."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

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
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - optional dependency
    ConversationBufferWindowMemory = None
    ConversationSummaryMemory = None
    CombinedMemory = None
    ChatOpenAI = None

from src.configs.config import load_config
from src.agents.classifier import ClassifierAgent
from src.agents.issue_insights import IssueInsightsAgent
from src.agents.api_validator import ApiValidatorAgent
from src.agents.jira_operations import JiraOperationsAgent
from src.agents.test_agent import TestAgent
from src.agents.issue_creator import IssueCreatorAgent
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
        self.use_memory = self.config.conversation_memory
        self.max_history = self.config.max_questions_to_remember
        if self.use_memory:
            if None in (
                ConversationBufferWindowMemory,
                CombinedMemory,
                ChatOpenAI,
            ):
                logger.warning("LangChain not installed; conversation memory disabled")
                self.use_memory = False
                self.memory = None
            else:
                if self.max_history > 3:
                    llm = ChatOpenAI(
                        model=self.config.openai_model,
                        api_key=self.config.openai_api_key,
                    )
                    buffer_mem = ConversationBufferWindowMemory(
                        k=self.max_history,
                        return_messages=True,
                    )
                    summary_mem = ConversationSummaryMemory(
                        llm=llm,
                        return_messages=True,
                    )
                    self.memory = CombinedMemory(memories=[buffer_mem, summary_mem])
                else:
                    self.memory = ConversationBufferWindowMemory(
                        k=self.max_history,
                        return_messages=True,
                    )
        else:
            self.memory = None
        self.session_memory = JiraContextMemory()
        if self.config.projects:
            pattern = "|".join(re.escape(p) for p in self.config.projects)
        else:
            pattern = r"[A-Za-z][A-Za-z0-9]+"
        self.issue_re = re.compile(rf"(?:{pattern})-\d+", re.IGNORECASE)
        self.router_prompt = load_prompt("router.txt")
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

    def _classify_and_validate(self, issue_id: str, **kwargs: Any) -> str:
        logger.info("Running classification/validation flow for %s", issue_id)
        issue_json = get_issue_by_id_tool.run(issue_id)
        issue = json.loads(issue_json)
        fields = issue.get("fields", {})
        prompt = safe_format(
            load_prompt("classifier.txt"),
            {
                "summary": fields.get("summary", ""),
                "description": fields.get("description", ""),
            },
        )
        label = self.classifier.classify(prompt, **kwargs)
        logger.debug("Classifier label: %s", label)
        if str(label).upper().startswith("API"):
            return self.validator.validate(issue, **kwargs)
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
                from src.utils import confirm_action

                if not confirm_action(
                    f"Should I post the suggested comment to {issue_id}?"
                ):
                    logger.info("User declined to add comment to %s", issue_id)
                    return False
            try:
                self.operations.add_comment(issue_id, comment)
                logger.info("Posted validation comment to %s", issue_id)
                return True
            except Exception:
                logger.exception("Failed to add validation comment to %s", issue_id)
        return False

    def _generate_test_cases(self, issue_id: str, question: str, **kwargs: Any) -> str:
        """Return test cases string generated from Jira ``issue_id``.

        The ``TestAgent`` will reply with ``HAS_TESTS`` if the description
        already contains test cases. In that case an empty string is returned.
        """
        try:
            issue_json = get_issue_by_id_tool.run(issue_id)
            issue = json.loads(issue_json)
            fields = issue.get("fields", {})
            summary = fields.get("summary", "") or ""
            description = fields.get("description", "") or ""
            text = f"{summary}\n{description}\n{question}"
            tests = self.tester.create_test_cases(text, None, **kwargs)
            return tests or ""
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

    def _validate_and_generate_tests(self, issue_id: str, question: str, **kwargs: Any) -> str:
        """Run validation and return generated test cases if possible."""
        self._classify_and_validate(issue_id, **kwargs)
        tests = self._generate_test_cases(issue_id, question, **kwargs)
        cleaned = normalize_newlines(tests)
        if cleaned and not cleaned.lower().startswith("not enough"):
            if self._add_tests_to_description(issue_id, cleaned):
                cleaned += "\n\nDescription updated with generated tests."
        return cleaned

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ask(self, question: str, **kwargs: Any) -> str:
        """Route ``question`` to the appropriate workflow."""
        logger.info("Router received question: %s", question)

        if self.use_memory and self.memory is not None:
            user_count = sum(
                1 for m in self.memory.chat_memory.messages if m.type == "human"
            )
            if user_count >= self.max_history:
                ans = (
                    input("max_context_window reached - start new conversation - Y/N: ")
                    .strip()
                    .lower()
                )
                if ans.startswith("y"):
                    self.memory.clear()
            self.memory.chat_memory.add_user_message(question)

        issue_id = self._extract_issue_id(question)
        if issue_id:
            self.session_memory.current_issue = issue_id
        else:
            issue_id = self.session_memory.current_issue

        try:
            intent = self._classify_intent(question, **kwargs)
            if intent.startswith("OPERATE"):
                logger.info("Routing to operations workflow")
                answer = self.operations.operate(question, issue_id=issue_id, **kwargs)
            elif intent.startswith("CREATE"):
                logger.info("Routing to issue creation workflow")
                project_key = self._extract_project_key(question)
                if not project_key and self.config.projects:
                    project_key = self.config.projects[0]
                if not project_key:
                    answer = "Please specify a Jira project key"
                else:
                    answer = self.creator.create_issue(question, project_key, **kwargs)
            else:
                if not issue_id:
                    answer = "No Jira ticket found in question"
                    if self.use_memory and self.memory is not None:
                        self.memory.chat_memory.add_ai_message(answer)
                    self.session_memory.save_context(
                        {"input": question}, {"output": answer}
                    )
                    return answer
                if intent.startswith("VALIDATE"):
                    logger.info("Routing to validation workflow")
                    answer = self._classify_and_validate(issue_id, **kwargs)
                    comment_posted = self._handle_validation_result(issue_id, answer)
                    if comment_posted:
                        answer += "\n\nValidation summary posted as a Jira comment."
                    tests = self._generate_test_cases(issue_id, question, **kwargs)
                    if tests:
                        cleaned = normalize_newlines(tests)
                        if not cleaned.lower().startswith("not enough") and self._add_tests_to_description(issue_id, cleaned):
                            answer += "\n\nDescription updated with generated tests."
                        answer += "\n\n" + cleaned
                elif intent.startswith("TEST"):
                    logger.info("Routing to test generation workflow")
                    answer = self._validate_and_generate_tests(issue_id, question, **kwargs)
                else:
                    logger.info("Routing to general insights workflow")
                    include_history = self._needs_history(question, **kwargs)
                    history_msgs = None
                    if self.use_memory:
                        role_map = {"human": "user", "ai": "assistant"}
                        history_msgs = [
                            {"role": role_map.get(m.type, m.type), "content": m.content}
                            for m in self.memory.chat_memory.messages
                        ]
                    answer = self.insights.ask(
                        issue_id,
                        question,
                        include_history=include_history,
                        history=history_msgs,
                        **kwargs,
                    )
        except JIRAError:
            logger.exception("Jira error while fetching issue %s", issue_id)
            answer = f"Sorry, I couldn't find the Jira issue {issue_id}. Please check the key and try again."
        except OpenAIError:
            logger.exception("OpenAI API error")
            answer = (
                "I'm having trouble communicating with the language model right now. "
                "Please try again later."
            )
        except Exception:
            logger.exception("Unexpected error while processing question")
            answer = "Sorry, something went wrong while handling your request."
        if self.use_memory and self.memory is not None:
            self.memory.chat_memory.add_ai_message(answer)
        self.session_memory.save_context({"input": question}, {"output": answer})
        return answer


__all__ = ["RouterAgent"]
