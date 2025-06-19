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
    from langchain.agents import (
        initialize_agent,
        AgentType,
        create_react_agent,
        AgentExecutor,
    )  # type: ignore
    from langchain.prompts import PromptTemplate
except Exception:  # pragma: no cover - optional dependency
    ConversationBufferWindowMemory = None
    ConversationSummaryMemory = None
    CombinedMemory = None
    initialize_agent = None  # type: ignore
    AgentType = None  # type: ignore
    create_react_agent = None  # type: ignore
    AgentExecutor = None  # type: ignore
    PromptTemplate = None  # type: ignore

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

        # create LangChain tools and optional agent executor
        self.langchain_tools = self._create_tools()
        self.agent_executor = self._create_agent_executor()

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

    def _parse_tool_input(self, input_str: str) -> dict[str, str]:
        """Return dict parsed from ``input_str`` formatted as 'key:value|key:value'."""
        result: dict[str, str] = {}
        for part in str(input_str).split("|"):
            if ":" in part:
                key, value = part.split(":", 1)
                result[key.strip()] = value.strip()
        return result

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

    def _classify_intent_with_score(self, question: str, **kwargs: Any) -> tuple[str, float]:
        """Return intent and a simple confidence score.

        The score is derived heuristically by comparing the plain intent
        classification with the classification that includes conversation
        history for additional context. If both classifications agree and are
        not ``UNKNOWN`` the result is considered high confidence.
        """

        basic = self._classify_intent(question, **kwargs)
        contextual = self._classify_intent_with_context(question, **kwargs)

        if basic == contextual and contextual != "UNKNOWN":
            score = 0.9
        elif contextual != "UNKNOWN" and basic != "UNKNOWN":
            score = 0.7
        elif contextual != "UNKNOWN" or basic != "UNKNOWN":
            score = 0.6
        else:
            score = 0.5

        return contextual, score

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
                self._request_confirmation(issue_id, comment)
                return False
            try:
                self.operations.add_comment(issue_id, comment)
                logger.info("Posted validation comment to %s", issue_id)
                return True
            except Exception:
                logger.exception("Failed to add validation comment to %s", issue_id)
        return False

    def _request_confirmation(self, issue_id: str, comment: str) -> None:
        """Store state for posting confirmation comment."""
        self._confirm_issue = issue_id
        self._confirm_comment = comment
        self._pending_confirmation = f"Post validation comment to {issue_id}? (yes/no)"
        logger.info("Awaiting user confirmation to comment on %s", issue_id)

    def _reset_confirmation(self) -> None:
        self._pending_confirmation = None
        self._confirm_issue = None
        self._confirm_comment = None

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
    # Tooling
    # ------------------------------------------------------------------
    def _create_tools(self) -> list[Any]:
        """Create LangChain tools for the agent."""
        if Tool is None:
            return []
        tools = [
            Tool(
                name="GetContext",
                func=lambda _: self.prepare_conversation_history(),
                description="Get recent conversation context for use in other tools.",
            ),
            Tool(
                name="AskInsight",
                func=self._tool_insight,
                description="Answer questions about an issue. Input 'issue:KEY|question:TEXT'",
            ),
            Tool(
                name="ValidateIssue",
                func=self._tool_validate,
                description="Validate an API on an issue. Input 'issue:KEY|question:TEXT'",
            ),
            Tool(
                name="JiraOperate",
                func=self._tool_operate,
                description="Perform operations on Jira issues. Input 'issue:KEY|question:TEXT'",
            ),
            Tool(
                name="GenerateTests",
                func=self._tool_generate_tests,
                description="Generate test cases. Input 'issue:KEY|question:TEXT'",
            ),
            Tool(
                name="CreateIssue",
                func=self._tool_create_issue,
                description="Create a new issue. Input 'project:KEY|request:TEXT'",
            ),
            Tool(
                name="ClassifyIntent",
                func=self._tool_classify_intent,
                description="Classify user intent. Input 'question:TEXT'",
            ),
        ]
        return tools

    def _create_agent_executor(self) -> Optional[Any]:
        """Create modern LangChain agent executor if possible."""
        if None in (create_react_agent, AgentExecutor, PromptTemplate):
            return None
        llm = create_langchain_llm(None)
        if llm is None:
            return None
        prompt = PromptTemplate.from_template(
            """
You are a helpful Jira assistant. Use the provided tools to answer the user.

Available tools: {tools}
Tool names: {tool_names}

Question: {input}
{agent_scratchpad}
"""
        )
        agent = create_react_agent(llm=llm, tools=self.langchain_tools, prompt=prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.langchain_tools,
            memory=self.memory,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
        )

    # Tool handlers ------------------------------------------------------
    def _tool_insight(self, input_str: str) -> str:
        params = self._parse_tool_input(input_str)
        issue = params.get("issue") or params.get("issue_id")
        question = params.get("question", "")
        if not issue:
            return "Error: missing issue"
        try:
            history = self.prepare_conversation_history()
            return self.insights.ask(issue, question, history=history)
        except Exception as exc:
            logger.exception("Insight tool failed")
            return f"Error: {exc}"

    def _tool_validate(self, input_str: str) -> str:
        params = self._parse_tool_input(input_str)
        issue = params.get("issue")
        question = params.get("question", "")
        if not issue:
            return "Error: missing issue"
        try:
            history = self.prepare_conversation_history()
            result = self._classify_and_validate(issue, history=history)
            self._handle_validation_result(issue, result)
            return result
        except Exception as exc:
            logger.exception("Validation tool failed")
            return f"Error: {exc}"

    def _tool_operate(self, input_str: str) -> str:
        params = self._parse_tool_input(input_str)
        issue = params.get("issue")
        question = params.get("question", "")
        try:
            history = self.prepare_conversation_history()
            return self.operations.operate(question, issue_id=issue, history=history)
        except Exception as exc:
            logger.exception("Operations tool failed")
            return f"Error: {exc}"

    def _tool_generate_tests(self, input_str: str) -> str:
        params = self._parse_tool_input(input_str)
        issue = params.get("issue")
        question = params.get("question", "")
        if not issue:
            return "Error: missing issue"
        try:
            history = self.prepare_conversation_history()
            return self._generate_tests(issue, question, history=history)
        except Exception as exc:
            logger.exception("Generate tests tool failed")
            return f"Error: {exc}"

    def _tool_create_issue(self, input_str: str) -> str:
        params = self._parse_tool_input(input_str)
        project = params.get("project")
        request = params.get("request", "")
        if not project:
            return "Error: missing project"
        try:
            history = self.prepare_conversation_history()
            return self.creator.create_issue(request, project, history=history)
        except Exception as exc:
            logger.exception("Create issue tool failed")
            return f"Error: {exc}"

    def _tool_classify_intent(self, input_str: str) -> str:
        params = self._parse_tool_input(input_str)
        question = params.get("question", "")
        intent, score = self._classify_intent_with_score(question)
        return json.dumps({"intent": intent, "confidence": score})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ask(self, question: str, **kwargs: Any) -> str:
        """Return an answer for ``question`` routed to the best workflow."""
        logger.info("Router received question: %s", question)
        used_executor = False

        notice = self._check_history_limit()

        if self._pending_confirmation:
            user_reply = question.strip().lower()
            issue = self._confirm_issue
            comment = self._confirm_comment
            self._reset_confirmation()
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
            intent, score = self._classify_intent_with_score(question, **kwargs)
            if score < 0.6:
                intent = "INSIGHT"
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
                    if self.agent_executor is not None:
                        answer = self.agent_executor.invoke({"input": question})
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
