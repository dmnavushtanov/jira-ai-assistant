"""Router agent that directs questions to the appropriate workflow."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

try:
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools import Tool, BaseTool  # type: ignore
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.prompts import PromptTemplate
except Exception:  # pragma: no cover - optional dependency
    AgentExecutor = None
    BaseTool = None
    Tool = None
    ConversationBufferWindowMemory = None
    PromptTemplate = None


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
        self.max_history = self.config.max_questions_to_remember

        if self.use_memory and ConversationBufferWindowMemory is not None:
            self.memory = ConversationBufferWindowMemory(
                k=self.max_history,
                return_messages=True,
                memory_key="chat_history",
            )
        else:
            if self.use_memory:
                logger.warning("LangChain not installed; conversation memory disabled")
            self.use_memory = False
            self.memory = None
        self.session_memory = JiraContextMemory()
        if self.config.projects:
            pattern = "|".join(re.escape(p) for p in self.config.projects)
        else:
            pattern = r"[A-Za-z][A-Za-z0-9]+"
        self.issue_re = re.compile(rf"(?:{pattern})-\d+", re.IGNORECASE)
        self.history_prompt = load_prompt("needs_history.txt")
        self.intent_prompt = load_prompt("intent_classifier.txt")
        self.llm = None
        self.tools: List[BaseTool] = []
        self.agent_executor: AgentExecutor | None = None

        if AgentExecutor is not None and Tool is not None and PromptTemplate is not None:
            self.llm = create_langchain_llm(config_path)
            self.tools = self._create_tools()
            if self.use_memory:
                self.agent_executor = self._create_agent_executor()
        else:
            logger.warning("LangChain not installed; advanced routing disabled")

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

    # ------------------------------------------------------------------
    # Tooling helpers
    # ------------------------------------------------------------------
    def _parse_tool_input(self, input_str: str) -> Dict[str, str]:
        """Parse tool input in format 'key:value|key:value'."""
        parts: Dict[str, str] = {}
        for part in input_str.split("|"):
            if ":" in part:
                key, value = part.split(":", 1)
                parts[key.strip()] = value.strip()
        return parts

    def _create_tools(self) -> List[BaseTool]:
        if Tool is None:
            return []
        tools: List[BaseTool] = []
        tools.append(
            Tool(
                name="get_issue_insights",
                func=self._handle_insights,
                description=(
                    "Get details and analysis about a Jira issue when the user asks for information. "
                    "Use only when explicitly asked for issue details, status, or insights. "
                    "Input: 'issue_id:PROJ-123|question:What is the status?'"
                ),
            )
        )

        tools.append(
            Tool(
                name="validate_api_issue",
                func=self._handle_validation,
                description=(
                    "Validate API-related Jira issues ONLY when explicitly requested by the user. "
                    "Do not use this tool unless the user specifically asks for validation. "
                    "Input: 'issue_id:PROJ-123'"
                ),
            )
        )

        tools.append(
            Tool(
                name="perform_jira_operation",
                func=self._handle_operations,
                description=(
                    "Perform operations on Jira issues such as adding comments, updating fields, or transitioning status. "
                    "Use this tool for action requests like 'add comment', 'update field', 'transition to status'. "
                    "Input: 'issue_id:PROJ-123|question:add comment \"nice jira\"'"
                ),
            )
        )

        tools.append(
            Tool(
                name="generate_test_cases",
                func=self._handle_test_generation,
                description=(
                    "Generate test cases for a Jira issue ONLY when explicitly requested. "
                    "Use only when the user asks for test cases or test generation. "
                    "Input: 'issue_id:PROJ-123|question:create tests'"
                ),
            )
        )

        tools.append(
            Tool(
                name="create_jira_issue",
                func=self._handle_issue_creation,
                description=(
                    "Create a new Jira issue when the user requests issue creation. "
                    "Input: 'description:Fix bug|project:PROJ'"
                ),
            )
        )

        tools.append(
            Tool(
                name="get_current_context",
                func=lambda _: self._get_current_context(),
                description="Return the current conversation context when asked.",
            )
        )

 #       tools.append(
#           Tool(
 #               name="classify_intent",
 #               func=self._classify_intent,
 #               description="Classify user intent as VALIDATE, OPERATE, INSIGHT, TEST, CREATE, or UNKNOWN",
 #           )
 #       )

        return tools

    def _create_agent_executor(self) -> AgentExecutor: # type: ignore
        # Try to load the multi-step reasoning template first
        try:
            multi_step_template = load_prompt("multi_step_reasoning.txt")
            if multi_step_template:
                template = multi_step_template
                logger.debug("Using multi-step reasoning template")
            else:
                raise FileNotFoundError("Multi-step template not found")
        except Exception:
            # Fallback to the original template
            logger.debug("Using fallback template")
            template = (
                "Answer the following questions as best you can. You have access to the following tools:\n\n"
                "{tools}\n\n"
                "Use the following format:\n\n"
                "Question: the input question you must answer\n"
                "Thought: you should always think about what to do\n"
                "Action: the action to take, should be one of [{tool_names}]\n"
                "Action Input: the input to the action\n"
                "Observation: the result of the action\n"
                "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
                "Thought: I now know the final answer\n"
                "Final Answer: the final answer to the original input question\n\n"
                "Begin!\n\n"
                "Question: {input}\n"
                "Thought: {agent_scratchpad}"
            )
        
        prompt = PromptTemplate.from_template(template)
        agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
        )

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------
    def _handle_insights(self, input_str: str) -> str:
        try:
            parts = self._parse_tool_input(input_str)
            issue_id = parts.get("issue_id", "")
            question = parts.get("question", "Tell me about this issue")
            if not issue_id:
                issue_id = self.session_memory.current_issue
                if not issue_id:
                    return "No issue ID provided and no current issue in context"
            self.session_memory.current_issue = issue_id
            return self.insights.ask(issue_id, question)
        except Exception as exc:
            logger.exception("Error in insights handler")
            return f"Error getting insights: {exc}"

    def _handle_validation(self, input_str: str) -> str:
        try:
            parts = self._parse_tool_input(input_str)
            issue_id = parts.get("issue_id", "")
            if not issue_id:
                issue_id = self.session_memory.current_issue
                if not issue_id:
                    return "No issue ID provided and no current issue in context"
            issue_json = get_issue_by_id_tool.run(issue_id)
            issue = json.loads(issue_json)
            result = self.validator.validate(issue)
            
            # When called as a tool from LangChain agent, don't auto-post validation comments
            # This prevents validation results from interfering with successful operations
            # The validation result is returned to the agent for decision-making only
            logger.debug("Validation tool called - returning result without auto-posting comment")
            return result
        except Exception as exc:
            logger.exception("Error in validation handler")
            return f"Error validating issue: {exc}"

    def _handle_operations(self, input_str: str) -> str:
        try:
            parts = self._parse_tool_input(input_str)
            issue_id = parts.get("issue_id", "")
            question = parts.get("question", "")
            if not issue_id:
                issue_id = self.session_memory.current_issue
                if not issue_id:
                    return "No issue ID provided and no current issue in context"
            self.session_memory.current_issue = issue_id
            return self.operations.operate(question, issue_id=issue_id)
        except Exception as exc:
            logger.exception("Error in operations handler")
            return f"Error performing operation: {exc}"

    def _handle_test_generation(self, input_str: str) -> str:
        try:
            parts = self._parse_tool_input(input_str)
            issue_id = parts.get("issue_id", "")
            question = parts.get("question", "Generate test cases")
            if not issue_id:
                issue_id = self.session_memory.current_issue
                if not issue_id:
                    return "No issue ID provided and no current issue in context"
            return self._generate_tests(issue_id, question)
        except Exception as exc:
            logger.exception("Error in test generation handler")
            return f"Error generating tests: {exc}"

    def _handle_issue_creation(self, input_str: str) -> str:
        try:
            parts = self._parse_tool_input(input_str)
            description = parts.get("description", "")
            project = parts.get("project", "")
            if not project and self.config.projects:
                project = self.config.projects[0]
            if not project:
                return "No project specified and no default project configured"
            return self.creator.create_issue(description, project)
        except Exception as exc:
            logger.exception("Error in issue creation handler")
            return f"Error creating issue: {exc}"

    def _get_current_context(self) -> str:
        context = {
            "current_issue": self.session_memory.current_issue,
            "recent_questions": len(self.session_memory.chat_history),
            "configured_projects": self.config.projects or [],
        }
        return json.dumps(context, indent=2)

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
        if comment and self.config.write_comments_to_jira:
            comment = normalize_newlines(comment)
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
        self, issue_id: str, question: str, **kwargs: Any
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
            tests = self.tester.create_test_cases(text, None, **kwargs)
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

    def _generate_tests(self, issue_id: str, question: str, **kwargs: Any) -> str:
        """Return generated test cases and update Jira when possible."""
        tests = self._generate_test_cases(issue_id, question, **kwargs)

        if tests is None or tests == EXISTING_TESTS_MSG:
            return EXISTING_TESTS_MSG
          
        cleaned = normalize_newlines(tests)
        if cleaned and not cleaned.lower().startswith("not enough"):
            if self._add_tests_to_description(issue_id, cleaned):
                cleaned += "\n\nDescription updated with generated tests."
        return cleaned

    def _execute_operations_plan(self, plan: dict[str, Any], **kwargs: Any) -> str:
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
        
        logger.info("Executing %d-step plan for issue %s", len(steps), issue_key)
        results = []
        for i, step in enumerate(steps, 1):
            agent = step.get("agent")
            action = step.get("action")
            params = step.get("parameters") or {}
            logger.info("Step %d/%d: %s.%s with params %s", i, len(steps), agent, action, params)
            
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
                    logger.warning("Step %d failed: %s", i, result)
                else:
                    results.append(f"Executed {action} successfully")
                    logger.info("Step %d completed successfully", i)
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
        """Answer ``question`` using LangChain when available."""
        logger.info("Router received question: %s", question)

        notice = self._check_history_limit()

        issue_id = self._extract_issue_id(question)
        if issue_id:
            self.session_memory.current_issue = issue_id

        if self.agent_executor is None:
            answer = self._basic_routing(question, **kwargs)
        else:
            try:
                context = self._get_current_context()
                enhanced_question = (
                    f"Current context: {context}\n\nQuestion: {question}"
                )
                result = self.agent_executor.invoke({"input": enhanced_question})
                answer = result.get("output", "I couldn't process your request.")
            except Exception as exc:
                logger.exception("Error in agent executor")
                answer = f"I encountered an error processing your request: {exc}"

        if notice:
            answer = f"{notice}\n\n{answer}"
        self.session_memory.save_context({"input": question}, {"output": answer})
        return answer

    # ------------------------------------------------------------------
    # Fallback routing
    # ------------------------------------------------------------------
    def _basic_routing(self, question: str, **kwargs: Any) -> str:
        issue_id = self._extract_issue_id(question) or self.session_memory.current_issue
        try:
            intent = self._classify_intent(question, **kwargs)
            if intent.startswith("OPERATE"):
                plan = self.planner.generate_plan(question, {"issue_key": issue_id or ""}, **kwargs)
                if plan.get("plan"):
                    return self._execute_operations_plan(plan, **kwargs)
                return self.operations.operate(question, issue_id=issue_id, **kwargs)
            if intent.startswith("CREATE"):
                project_key = self._extract_project_key(question) or (self.config.projects[0] if self.config.projects else None)
                if not project_key:
                    return "Please specify a Jira project key"
                return self.creator.create_issue(question, project_key, **kwargs)
            if not issue_id:
                return (
                    "I'm sorry, I didn't catch an issue key in your question. "
                    "Could you specify which Jira issue you mean?"
                )
            if intent.startswith("VALIDATE"):
                result = self._classify_and_validate(issue_id, **kwargs)
                if self._handle_validation_result(issue_id, result):
                    result += "\n\nValidation summary posted as a Jira comment."
                return result
            if intent.startswith("TEST"):
                return self._generate_tests(issue_id, question, **kwargs)
            return self.insights.ask(issue_id, question, **kwargs)
        except JIRAError:
            logger.exception("Jira error while fetching issue %s", issue_id)
            return f"Sorry, I couldn't find the Jira issue {issue_id}. Please check the key and try again."
        except OpenAIError:
            logger.exception("OpenAI API error")
            return (
                "I'm having trouble communicating with the language model right now. "
                "Please try again later."
            )
        except Exception:
            logger.exception("Unexpected error while processing question")
            return "Sorry, something went wrong while handling your request."


__all__ = ["RouterAgent"]