"""Router agent that directs questions to the appropriate workflow."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

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
from src.prompts import load_prompt
from src.services.jira_service import get_issue_by_id_tool
from src.utils import safe_format

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
        if self.config.projects:
            pattern = "|".join(re.escape(p) for p in self.config.projects)
        else:
            pattern = r"[A-Za-z][A-Za-z0-9]+"
        self.issue_re = re.compile(rf"(?:{pattern})-\d+", re.IGNORECASE)
        self.router_prompt = load_prompt("router.txt")
        self.history_prompt = load_prompt("needs_history.txt")

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

    def _should_validate(self, question: str, **kwargs: Any) -> bool:
        """Return ``True`` if ``question`` explicitly requests validation."""
        lowered = question.lower()
        explicit_phrases = [
            r"\bvalidate\s+this\s+jira\b",
            r"\btest\s+this\s+jira\b",
        ]
        for pattern in explicit_phrases:
            if re.search(pattern, lowered):
                logger.debug("Explicit validation phrase detected: %s", pattern)
                return True
        return False

    def _needs_history(self, question: str, **kwargs: Any) -> bool:
        """Return True if the LLM determines the changelog is required."""
        if self.history_prompt:
            prompt = safe_format(self.history_prompt, {"question": question})
        else:
            prompt = (
                "Do we need the change history to answer this question? "
                "Respond with HISTORY or NO_HISTORY.\nQuestion: " + question
            )
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
        if not issue_id:
            answer = "No Jira ticket found in question"
            if self.use_memory and self.memory is not None:
                self.memory.chat_memory.add_ai_message(answer)
            return answer

        if self._should_validate(question, **kwargs):
            logger.info("Routing to validation workflow")
            answer = self._classify_and_validate(issue_id, **kwargs)
            if self.use_memory and self.memory is not None:
                self.memory.chat_memory.add_ai_message(answer)
            return answer

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
        if self.use_memory and self.memory is not None:
            self.memory.chat_memory.add_ai_message(answer)
        return answer


__all__ = ["RouterAgent"]
