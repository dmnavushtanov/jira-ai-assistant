"""Agent for generating test cases and exercising APIs.

This agent wraps its ``create_test_cases`` helper in a LangChain ``Tool`` so it
can be composed with other agents.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import json
import re

from src.configs.config import load_config
from src.llm_clients import create_llm_client, create_langchain_llm
from src.prompts import load_prompt
from src.utils import safe_format
from src.agents.planning import create_planning_pipeline

try:
    from langchain.chains import LLMChain, SequentialChain  # type: ignore
    from langchain.prompts import PromptTemplate  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LLMChain = None  # type: ignore
    SequentialChain = None  # type: ignore
    PromptTemplate = None  # type: ignore

try:
    from langchain.tools import Tool  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Tool = None

logger = logging.getLogger(__name__)
logger.debug("test_agent module loaded")


class TestAgent:
    """Agent that generates test cases from freeform text.

    The text can be a validation summary, the Jira issue description or any
    other contextual information combined with the user's question.
    """

    def __init__(self, config_path: str | None = None) -> None:
        logger.debug("Initializing TestAgent with config_path=%s", config_path)
        self.config = load_config(config_path)
        self.client = create_llm_client(config_path)
        self.prompts = {
            "GET": load_prompt("tests/get_test_cases.txt"),
            "POST": load_prompt("tests/post_test_cases.txt"),
            "PUT": load_prompt("tests/put_test_cases.txt"),
            "DELETE": load_prompt("tests/delete_test_cases.txt"),
        }
        self.default_prompt = load_prompt("tests/testCasesGeneration.txt")

        # setup optional planning components using LangChain if available
        self.llm = None
        self.method_prompt = None
        self.context_prompt = None
        self.test_prompt = None
        if PromptTemplate is not None:
            # use a LangChain LLM that matches the configured provider
            self.llm = create_langchain_llm(config_path)
            if self.llm is not None:
                method_instruction = load_prompt("tests/method_detection.txt")
                context_instruction = load_prompt("tests/context_analysis.txt")

                base_method = PromptTemplate(
                    input_variables=["question", "instruction"],
                    template="{instruction}\nQuestion: {question}\nMethod:",
                )
                self.method_prompt = base_method.partial(
                    instruction=method_instruction
                )

                base_context = PromptTemplate(
                    input_variables=["jira_content", "method", "instruction"],
                    template="{instruction}\nMethod: {method}\n{jira_content}\nPlan:",
                )
                self.context_prompt = base_context.partial(
                    instruction=context_instruction
                )

                base_test = PromptTemplate(
                    input_variables=["summary", "method", "instruction"],
                    template="{instruction}\nMethod: {method}\nSummary: {summary}",
                )
                self.test_prompt = base_test.partial(
                    instruction=self.default_prompt
                )

        # Tools exposed by this agent
        self.tools = []
        self.generate_tests_tool = None
        if Tool is not None:
            self.generate_tests_tool = Tool(
                name="GenerateTests",
                func=self.create_test_cases,
                description="Generate test cases based on validation results.",
            )
            self.tools.append(self.generate_tests_tool)

    def _extract_method(self, text: str) -> Optional[str]:
        """Return HTTP method found in ``text`` if any."""
        try:
            data: Dict[str, Any] = json.loads(text)
        except Exception:
            from src.utils import parse_json_block

            data = parse_json_block(text) or {}
        if isinstance(data, dict):
            parsed = data.get("parsed") or {}
            method = parsed.get("method")
            if method:
                return str(method).upper()
        match = re.search(r"\b(GET|POST|PUT|DELETE)\b", text, re.I)
        return match.group(1).upper() if match else None

    def create_test_cases(
        self, text: str, method: Optional[str] = None, **kwargs: Any
    ) -> Optional[str]:
        """Return generated test cases or ``None`` if tests already exist.

        The LLM checks the provided ``text`` for existing test cases. If they
        are present it responds with ``HAS_TESTS`` and this method returns
        ``None``. Otherwise the response contains the new tests which are
        returned as-is.
        """
        method = (method or self._extract_method(text) or "GET").upper()
        template = self.prompts.get(method) or self.default_prompt
        if not template:
            raise RuntimeError("Test case generation prompt not found")
        prompt = safe_format(template, {"summary": text})
        logger.info("Generating test cases from provided text")
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat_completion(messages, **kwargs)
        try:
            result = response.choices[0].message.content.strip()
        except Exception:
            try:
                result = response["choices"][0]["message"]["content"].strip()
            except Exception:  # pragma: no cover - handle unexpected structure
                logger.exception("Failed to parse response")
                result = str(response)
        logger.debug("Generated test cases: %s", result)
        if result.upper().startswith("HAS_TESTS"):
            return None
        return result

    # ------------------------------------------------------------------
    # Planning Pipeline
    # ------------------------------------------------------------------
    def create_planning_pipeline(self) -> Any:
        """Create a multi-step planning pipeline."""
        if None in (
            self.llm,
            LLMChain,
            SequentialChain,
            self.method_prompt,
            self.context_prompt,
            self.test_prompt,
        ):
            logger.warning("Planning pipeline not available")
            return None
        return create_planning_pipeline(
            self.llm,
            self.method_prompt,
            self.context_prompt,
            self.test_prompt,
        )

    def plan_and_generate(self, question: str, jira_content: str, **kwargs: Any) -> str:
        """Plan test cases then generate them in a single pipeline."""
        pipeline = self.create_planning_pipeline()
        if pipeline is None:
            return self.create_test_cases(
                jira_content, method=self._extract_method(question)
            )
        result = pipeline({"question": question, "jira_content": jira_content})
        return str(result.get("test_cases", ""))


__all__ = ["TestAgent"]
