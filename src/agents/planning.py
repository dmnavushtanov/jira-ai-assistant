"""Utilities for creating multi-step planning chains."""

from __future__ import annotations

import logging
from typing import Any


try:
    from langchain.chains import LLMChain, SequentialChain  # type: ignore
    from langchain.prompts import PromptTemplate  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    LLMChain = None  # type: ignore
    SequentialChain = None  # type: ignore
    PromptTemplate = object  # type: ignore

logger = logging.getLogger(__name__)
logger.debug("planning module loaded")


def create_planning_pipeline(
    llm: Any,
    method_prompt: PromptTemplate, # type: ignore
    context_prompt: PromptTemplate, # type: ignore
    test_prompt: PromptTemplate, # type: ignore
) -> SequentialChain | None: # type: ignore
    """Return a SequentialChain for method detection, context analysis and test generation."""
    if None in (LLMChain, SequentialChain, method_prompt, context_prompt, test_prompt, llm):
        logger.warning("Planning pipeline not available")
        return None

    # Step 1: detect HTTP method
    method_chain = LLMChain(
        llm=llm,
        prompt=method_prompt,
        output_key="method",
    )

    # Step 2: analyze context
    context_chain = LLMChain(
        llm=llm,
        prompt=context_prompt,
        output_key="summary",
    )

    # Step 3: generate test cases
    test_chain = LLMChain(
        llm=llm,
        prompt=test_prompt,
        output_key="test_cases",
    )

    return SequentialChain(
        chains=[method_chain, context_chain, test_chain],
        input_variables=["question", "jira_content"],
        output_variables=["test_cases"],
    )


__all__ = ["create_planning_pipeline"]
