from __future__ import annotations

from typing import Any

try:
    from langchain.callbacks.base import BaseCallbackHandler
except Exception:  # pragma: no cover - langchain optional
    BaseCallbackHandler = object  # type: ignore[misc,assignment]

try:
    from rich.console import Console
except Exception:  # pragma: no cover - rich not installed
    Console = None  # type: ignore

console = Console() if Console else None


class RichLogger(BaseCallbackHandler):
    """Simple LangChain callback for colorful logs."""

    def on_tool_start(self, serialized: Any, input_str: str | dict, **kwargs: Any) -> None:  # type: ignore[override]
        if not console:
            return
        console.rule(f"[bold blue]Tool: {serialized}")
        console.print(f"[cyan]Input:[/] {input_str}")

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:  # type: ignore[override]
        if not console:
            return
        console.print(f"[green]Output:[/] {output}")

    def on_llm_start(self, serialized: Any, prompts: list[str], **kwargs: Any) -> None:  # type: ignore[override]
        if not console:
            return
        for prompt in prompts:
            console.print(f"[blue][LLM Prompt][/]: {prompt}")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # type: ignore[override]
        if not console:
            return
        console.print(f"[magenta][LLM Response][/]: {response}")
