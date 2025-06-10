from __future__ import annotations

from typing import Any, Optional

try:
    from langchain.callbacks.base import BaseCallbackHandler
except Exception:  # pragma: no cover - langchain optional
    BaseCallbackHandler = object  # type: ignore[misc,assignment]

try:
    from rich.console import Console
except Exception:  # pragma: no cover - rich not installed
    Console = None  # type: ignore


class RichLogger(BaseCallbackHandler):
    """Simple LangChain callback for colorful logs."""

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or (Console() if Console else None)

    def on_tool_start(self, serialized: Any, input_str: str | dict, **kwargs: Any) -> None:  # type: ignore[override]
        if not self.console:
            return
        self.console.rule(f"[bold blue]Tool: {serialized}")
        self.console.print(f"[cyan]Input:[/] {input_str}")

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:  # type: ignore[override]
        if not self.console:
            return
        self.console.print(f"[green]Output:[/] {output}")

    def on_llm_start(self, serialized: Any, prompts: list[str], **kwargs: Any) -> None:  # type: ignore[override]
        if not self.console:
            return
        for prompt in prompts:
            self.console.print(f"[blue][LLM Prompt][/]: {prompt}")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # type: ignore[override]
        if not self.console:
            return
        self.console.print(f"[magenta][LLM Response][/]: {response}")
