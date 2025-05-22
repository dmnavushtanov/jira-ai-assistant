# Jira AI Assistant

This project provides a command line assistant that interacts with Jira.

Phase 1 implemented a basic project scaffold with a minimal REST adapter and a Typer CLI.

Phase 2 introduces a simple LangChain agent. The agent wraps the LLM through ``src.llm.llm_wrapper`` and exposes tools for describing, searching and transitioning Jira issues.

```
python -m src.cli.main hello
```
