# Jira AI Assistant

This project provides a command line assistant that interacts with Jira.

Phase 1 implemented a basic project scaffold with a minimal REST adapter and a Typer CLI.

Phase 2 introduces a simple LangChain agent. The agent wraps the LLM through ``src.llm.llm_wrapper`` and exposes tools for describing, searching and transitioning Jira issues.

Phase 3 adds a YAML based configuration under ``src/config/criteria.yaml`` and a ``HotReloader`` utility to automatically reload files when they change.

Phase 4 introduces ``OpenAIService`` which loads your API key and model from ``config.py`` and provides a light wrapper around the OpenAI response API used by the agent.

```
python -m src.cli.main hello
```
