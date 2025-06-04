# Jira AI Assistant

This repository demonstrates a very small assistant that communicates with the OpenAI API and exposes a number of Jira related tools.  A simple wrapper around the OpenAI SDK is provided together with a service layer and a minimal agent.

## Configuration

Create a `.env` file and provide your OpenAI API key under `OPENAI_API_KEY`.  The default model can be changed in `config.yaml` or by setting `OPENAI_MODEL` in the environment.  Jira credentials should be supplied via `JIRA_BASE_URL`, `JIRA_EMAIL` and `JIRA_API_TOKEN` when using the Jira tools.

## Usage

The new `OpenAIService` class can be used to ask arbitrary questions via the OpenAI chat API:

```bash
python openai_service.py "What is 2 + 2?"
```

`JiraAIAgent` collects all tools defined in `jira_service.py` and exposes an `ask` method that uses the OpenAI service.  The agent can be invoked from the command line as well:

```bash
python agent.py "List the issue details for PROJ-1"
```

These examples require valid API keys for both OpenAI and Jira if the Jira tools are exercised.
