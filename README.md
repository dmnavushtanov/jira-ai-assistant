# Jira AI Assistant

This repository contains a Jira AI assistant that communicates with the OpenAI API and provides utilities for interacting with Jira issues. It includes:

- Simple utilities for retrieving Jira issue details
- OpenAI service integration for AI-powered assistance
- A minimal agent that combines both capabilities
- Optional Langfuse integration for monitoring LLM calls

## Configuration

Create a `.env` file and provide your credentials:

```
JIRA_BASE_URL=https://your-domain.atlassian.net
JIRA_EMAIL=your-email@example.com
JIRA_API_TOKEN=your-api-token
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o-mini
BASE_LLM=openai  # or 'anthropic'
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
# Optional custom host, defaults to Langfuse cloud
LANGFUSE_HOST=https://app.langfuse.com
```

The default model and provider can be changed in `src/configs/config.yml` or by setting `OPENAI_MODEL` and `BASE_LLM` in the environment.

If `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are provided, all LLM calls
will be reported to Langfuse. Set `LANGFUSE_HOST` when using a self-hosted
instance.

## Usage

### Router Agent CLI

`main.py` now uses the `RouterAgent` to decide whether your question requires
API validation or general issue insights. Run the script and type any question
containing a Jira key:

```bash
python main.py
```

Example questions:

```text
Validate RB-1234
What is the status of SD-99?
```

### OpenAI Service

The `OpenAIService` class can be used to ask arbitrary questions via the OpenAI chat API:

```bash
python openai_service.py "What is 2 + 2?"
```

### AI Agent with Jira Tools

`JiraAIAgent` collects all tools defined in `jira_service.py` and exposes an `ask` method that uses the OpenAI service. The agent can be invoked from the command line as well:

```bash
python agent.py "List the issue details for PROJ-1"
```

These examples require valid API keys for both OpenAI and Jira when using the respective features.
