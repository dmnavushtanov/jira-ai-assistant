# Jira AI Assistant

This repository contains a Jira AI assistant that communicates with the OpenAI API and provides utilities for interacting with Jira issues. It includes:

- Simple utilities for retrieving Jira issue details
- OpenAI service integration for AI-powered assistance
- A minimal agent that combines both capabilities

## Configuration

Create a `.env` file and provide your credentials:

```
JIRA_BASE_URL=https://your-domain.atlassian.net
JIRA_EMAIL=your-email@example.com
JIRA_API_TOKEN=your-api-token
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o-mini
BASE_LLM=openai  # or 'anthropic'
INCLUDE_WHOLE_API_BODY=false
LANGCHAIN_DEBUG=false
RICH_LOGGING=true
STRIP_UNUSED_JIRA_DATA=true
FOLLOW_RELATED_JIRAS=false
```

The default model and provider can be changed in `src/configs/config.yml` or by setting `OPENAI_MODEL` and `BASE_LLM` in the environment. The flag `INCLUDE_WHOLE_API_BODY` controls whether validation prompts should return the full API bodies or only boolean indicators of their validity.


Conversation memory can be enabled with `conversation_memory: true` in the same file. The number of previous questions remembered defaults to three and can be adjusted via `max_questions_to_remember`. When LangChain is available the agent uses a `ConversationBufferWindowMemory` of size `k`. If `k` is greater than three the buffer is combined with `ConversationSummaryMemory` so older turns are summarized automatically. When the limit is reached you'll be prompted to start a new conversation.

The assistant also remembers the last Jira key you referenced. Follow-up questions can omit the key and it will use the stored value. Include the word `forget` in your message to clear this memory.

Set `strip_unused_jira_data: true` in the config to remove avatar URLs and ID fields from Jira payloads for more concise outputs.
Set `follow_related_jiras: true` to automatically fetch and summarize linked issues and subtasks when answering questions.

### Debug Logging

When `DEBUG=true` the log level is set to ``DEBUG``. Colored output using the
`rich` library can be toggled with `RICH_LOGGING` (defaults to `true`). Any
tracebacks will also be displayed with syntax highlighting when rich logging is
enabled. The utility class `RichLogger` can be passed as a callback to LangChain
components for detailed, step-by-step logs. Set `LANGCHAIN_DEBUG=true` to enable
verbose logs from LangChain itself.

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

### FastAPI Server

A minimal FastAPI application is included to expose the assistant over HTTP. Start the server with:

```bash
uvicorn app:app --reload
```

The `/ask` endpoint accepts a JSON payload containing a `question` field and returns the answer from the `RouterAgent`.

