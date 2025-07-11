# Jira AI Assistant

This repository contains a Jira AI assistant that communicates with the OpenAI API and provides utilities for interacting with Jira issues. It includes:

- Simple utilities for retrieving Jira issue details
- OpenAI service integration for AI-powered assistance
- A minimal agent that combines both capabilities
- Automated test case generation for API issues

This project requires the LangChain library and related packages. Install all
dependencies with:

```bash
pip install -r requirements.txt
```

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
LOG_JIRA_PAYLOADS=true
```

The default model and provider can be changed in `src/configs/config.yml` or by setting `OPENAI_MODEL` and `BASE_LLM` in the environment. The flag `INCLUDE_WHOLE_API_BODY` controls whether validation prompts should return the full API bodies or only boolean indicators of their validity.


Conversation memory can be enabled with `conversation_memory: true` in the same file. The number of previous questions remembered defaults to three and can be adjusted via `max_questions_to_remember`. The agent uses a LangChain `ConversationBufferWindowMemory` of size `k`. If `k` is greater than three the buffer is combined with `ConversationSummaryMemory` so older turns are summarized automatically. When the limit is reached you'll be prompted to start a new conversation. If memory is disabled the session is cleared automatically once the limit is reached and a short notice is returned before the answer.

The assistant also remembers the last Jira key you referenced. Follow-up questions can omit the key and it will use the stored value. Include the word `forget` in your message to clear this memory.

Set `strip_unused_jira_data: true` in the config to remove avatar URLs and ID fields from Jira payloads for more concise outputs.
Set `follow_related_jiras: true` to automatically fetch and summarize linked issues and subtasks when answering questions. Comments from those related tickets are also retrieved so important context isn't missed.
Set `log_jira_payloads: false` to disable logging raw Jira API payloads returned by the tools. When disabled, some tools such as `transition_issue` return only minimal data to avoid large debug logs.

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
python src/services/openai_service.py "What is 2 + 2?"
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

### Test Case Generation

Ask the assistant for test cases and it will attempt to generate them directly. Validation can be run separately if needed. The generation step first checks the description for existing tests and informs you when they are already present instead of generating new ones. The **TestAgent** runs a planning pipeline that detects the HTTP method and summarizes the context before generating tests. If the method cannot be identified, the default prompt is used. If the ticket lacks enough details the assistant will reply "Not enough information to generate test cases." otherwise it returns basic scenarios. A ReAct agent is also created from these tools so the pipeline can be invoked externally when needed.

When tests are successfully generated they are appended to the end of the issue's
**Description** field using the Jira API.

```bash
python main.py "Generate test cases for RB-1234"
```

### Issue Creation

`IssueCreatorAgent` plans Jira tickets from a free-form request. It extracts the
summary, description and issue type (Task, Story, Bug or Sub-task). When
creating a sub-task a parent key is required. If missing the agent will ask for
it.

```python
from src.agents import IssueCreatorAgent
creator = IssueCreatorAgent()
result = creator.create_issue(
    "Create a bug: API returns 500 when posting to /api/books", "RB"
)
print(result)
```

When used with the `RouterAgent` a request to create an issue automatically
invokes this agent through the planning pipeline.

### Multi-step Operations

The `PlanningAgent` can break down a request into multiple Jira actions. For
example asking to add a comment and then move the ticket will produce a plan
containing both steps. The router executes each step in order and reports which
actions succeeded or failed. Results from each step are stored and may be
referenced by later steps using placeholders like `$step1` or
`$step1.field`. Tasks are executed strictly in the planned sequence.

### Error Handling

The assistant now catches common issues such as invalid Jira keys or failures
communicating with OpenAI. Instead of a stack trace you will receive a polite
message explaining what went wrong so you can try again later.

### Simple HTTP Client

The ``SimpleHttpClient`` utility provides a minimal wrapper around
``requests`` for quickly exercising external APIs.  It supports basic
``get``, ``post``, ``put`` and ``delete`` helpers and can be initialized
with an optional ``base_url``.  This makes it easy to experiment with new
endpoints directly from Python without pulling in a larger framework.

