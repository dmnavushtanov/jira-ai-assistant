# Jira AI Assistant

An AI-powered assistant for interacting with Jira using natural language.

## Installation

1. Clone the repository:
   ```
   git clone <your-repository-url>
   cd jira-assistant
   ```

2. Install the package in development mode:
   ```
   pip install -e .
   ```

3. Create a `.env` file in the project root with your credentials:
   ```
   JIRA_URL=https://your-domain.atlassian.net/
   OPENAI_API_KEY=your_openai_api_key
   JIRA_USERNAME=your.email@example.com
   JIRA_API_TOKEN=your_jira_api_token
   OPENAI_MODEL=gpt-3.5-turbo
   ```

4. Verify your environment configuration:
   ```
   python check_env.py
   ```

## Usage

Use the CLI to interact with Jira:

```
python main.py hello
```

## Project Structure

- `adapters/`: Low-level API clients
- `agents/`: LLM-powered agents
- `cli/`: Command-line interface
- `config/`: Configuration files
- `core/`: Core utilities
- `llm/`: LLM wrappers
- `models/`: Data models
- `services/`: Business logic services
