# Jira AI Assistant

This repository contains small utilities for interacting with Jira issues. The `main.py` script prompts for an issue ID and prints the raw details retrieved from Jira.

## Configuration

Create a `.env` file based on `.env.example` and provide your Jira credentials:

```
JIRA_BASE_URL=https://your-domain.atlassian.net
JIRA_EMAIL=your-email@example.com
JIRA_API_TOKEN=your-api-token
```

## Usage

Run the script with Python:

```bash
python main.py
```

You will be asked for a Jira issue ID and the issue data will be printed to the terminal.
