from openai import OpenAI
import config
import subprocess

def main():
    # Initialize OpenAI client
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    print("OpenAI client initialized.")
    print("Hello, World!")

def start_mcp_atlassian_cli_session():
    """Starts the mcp-atlassian CLI tool with credentials from config.py."""
    
    command = [
        "mcp-atlassian",
        "--jira-url", config.JIRA_URL,
        "--jira-username", config.JIRA_USERNAME,
        "--jira-token", config.JIRA_API_TOKEN
    ]

    # Add Confluence arguments if the token is present (assuming if token is there, user wants to use it)
    confluence_token_to_redact = ""
    if hasattr(config, 'CONFLUENCE_API_TOKEN') and config.CONFLUENCE_API_TOKEN and config.CONFLUENCE_API_TOKEN != "YOUR_CONFLUENCE_API_TOKEN":
        if hasattr(config, 'CONFLUENCE_URL') and config.CONFLUENCE_URL and hasattr(config, 'CONFLUENCE_USERNAME') and config.CONFLUENCE_USERNAME:
            command.extend([
                "--confluence-url", config.CONFLUENCE_URL,
                "--confluence-username", config.CONfluence_USERNAME,
                "--confluence-token", config.CONFLUENCE_API_TOKEN
            ])
            confluence_token_to_redact = config.CONFLUENCE_API_TOKEN
        else:
            print("Confluence API token found, but URL or Username is missing in config.py. Skipping Confluence setup.")

    print(f"Starting mcp-atlassian CLI with command: {' '.join(command).replace(config.JIRA_API_TOKEN, '[JIRA_TOKEN_REDACTED]').replace(confluence_token_to_redact, '[CONFLUENCE_TOKEN_REDACTED]' if confluence_token_to_redact else 'NOT_REPLACING_CONFLUENCE_BECAUSE_EMPTY')}")
    print("Output from mcp-atlassian will be shown below:")

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)

        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(f"[mcp-atlassian STDOUT]: {line.strip()}", flush=True)
                if process.poll() is not None:
                    break
        
        if process.stderr:
            for line in iter(process.stderr.readline, ''):
                if line:
                    print(f"[mcp-atlassian STDERR]: {line.strip()}", flush=True)
                if process.poll() is not None:
                    break

        process.wait()

        if process.returncode == 0:
            print("mcp-atlassian process finished.")
        else:
            print(f"mcp-atlassian process failed with return code {process.returncode}.")
            if process.stderr:
                remaining_stderr = process.stderr.read()
                if remaining_stderr:
                    print(f"[mcp-atlassian STDERR]: {remaining_stderr.strip()}", flush=True)

    except FileNotFoundError:
        print("Error: 'mcp-atlassian' command not found. Please ensure it is installed and in your system's PATH.")
        print("You might need to install it from: https://github.com/sooperset/mcp-atlassian")
    except Exception as e:
        print(f"An error occurred while trying to start mcp-atlassian: {e}")

if __name__ == "__main__":
    main()
    start_mcp_atlassian_cli_session() 