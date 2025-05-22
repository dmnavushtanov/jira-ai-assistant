#!/usr/bin/env python3
"""
Simple script to check if environment variables are properly loaded.
Run this script to verify your configuration is working.
"""

import config

def main():
    """Print configuration values to verify they're loaded correctly."""
    print("\n=== Jira AI Assistant Configuration ===\n")
    
    # Check Jira configuration
    print(f"JIRA_URL: {config.JIRA_URL}")
    print(f"JIRA_USERNAME: {config.JIRA_USERNAME}")
    print(f"JIRA_API_TOKEN: {'*' * 8 + config.JIRA_API_TOKEN[-4:] if config.JIRA_API_TOKEN else 'Not set'}")
    
    # Check OpenAI configuration
    print(f"OPENAI_MODEL: {config.OPENAI_MODEL}")
    print(f"OPENAI_API_KEY: {'*' * 8 + config.OPENAI_API_KEY[-4:] if config.OPENAI_API_KEY else 'Not set'}")
    
    print("\nIf any values are missing or incorrect, please check your .env file or environment variables.")
    print("Remember to run 'pip install python-dotenv' if you haven't already.")

if __name__ == "__main__":
    main() 