"""Simple interactive entry point for Jira AI Assistant."""

from src.services import JiraService


def main() -> None:
    issue_key = input("Enter Jira issue key: ").strip()
    if not issue_key:
        print("No issue key provided.")
        return

    service = JiraService()
    description = service.get_issue_description(issue_key)
    if description:
        print(f"Description for {issue_key}:\n{description}")
    else:
        print(f"No description found for {issue_key}.")


if __name__ == "__main__":
    main()
