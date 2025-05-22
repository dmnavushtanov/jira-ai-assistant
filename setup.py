from setuptools import setup, find_packages

setup(
    name="jira-assistant",
    version="0.1.0",
    description="Jira AI Assistant using OpenAI",
    author="Dimitar Navushtanov",
    author_email="dimitar.navushtanov@fadata.eu",
    packages=find_packages(),
    install_requires=[
        "openai",
        "requests",
        "typer",
        "pydantic",
        "langchain",
        "python-dotenv",
    ],
    python_requires=">=3.8",
) 