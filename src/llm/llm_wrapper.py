"""Simple wrapper to allow pluggable LLMs."""

from langchain.llms import OpenAI
import config


def get_llm():
    return OpenAI(api_key=config.OPENAI_API_KEY)
