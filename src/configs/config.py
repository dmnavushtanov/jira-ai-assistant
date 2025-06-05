from dataclasses import dataclass
import os
import yaml
from dotenv import load_dotenv

@dataclass
class Config:
    base_llm: str
    openai_api_key: str
    openai_model: str
    anthropic_api_key: str
    anthropic_model: str


def load_config(path: str = None) -> Config:
    """Load configuration from YAML file and environment variables."""
    load_dotenv()
    
    # If no path provided, use default relative to this config.py file
    if path is None:
        config_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(config_dir, "config.yaml")
    
    data = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    return Config(
        base_llm=data.get("base_llm", os.getenv("BASE_LLM", "openai")),
        openai_api_key=os.getenv("OPENAI_API_KEY", data.get("openai_api_key", "")),
        openai_model=os.getenv("OPENAI_MODEL", data.get("openai_model", "gpt-3.5-turbo")),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", data.get("anthropic_api_key", "")),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", data.get("anthropic_model", "claude-3-opus")),
    )
