from dataclasses import dataclass
import os
import yaml
from dotenv import load_dotenv

@dataclass
class Config:
    openai_api_key: str
    openai_model: str


def load_config(path: str = "config.yaml") -> Config:
    """Load configuration from YAML file and environment variables."""
    load_dotenv()
    data = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    return Config(
        openai_api_key=os.getenv("OPENAI_API_KEY", data.get("openai_api_key", "")),
        openai_model=os.getenv("OPENAI_MODEL", data.get("openai_model", "gpt-3.5-turbo")),
    )
