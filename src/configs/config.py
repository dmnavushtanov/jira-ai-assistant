from dataclasses import dataclass
import os
import logging
import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class Config:
    app_name: str
    environment: str
    debug: bool
    base_llm: str
    openai_api_key: str
    openai_model: str
    anthropic_api_key: str
    anthropic_model: str


def setup_logging(config: "Config") -> None:
    """Configure logging level based on ``config.debug``."""
    level = logging.DEBUG if config.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.debug("Logging initialized at level %s", logging.getLevelName(level))


def load_config(path: str = None) -> Config:
    """Load configuration from YAML file and environment variables."""
    load_dotenv()
    logger.debug("Loading configuration from %s", path or "default config.yml")
    
    # If no path provided, use default relative to this config.py file
    if path is None:
        config_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(config_dir, "config.yml")
    
    data = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        logger.debug("Loaded YAML configuration from %s", path)

    def _env_bool(name: str, default: bool) -> bool:
        val = os.getenv(name)
        if val is None:
            return bool(data.get(name.lower(), default))
        return val.lower() in {"1", "true", "yes", "on"}

    return Config(
        app_name=data.get("app_name", os.getenv("APP_NAME", "JiraAIAssistant")),
        environment=data.get("environment", os.getenv("ENVIRONMENT", "production")),
        debug=_env_bool("DEBUG", data.get("debug", False)),
        base_llm=data.get("base_llm", os.getenv("BASE_LLM", "openai")),
        openai_api_key=os.getenv("OPENAI_API_KEY", data.get("openai_api_key", "")),
        openai_model=os.getenv("OPENAI_MODEL", data.get("openai_model", "gpt-3.5-turbo")),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", data.get("anthropic_api_key", "")),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", data.get("anthropic_model", "claude-3-opus")),
    )
