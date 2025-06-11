from dataclasses import dataclass
import os
import logging
import yaml
from dotenv import load_dotenv

try:
    from rich.logging import RichHandler
    from rich.traceback import install as install_rich_traceback
except Exception:  # pragma: no cover - rich not installed
    RichHandler = None
    def install_rich_traceback():
        pass

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
    projects: list[str]
    include_whole_api_body: bool
    langchain_debug: bool
    rich_logging: bool
    conversation_memory: bool
    max_questions_to_remember: int
    strip_unused_jira_data: bool
    follow_related_jiras: bool
    write_comments_to_jira: bool
    ask_for_confirmation: bool
    reuse_jira_client: bool
    validation_prompts_dir: str = "validation"


def setup_logging(config: "Config") -> None:
    """Configure logging level based on ``config.debug``."""
    level = logging.DEBUG if config.debug else logging.INFO
    use_rich = config.rich_logging and RichHandler is not None
    if use_rich:
        install_rich_traceback()
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[RichHandler(rich_tracebacks=True)],
        )
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    logger.debug("Logging initialized at level %s", logging.getLevelName(level))
    # Suppress noisy debug output from third-party HTTP clients
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        import langchain

        langchain.debug = config.langchain_debug
        logger.debug("LangChain debug mode set to %s", config.langchain_debug)
    except Exception:
        logger.debug("LangChain not installed; skipping debug configuration")


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

    def _env_int(name: str, default: int) -> int:
        val = os.getenv(name)
        if val is None:
            return int(data.get(name.lower(), default))
        try:
            return int(val)
        except ValueError:
            return default

    return Config(
        app_name=data.get("app_name", os.getenv("APP_NAME", "JiraAIAssistant")),
        environment=data.get("environment", os.getenv("ENVIRONMENT", "production")),
        debug=_env_bool("DEBUG", data.get("debug", False)),
        base_llm=data.get("base_llm", os.getenv("BASE_LLM", "openai")),
        openai_api_key=os.getenv("OPENAI_API_KEY", data.get("openai_api_key", "")),
        openai_model=os.getenv("OPENAI_MODEL", data.get("openai_model", "gpt-3.5-turbo")),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", data.get("anthropic_api_key", "")),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", data.get("anthropic_model", "claude-3-opus")),
        projects=[p.strip().upper() for p in os.getenv("PROJECTS", ",".join(data.get("projects", []))).split(",") if p.strip()] or [],
        include_whole_api_body=_env_bool("INCLUDE_WHOLE_API_BODY", data.get("include_whole_api_body", False)),
        langchain_debug=_env_bool("LANGCHAIN_DEBUG", data.get("langchain_debug", False)),
        rich_logging=_env_bool("RICH_LOGGING", data.get("rich_logging", True)),
        conversation_memory=_env_bool("CONVERSATION_MEMORY", data.get("conversation_memory", False)),
        max_questions_to_remember=_env_int("MAX_NUMBER_OF_QUESTIONS_TO_REMEMBER", data.get("max_questions_to_remember", 3)),
        strip_unused_jira_data=_env_bool("STRIP_UNUSED_JIRA_DATA", data.get("strip_unused_jira_data", False)),
        follow_related_jiras=_env_bool("FOLLOW_RELATED_JIRAS", data.get("follow_related_jiras", False)),
        write_comments_to_jira=_env_bool("WRITE_COMMENTS_TO_JIRA", data.get("write_comments_to_jira", False)),
        ask_for_confirmation=_env_bool("ASK_FOR_CONFIRMATION", data.get("ask_for_confirmation", False)),
        reuse_jira_client=_env_bool("REUSE_JIRA_CLIENT", data.get("reuse_jira_client", True)),
        validation_prompts_dir=os.getenv("VALIDATION_PROMPTS_DIR", data.get("validation_prompts_dir", "validation")),
    )
