import logging
from typing import Any, Optional

from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)


def parse_json_block(text: str) -> Optional[Any]:
    """Return parsed JSON from ``text`` using ``JsonOutputParser``.

    The LangChain parser handles code fences and minor formatting issues
    so custom regex extraction is no longer required.
    """
    if not isinstance(text, str):
        return None

    cleaned = text.strip()

    parser = JsonOutputParser()
    try:
        return parser.parse(cleaned)
    except Exception:
        logger.debug("JsonOutputParser failed", exc_info=True)
    return None
