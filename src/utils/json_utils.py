import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def parse_json_block(text: str) -> Optional[Any]:
    """Return parsed JSON from ``text`` which may include markdown fences."""
    if not isinstance(text, str):
        return None

    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            logger.debug("Failed to parse JSON block", exc_info=True)
    return None
