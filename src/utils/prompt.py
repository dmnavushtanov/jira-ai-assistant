import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def safe_format(template: str, values: Dict[str, Any]) -> str:
    """Return ``template`` with ``values`` substituted, ignoring other braces."""
    try:
        return template.format(**values)
    except Exception:
        logger.debug(
            "Falling back to manual placeholder replacement", exc_info=True
        )
        result = template
        for key, val in values.items():
            result = result.replace(f"{{{key}}}", str(val))
        return result
