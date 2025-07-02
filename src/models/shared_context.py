from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class SharedContext:
    """Simple container for passing results between agents."""

    validation_result: Optional[str] = None
    generated_tests: Optional[str] = None
    operation_outcome: Dict[str, Any] = field(default_factory=dict)

    def clear(self) -> None:
        self.validation_result = None
        self.generated_tests = None
        self.operation_outcome.clear()

