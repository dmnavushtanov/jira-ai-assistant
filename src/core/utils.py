"""Utility helpers for Jira AI Assistant."""

from pathlib import Path
from typing import Any, Callable, Optional
import yaml


def format_timedelta(seconds: int) -> str:
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, _ = divmod(remainder, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    return " ".join(parts) if parts else "0m"


class HotReloader:
    """Load a file and automatically reload it if it changes."""

    def __init__(self, path: str, loader: Callable[[str], Any] = yaml.safe_load) -> None:
        self.path = Path(path)
        self.loader = loader
        self._mtime: Optional[float] = None
        self._data: Any = None
        self._load()

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as f:
            self._data = self.loader(f)
        self._mtime = self.path.stat().st_mtime

    def get(self) -> Any:
        """Return cached data, reloading if the file has changed."""
        mtime = self.path.stat().st_mtime
        if self._mtime is None or mtime != self._mtime:
            self._load()
        return self._data
