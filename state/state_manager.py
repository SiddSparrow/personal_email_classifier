import json
import logging
from pathlib import Path

from core.interfaces import StateManager

logger = logging.getLogger(__name__)


class JsonStateManager(StateManager):
    """Persists processed email IDs in a JSON file."""

    def __init__(self, state_file: Path) -> None:
        self._state_file = state_file
        self._processed_ids: set[str] = set()

    def load(self) -> None:
        if self._state_file.exists():
            with open(self._state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._processed_ids = set(data.get("processed_ids", []))
            logger.info(f"Loaded {len(self._processed_ids)} processed IDs from state")
        else:
            self._processed_ids = set()
            logger.info("No state file found; starting fresh")

    def save(self) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_file, "w", encoding="utf-8") as f:
            json.dump({"processed_ids": sorted(self._processed_ids)}, f, indent=2)
        logger.debug(f"Saved {len(self._processed_ids)} processed IDs to state")

    def is_processed(self, message_id: str) -> bool:
        return message_id in self._processed_ids

    def mark_processed(self, message_id: str) -> None:
        self._processed_ids.add(message_id)
