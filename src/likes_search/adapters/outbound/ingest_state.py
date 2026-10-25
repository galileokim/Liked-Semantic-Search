from __future__ import annotations

from pathlib import Path

from likes_search.domain.models import IngestState
from likes_search.ports import IngestStateRepository


class JsonIngestStateRepository(IngestStateRepository):
    """A single JSON file next to the Chroma index. Missing or unreadable means
    "nothing recorded yet"."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> IngestState:
        try:
            return IngestState.model_validate_json(self._path.read_text())
        except (FileNotFoundError, ValueError):
            return IngestState()

    def save(self, state: IngestState) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(state.model_dump_json(indent=2))
        tmp.replace(self._path)
