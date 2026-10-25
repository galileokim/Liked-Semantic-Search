from __future__ import annotations

from likes_search.domain.models import LibraryStatus
from likes_search.ports import IngestStateRepository, VectorStore


class GetLibraryStatus:
    """Combine the index (what loaded) with the ingest state (what failed, and the
    account's like count) into one breakdown of the liked videos."""

    def __init__(self, store: VectorStore, state: IngestStateRepository) -> None:
        self._store = store
        self._state = state

    def run(self) -> LibraryStatus:
        indexed = self._store.indexed_video_ids()
        with_transcript = len(self._store.video_ids_with_transcripts() & indexed)
        state = self._state.load()
        failed = [f for f in state.failed if f.video_id not in indexed]
        blocked = sum(f.blocked for f in failed)

        total = state.total_liked
        # Videos liked but never reached. Clamped because indexed videos you have since
        # unliked still count toward `indexed`.
        not_loaded = max(0, total - len(indexed) - len(failed)) if total is not None else 0
        return LibraryStatus(
            total_liked=total,
            indexed=len(indexed),
            with_transcript=with_transcript,
            metadata_only=len(indexed) - with_transcript,
            blocked=blocked,
            failed_other=len(failed) - blocked,
            not_loaded=not_loaded,
            last_run_at=state.last_run_at,
        )
