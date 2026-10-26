from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from likes_search.domain.chunking import ChunkingConfig, chunk_video
from likes_search.domain.models import FailedVideo, IngestState
from likes_search.ports import (
    Embedder,
    IngestStateRepository,
    LikedVideoSource,
    TranscriptFetchError,
    TranscriptProvider,
    VectorStore,
)


@dataclass
class IngestReport:
    seen: int = 0
    skipped_existing: int = 0
    indexed: int = 0
    without_transcript: int = 0
    failed: list[str] = field(default_factory=list)
    blocked: int = 0  # subset of `failed` that YouTube refused (IP block / bot check)


class IngestLikedVideos:
    """Pull liked videos, fetch transcripts, chunk, embed, and store.

    Idempotent: videos already in the store are skipped, so re-running picks up new
    likes and retries previously failed fetches. Failures are recorded in the ingest
    state until a later run indexes the video.
    """

    def __init__(
        self,
        source: LikedVideoSource,
        transcripts: TranscriptProvider,
        embedder: Embedder,
        store: VectorStore,
        state: IngestStateRepository,
        chunking: ChunkingConfig = ChunkingConfig(),
    ) -> None:
        self._source = source
        self._transcripts = transcripts
        self._embedder = embedder
        self._store = store
        self._state = state
        self._chunking = chunking

    def run(
        self, limit: int | None = None, on_progress: Callable[[str], None] | None = None
    ) -> IngestReport:
        log = on_progress or (lambda _msg: None)
        report = IngestReport()
        existing = self._store.indexed_video_ids()
        previous = self._state.load()
        failed = {f.video_id: f for f in previous.failed if f.video_id not in existing}

        for video in self._source.iter_liked_videos(limit=limit):
            report.seen += 1
            if video.video_id in existing:
                report.skipped_existing += 1
                continue

            try:
                transcript = self._transcripts.fetch(video.video_id)
            except TranscriptFetchError as e:
                # Don't index metadata-only: leaving it out means the next run retries.
                report.failed.append(video.video_id)
                report.blocked += e.blocked
                failed[video.video_id] = FailedVideo(
                    video_id=video.video_id,
                    title=video.title,
                    channel=video.channel,
                    reason=e.reason,
                    blocked=e.blocked,
                )
                log(f"{'BLOCK' if e.blocked else 'FAIL '} {video.title} ({e.reason})")
                continue

            if transcript is None:
                report.without_transcript += 1

            chunks = chunk_video(video, transcript, self._chunking)
            embeddings = self._embedder.embed_documents([c.text for c in chunks])
            self._store.upsert(chunks, embeddings)
            failed.pop(video.video_id, None)
            report.indexed += 1
            note = "" if transcript else ", no transcript"
            log(f"OK    {video.title} ({len(chunks)} chunks{note})")

        total = self._source.count_liked()
        self._state.save(
            IngestState(
                total_liked=total if total is not None else previous.total_liked,
                last_run_at=datetime.now(UTC),
                failed=tuple(failed.values()),
            )
        )
        return report
