from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from likes_search.domain.models import Chunk, ChunkKind, Transcript, TranscriptSegment, Video

_MAX_DESCRIPTION_CHARS = 1500


@dataclass(frozen=True)
class ChunkingConfig:
    max_words: int = 200
    overlap_words: int = 40

    def __post_init__(self) -> None:
        if not 0 <= self.overlap_words < self.max_words:
            raise ValueError("overlap_words must be in [0, max_words)")


def chunk_video(
    video: Video, transcript: Transcript | None, config: ChunkingConfig = ChunkingConfig()
) -> list[Chunk]:
    """One metadata chunk (always) plus overlapping transcript windows (when available).

    The metadata chunk means videos without transcripts are still searchable by
    title/description/tags.
    """
    chunks = [_metadata_chunk(video)]
    if transcript is not None:
        chunks.extend(_transcript_chunks(video, transcript.segments, config))
    return chunks


def _metadata_chunk(video: Video) -> Chunk:
    parts = [video.title, f"Channel: {video.channel}"]
    if video.tags:
        parts.append("Tags: " + ", ".join(video.tags))
    if video.description:
        parts.append(video.description[:_MAX_DESCRIPTION_CHARS])
    return Chunk(
        chunk_id=f"{video.video_id}:meta",
        video_id=video.video_id,
        title=video.title,
        channel=video.channel,
        kind=ChunkKind.METADATA,
        text="\n".join(parts),
    )


def _transcript_chunks(
    video: Video, segments: Sequence[TranscriptSegment], config: ChunkingConfig
) -> list[Chunk]:
    """Greedy word-count windows at segment granularity, carrying a tail of
    ~overlap_words into the next window so phrases spanning a boundary aren't lost."""
    windows: list[list[TranscriptSegment]] = []
    window: list[TranscriptSegment] = []
    words = 0
    for seg in segments:
        n = _word_count(seg)
        if n == 0:
            continue
        if window and words + n > config.max_words:
            windows.append(window)
            window = _tail(window, config.overlap_words)
            words = sum(_word_count(s) for s in window)
        window.append(seg)
        words += n
    if window:
        windows.append(window)

    return [
        Chunk(
            chunk_id=f"{video.video_id}:t{i}",
            video_id=video.video_id,
            title=video.title,
            channel=video.channel,
            kind=ChunkKind.TRANSCRIPT,
            text=" ".join(s.text.strip() for s in w),
            start_seconds=w[0].start,
        )
        for i, w in enumerate(windows)
    ]


def _tail(window: list[TranscriptSegment], max_words: int) -> list[TranscriptSegment]:
    tail: list[TranscriptSegment] = []
    words = 0
    for seg in reversed(window):
        n = _word_count(seg)
        if words + n > max_words:
            break
        tail.append(seg)
        words += n
    tail.reverse()
    return tail


def _word_count(seg: TranscriptSegment) -> int:
    return len(seg.text.split())
