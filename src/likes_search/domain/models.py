from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, computed_field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Video(_Frozen):
    video_id: str
    title: str
    channel: str
    description: str = ""
    tags: tuple[str, ...] = ()
    published_at: datetime | None = None


class TranscriptSegment(_Frozen):
    text: str
    start: float
    duration: float


class Transcript(_Frozen):
    video_id: str
    language: str
    segments: tuple[TranscriptSegment, ...]


class ChunkKind(StrEnum):
    METADATA = "metadata"
    TRANSCRIPT = "transcript"


class Chunk(_Frozen):
    """The unit that gets embedded. Video title/channel are denormalized onto every chunk
    so search results can be rendered straight from the vector store."""

    chunk_id: str
    video_id: str
    title: str
    channel: str
    kind: ChunkKind
    text: str
    start_seconds: float | None = None


class ChunkHit(_Frozen):
    chunk: Chunk
    score: float  # cosine similarity; higher is better


class MatchedMoment(_Frozen):
    """Another passage of the same video that also ranked among the candidates."""

    start_seconds: float
    score: float


class VideoMatch(_Frozen):
    video_id: str
    title: str
    channel: str
    score: float
    snippet: str
    start_seconds: float | None = None
    match_kind: ChunkKind = ChunkKind.TRANSCRIPT
    # Evidence for "why this matched": how many of the candidate chunks came from this
    # video, and its other matching passages (best first), beyond the one in `snippet`.
    matched_chunks: int = 1
    other_moments: tuple[MatchedMoment, ...] = ()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def url(self) -> str:
        base = f"https://www.youtube.com/watch?v={self.video_id}"
        if self.start_seconds is None:
            return base
        return f"{base}&t={int(self.start_seconds)}s"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def thumbnail_url(self) -> str:
        # YouTube serves thumbnails at a fixed URL per video; mqdefault is 320x180.
        return f"https://i.ytimg.com/vi/{self.video_id}/mqdefault.jpg"
