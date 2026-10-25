"""Outbound ports: the interfaces the application core depends on.

Adapters (YouTube, Chroma, sentence-transformers, ...) implement these; the core never
imports an adapter. Swapping Chroma for Azure AI Search, or a local model for an
Azure OpenAI embedding deployment, means writing one new adapter class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

from likes_search.domain.models import Chunk, ChunkHit, IngestState, Transcript, Video

Vector = Sequence[float]


class TranscriptFetchError(Exception):
    """Transient/unexpected failure (rate limit, IP block, network). The video should be
    retried on a later run, unlike a video that simply has no transcript.

    `blocked` marks YouTube refusing the request (IP block / bot check), as opposed to
    any other failure.
    """

    def __init__(self, reason: str, blocked: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.blocked = blocked


class LikedVideoSource(ABC):
    @abstractmethod
    def iter_liked_videos(self, limit: int | None = None) -> Iterator[Video]:
        """Yield liked videos, most recently liked first."""


class IngestStateRepository(ABC):
    """Persists what the vector store can't tell us: the account's like count and the
    videos whose transcript fetch failed (those are deliberately not indexed)."""

    @abstractmethod
    def load(self) -> IngestState: ...

    @abstractmethod
    def save(self, state: IngestState) -> None: ...


class TranscriptProvider(ABC):
    @abstractmethod
    def fetch(self, video_id: str) -> Transcript | None:
        """Return the transcript, or None if the video has none.

        Raises TranscriptFetchError for failures worth retrying.
        """


class Embedder(ABC):
    """Documents and queries are embedded separately because many retrieval models
    (BGE, E5, ...) are asymmetric and expect a query-side instruction prefix."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Vector]) -> None: ...

    @abstractmethod
    def query(self, embedding: Vector, k: int) -> list[ChunkHit]:
        """Top-k chunks by similarity, best first."""

    @abstractmethod
    def indexed_video_ids(self) -> set[str]: ...
