"""In-memory port implementations: let the application layer be tested with no network,
no model download, and no database."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterator, Sequence

from likes_search.domain.models import (
    Chunk,
    ChunkHit,
    ChunkKind,
    IngestState,
    Transcript,
    Video,
)
from likes_search.ports import (
    Embedder,
    IngestStateRepository,
    LikedVideoSource,
    TranscriptFetchError,
    TranscriptProvider,
    Vector,
    VectorStore,
)


class FakeLikedVideoSource(LikedVideoSource):
    def __init__(self, videos: Sequence[Video]) -> None:
        self._videos = list(videos)

    def iter_liked_videos(self, limit: int | None = None) -> Iterator[Video]:
        yield from self._videos[:limit]

    def count_liked(self) -> int | None:
        return len(self._videos)


class FakeTranscriptProvider(TranscriptProvider):
    def __init__(
        self,
        transcripts: dict[str, Transcript],
        failing: frozenset[str] = frozenset(),
        blocked: frozenset[str] = frozenset(),
    ) -> None:
        self._transcripts = transcripts
        self._failing = failing
        self._blocked = blocked

    def fetch(self, video_id: str) -> Transcript | None:
        if video_id in self._blocked:
            raise TranscriptFetchError("IpBlocked", blocked=True)
        if video_id in self._failing:
            raise TranscriptFetchError("simulated failure")
        return self._transcripts.get(video_id)


class HashingEmbedder(Embedder):
    """Deterministic bag-of-words embedding: shared words => higher cosine similarity."""

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self._dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self.records: dict[str, tuple[Chunk, list[float]]] = {}

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Vector]) -> None:
        for chunk, emb in zip(chunks, embeddings, strict=True):
            self.records[chunk.chunk_id] = (chunk, list(emb))

    def query(self, embedding: Vector, k: int) -> list[ChunkHit]:
        hits = [
            ChunkHit(chunk=chunk, score=sum(a * b for a, b in zip(embedding, emb, strict=True)))
            for chunk, emb in self.records.values()
        ]
        return sorted(hits, key=lambda h: h.score, reverse=True)[:k]

    def indexed_video_ids(self) -> set[str]:
        return {c.video_id for c, _ in self.records.values() if c.kind is ChunkKind.METADATA}

    def indexed_video_channels(self) -> dict[str, str]:
        return {
            c.video_id: c.channel for c, _ in self.records.values() if c.kind is ChunkKind.METADATA
        }

    def video_ids_with_transcripts(self) -> set[str]:
        return {c.video_id for c, _ in self.records.values() if c.kind is ChunkKind.TRANSCRIPT}


class InMemoryIngestStateRepository(IngestStateRepository):
    def __init__(self) -> None:
        self.state = IngestState()

    def load(self) -> IngestState:
        return self.state

    def save(self, state: IngestState) -> None:
        self.state = state
