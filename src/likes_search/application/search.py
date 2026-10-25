from __future__ import annotations

from likes_search.domain.models import ChunkHit, VideoMatch
from likes_search.ports import Embedder, VectorStore


class SearchLikedVideos:
    """Chunk-level retrieval, aggregated to one result per video.

    We over-fetch chunks (k * candidate_multiplier) because several top chunks often
    belong to the same video; each video is scored by its best chunk (max-pooling),
    and that chunk's text/timestamp become the snippet and deep link.
    """

    def __init__(
        self, embedder: Embedder, store: VectorStore, candidate_multiplier: int = 5
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._candidate_multiplier = candidate_multiplier

    def run(self, query: str, k: int = 10) -> list[VideoMatch]:
        query = query.strip()
        if not query or k <= 0:
            return []

        hits = self._store.query(self._embedder.embed_query(query), k * self._candidate_multiplier)

        best: dict[str, ChunkHit] = {}
        for hit in hits:
            current = best.get(hit.chunk.video_id)
            if current is None or hit.score > current.score:
                best[hit.chunk.video_id] = hit

        ranked = sorted(best.values(), key=lambda h: h.score, reverse=True)[:k]
        return [
            VideoMatch(
                video_id=h.chunk.video_id,
                title=h.chunk.title,
                channel=h.chunk.channel,
                score=h.score,
                snippet=h.chunk.text,
                start_seconds=h.chunk.start_seconds,
            )
            for h in ranked
        ]
