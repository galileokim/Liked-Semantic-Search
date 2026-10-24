from __future__ import annotations

from likes_search.domain.models import ChunkHit, MatchedMoment, VideoMatch
from likes_search.ports import Embedder, VectorStore

_MAX_OTHER_MOMENTS = 3


class SearchLikedVideos:
    """Chunk-level retrieval, aggregated to one result per video.

    We over-fetch chunks (k * candidate_multiplier) because several top chunks often
    belong to the same video; each video is scored by its best chunk (max-pooling),
    and that chunk's text/timestamp become the snippet and deep link. The video's other
    candidate chunks are kept as evidence for why it matched.
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

        by_video: dict[str, list[ChunkHit]] = {}
        for hit in hits:
            by_video.setdefault(hit.chunk.video_id, []).append(hit)
        for group in by_video.values():
            group.sort(key=lambda h: h.score, reverse=True)

        ranked = sorted(by_video.values(), key=lambda g: g[0].score, reverse=True)[:k]
        return [_to_match(group) for group in ranked]


def _to_match(group: list[ChunkHit]) -> VideoMatch:
    best, rest = group[0], group[1:]
    others = tuple(
        MatchedMoment(start_seconds=h.chunk.start_seconds, score=h.score)
        for h in rest
        if h.chunk.start_seconds is not None and h.chunk.start_seconds != best.chunk.start_seconds
    )[:_MAX_OTHER_MOMENTS]
    return VideoMatch(
        video_id=best.chunk.video_id,
        title=best.chunk.title,
        channel=best.chunk.channel,
        score=best.score,
        snippet=best.chunk.text,
        start_seconds=best.chunk.start_seconds,
        match_kind=best.chunk.kind,
        matched_chunks=len(group),
        other_moments=others,
    )
