from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import chromadb
import numpy as np

from likes_search.domain.models import Chunk, ChunkHit, ChunkKind
from likes_search.ports import Vector, VectorStore

# Chroma metadata values can't be None.
_NO_TIMESTAMP = -1.0


class ChromaVectorStore(VectorStore):
    def __init__(self, path: Path, collection_name: str = "liked_videos") -> None:
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Vector]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=np.asarray(embeddings, dtype=np.float32),
            documents=[c.text for c in chunks],
            metadatas=[_to_metadata(c) for c in chunks],
        )

    def query(self, embedding: Vector, k: int) -> list[ChunkHit]:
        count = self._collection.count()
        if count == 0 or k <= 0:
            return []
        result = self._collection.query(
            query_embeddings=np.asarray([embedding], dtype=np.float32),
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = (result["documents"] or [[]])[0]
        metadatas = (result["metadatas"] or [[]])[0]
        distances = (result["distances"] or [[]])[0]
        return [
            ChunkHit(chunk=_from_record(chunk_id, doc, meta), score=1.0 - dist)
            for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances, strict=True)
        ]

    def indexed_video_ids(self) -> set[str]:
        # Every indexed video has exactly one metadata chunk.
        result = self._collection.get(
            where={"kind": ChunkKind.METADATA.value}, include=["metadatas"]
        )
        return {str(m["video_id"]) for m in result["metadatas"] or []}


def _to_metadata(chunk: Chunk) -> dict[str, str | float]:
    return {
        "video_id": chunk.video_id,
        "title": chunk.title,
        "channel": chunk.channel,
        "kind": chunk.kind.value,
        "start_seconds": _NO_TIMESTAMP if chunk.start_seconds is None else chunk.start_seconds,
    }


def _from_record(chunk_id: str, document: str, meta: Mapping[str, object]) -> Chunk:
    raw_start = meta.get("start_seconds")
    start = float(raw_start) if isinstance(raw_start, int | float) else _NO_TIMESTAMP
    return Chunk(
        chunk_id=chunk_id,
        video_id=str(meta["video_id"]),
        title=str(meta.get("title", "")),
        channel=str(meta.get("channel", "")),
        kind=ChunkKind(str(meta["kind"])),
        text=document,
        start_seconds=None if start < 0 else start,
    )
