"""Composition root: the only place that knows which adapter backs which port.

Heavy adapters are built lazily so e.g. `likes search` never triggers OAuth and
`likes auth` never loads the embedding model.
"""

from __future__ import annotations

from functools import cached_property

from likes_search.adapters.outbound.chroma_store import ChromaVectorStore
from likes_search.adapters.outbound.embedder import SentenceTransformerEmbedder
from likes_search.adapters.outbound.transcripts import YouTubeTranscriptApiProvider
from likes_search.adapters.outbound.youtube_api import (
    YouTubeApiLikedVideoSource,
    load_credentials,
)
from likes_search.application.ingest import IngestLikedVideos
from likes_search.application.search import SearchLikedVideos
from likes_search.config import Settings
from likes_search.domain.chunking import ChunkingConfig
from likes_search.ports import Embedder, VectorStore


class Container:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    @cached_property
    def embedder(self) -> Embedder:
        return SentenceTransformerEmbedder(
            self.settings.embedding_model, query_prefix=self.settings.query_prefix
        )

    @cached_property
    def store(self) -> VectorStore:
        return ChromaVectorStore(self.settings.chroma_path)

    def authenticate(self) -> None:
        load_credentials(self.settings.client_secrets_file, self.settings.token_file)

    def ingest(self) -> IngestLikedVideos:
        creds = load_credentials(self.settings.client_secrets_file, self.settings.token_file)
        return IngestLikedVideos(
            source=YouTubeApiLikedVideoSource(creds),
            transcripts=YouTubeTranscriptApiProvider(
                languages=self.settings.transcript_languages,
                delay_seconds=self.settings.transcript_delay_seconds,
            ),
            embedder=self.embedder,
            store=self.store,
            chunking=ChunkingConfig(
                max_words=self.settings.chunk_max_words,
                overlap_words=self.settings.chunk_overlap_words,
            ),
        )

    def search(self) -> SearchLikedVideos:
        return SearchLikedVideos(self.embedder, self.store)
