from __future__ import annotations

from collections.abc import Sequence

from sentence_transformers import SentenceTransformer

from likes_search.ports import Embedder


class SentenceTransformerEmbedder(Embedder):
    """Local Hugging Face embedding model. Vectors are L2-normalized so cosine similarity
    and dot product agree."""

    def __init__(self, model_name: str, query_prefix: str = "", batch_size: int = 32) -> None:
        self._model = SentenceTransformer(model_name)
        self._query_prefix = query_prefix
        self._batch_size = batch_size

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts), batch_size=self._batch_size, normalize_embeddings=True
        )
        return [[float(x) for x in v] for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(self._query_prefix + text, normalize_embeddings=True)
        return [float(x) for x in vector]
