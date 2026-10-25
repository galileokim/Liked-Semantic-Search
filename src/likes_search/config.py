from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Override any field with an env var (LIKES_<FIELD>) or a .env file."""

    model_config = SettingsConfigDict(env_prefix="LIKES_", env_file=".env", extra="ignore")

    data_dir: Path = Path(".data")
    client_secrets_file: Path = Path("client_secret.json")

    # BGE is asymmetric: queries get an instruction prefix, documents don't.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = "Represent this sentence for searching relevant passages: "

    transcript_languages: list[str] = ["en"]
    transcript_delay_seconds: float = 1.0

    chunk_max_words: int = 200
    chunk_overlap_words: int = 40

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def ingest_state_file(self) -> Path:
        return self.data_dir / "ingest_state.json"

    @property
    def token_file(self) -> Path:
        return self.data_dir / "token.json"
