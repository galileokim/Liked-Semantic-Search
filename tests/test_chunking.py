import pytest

from likes_search.domain.chunking import ChunkingConfig, chunk_video
from likes_search.domain.models import ChunkKind, Transcript, TranscriptSegment, Video

VIDEO = Video(video_id="abc", title="Title", channel="Chan", description="Desc", tags=("x", "y"))


def _transcript(n_segments: int, words_per_segment: int = 10) -> Transcript:
    return Transcript(
        video_id="abc",
        language="en",
        segments=tuple(
            TranscriptSegment(
                text=" ".join(f"w{i}_{j}" for j in range(words_per_segment)),
                start=float(i * 5),
                duration=5.0,
            )
            for i in range(n_segments)
        ),
    )


def test_metadata_chunk_only_when_no_transcript() -> None:
    chunks = chunk_video(VIDEO, None)
    assert len(chunks) == 1
    assert chunks[0].kind is ChunkKind.METADATA
    assert "Title" in chunks[0].text and "Tags: x, y" in chunks[0].text
    assert chunks[0].start_seconds is None


def test_transcript_windows_respect_max_words_and_overlap() -> None:
    config = ChunkingConfig(max_words=50, overlap_words=20)
    chunks = chunk_video(VIDEO, _transcript(20), config)[1:]

    assert all(len(c.text.split()) <= 50 for c in chunks)
    # Consecutive windows share the overlap tail (2 segments of 10 words).
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-20:] == second[:20]
    assert second[0] == "w3_0"
    assert chunks[1].start_seconds == 15.0


def test_every_segment_is_covered() -> None:
    chunks = chunk_video(VIDEO, _transcript(37), ChunkingConfig(max_words=60, overlap_words=10))
    text = " ".join(c.text for c in chunks)
    assert all(f"w{i}_0" in text for i in range(37))


def test_chunk_ids_are_stable_and_unique() -> None:
    ids = [c.chunk_id for c in chunk_video(VIDEO, _transcript(30))]
    assert ids == [c.chunk_id for c in chunk_video(VIDEO, _transcript(30))]
    assert len(ids) == len(set(ids))


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(max_words=10, overlap_words=10)
