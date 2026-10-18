from fakes import (
    FakeLikedVideoSource,
    FakeTranscriptProvider,
    HashingEmbedder,
    InMemoryVectorStore,
)
from likes_search.application.ingest import IngestLikedVideos
from likes_search.domain.models import Transcript, TranscriptSegment, Video

VIDEOS = [
    Video(video_id="cook", title="Weeknight dinner ideas", channel="Kitchen"),
    Video(video_id="gpu", title="Building a PC", channel="Tech"),
    Video(video_id="music", title="Lofi beats to study to", channel="Chill"),
    Video(video_id="blocked", title="Some video", channel="Other"),
]

TRANSCRIPTS = {
    "cook": Transcript(
        video_id="cook",
        language="en",
        segments=(
            TranscriptSegment(text="today we make a garlic butter pasta", start=0.0, duration=4),
            TranscriptSegment(text="add the parmesan and stir slowly", start=42.0, duration=4),
        ),
    ),
    "gpu": Transcript(
        video_id="gpu",
        language="en",
        segments=(
            TranscriptSegment(
                text="seat the graphics card in the pcie slot", start=0.0, duration=4
            ),
        ),
    ),
}


def _ingest(store: InMemoryVectorStore) -> IngestLikedVideos:
    return IngestLikedVideos(
        source=FakeLikedVideoSource(VIDEOS),
        transcripts=FakeTranscriptProvider(TRANSCRIPTS, failing=frozenset({"blocked"})),
        embedder=HashingEmbedder(),
        store=store,
    )


def test_ingest_reports_and_skips_failed_videos() -> None:
    store = InMemoryVectorStore()
    report = _ingest(store).run()

    assert report.seen == 4
    assert report.indexed == 3
    assert report.without_transcript == 1  # music
    assert report.failed == ["blocked"]
    assert store.indexed_video_ids() == {"cook", "gpu", "music"}


def test_ingest_is_idempotent() -> None:
    store = InMemoryVectorStore()
    _ingest(store).run()
    second = _ingest(store).run()
    assert second.indexed == 0
    assert second.skipped_existing == 3
