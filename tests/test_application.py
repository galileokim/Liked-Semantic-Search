from fakes import (
    FakeLikedVideoSource,
    FakeTranscriptProvider,
    HashingEmbedder,
    InMemoryVectorStore,
)
from likes_search.application.evaluate import EvalCase, evaluate
from likes_search.application.ingest import IngestLikedVideos
from likes_search.application.search import SearchLikedVideos
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


def test_search_returns_one_result_per_video_with_deep_link() -> None:
    store = InMemoryVectorStore()
    _ingest(store).run()
    results = SearchLikedVideos(HashingEmbedder(), store).run("garlic butter pasta", k=3)

    assert results[0].video_id == "cook"
    assert len({r.video_id for r in results}) == len(results)
    assert results[0].start_seconds == 0.0
    assert results[0].url.endswith("&t=0s")


def test_search_finds_video_without_transcript_via_metadata() -> None:
    store = InMemoryVectorStore()
    _ingest(store).run()
    results = SearchLikedVideos(HashingEmbedder(), store).run("lofi beats study", k=1)
    assert results[0].video_id == "music"
    assert results[0].start_seconds is None


def test_evaluate_metrics() -> None:
    store = InMemoryVectorStore()
    _ingest(store).run()
    result = evaluate(
        SearchLikedVideos(HashingEmbedder(), store),
        [
            EvalCase(query="graphics card pcie", expected_video_id="gpu"),
            EvalCase(query="parmesan pasta", expected_video_id="cook"),
            EvalCase(query="anything", expected_video_id="does-not-exist"),
        ],
        k=2,
    )
    assert result.recall_at_k == 2 / 3
    assert result.misses == ["anything"]
