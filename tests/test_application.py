from fakes import (
    FakeLikedVideoSource,
    FakeTranscriptProvider,
    HashingEmbedder,
    InMemoryIngestStateRepository,
    InMemoryVectorStore,
)
from likes_search.application.evaluate import EvalCase, evaluate
from likes_search.application.ingest import IngestLikedVideos
from likes_search.application.library import GetLibraryStatus
from likes_search.application.search import SearchLikedVideos
from likes_search.domain.models import (
    Chunk,
    ChunkKind,
    MusicFilter,
    Transcript,
    TranscriptSegment,
    Video,
)

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


def _ingest(
    store: InMemoryVectorStore,
    state: InMemoryIngestStateRepository | None = None,
    blocked: frozenset[str] = frozenset(),
) -> IngestLikedVideos:
    return IngestLikedVideos(
        source=FakeLikedVideoSource(VIDEOS),
        transcripts=FakeTranscriptProvider(
            TRANSCRIPTS, failing=frozenset({"blocked"}), blocked=blocked
        ),
        embedder=HashingEmbedder(),
        store=store,
        state=state or InMemoryIngestStateRepository(),
    )


def test_ingest_reports_and_skips_failed_videos() -> None:
    store = InMemoryVectorStore()
    report = _ingest(store).run()

    assert report.seen == 4
    assert report.indexed == 3
    assert report.without_transcript == 1  # music
    assert report.failed == ["blocked"]
    assert store.indexed_video_ids() == {"cook", "gpu", "music"}


def test_ingest_records_failures_and_library_status() -> None:
    store, state = InMemoryVectorStore(), InMemoryIngestStateRepository()
    report = _ingest(store, state, blocked=frozenset({"gpu"})).run()

    assert report.failed == ["gpu", "blocked"]
    assert report.blocked == 1
    status = GetLibraryStatus(store, state).run()
    assert status.total_liked == 4
    assert status.indexed == 2  # cook, music
    assert status.with_transcript == 1
    assert status.metadata_only == 1
    assert status.blocked == 1
    assert status.failed_other == 1
    assert status.not_loaded == 0
    assert status.last_run_at is not None

    # A later run that gets through clears the recorded failure.
    _ingest(store, state).run()
    status = GetLibraryStatus(store, state).run()
    assert status.blocked == 0
    assert status.with_transcript == 2
    assert [f.video_id for f in state.state.failed] == ["blocked"]


def test_library_status_follows_music_filter() -> None:
    store, state = InMemoryVectorStore(), InMemoryIngestStateRepository()
    songs = [Video(video_id="s1", title="Song", channel="Band - Topic")]
    IngestLikedVideos(
        source=FakeLikedVideoSource([*VIDEOS, *songs]),
        transcripts=FakeTranscriptProvider(TRANSCRIPTS, blocked=frozenset({"gpu"})),
        embedder=HashingEmbedder(),
        store=store,
        state=state,
    ).run()
    library = GetLibraryStatus(store, state)

    everything = library.run()
    assert (everything.total_liked, everything.indexed, everything.blocked) == (5, 4, 1)

    only = library.run(MusicFilter.ONLY)
    assert (only.total_liked, only.indexed, only.metadata_only, only.blocked) == (None, 1, 1, 0)

    hidden = library.run(MusicFilter.EXCLUDE)
    assert (hidden.indexed, hidden.with_transcript, hidden.blocked) == (3, 1, 1)
    assert hidden.total_liked is None and hidden.not_loaded == 0


def test_library_status_counts_unprocessed_likes() -> None:
    store, state = InMemoryVectorStore(), InMemoryIngestStateRepository()
    _ingest(store, state).run(limit=1)
    status = GetLibraryStatus(store, state).run()
    assert (status.total_liked, status.indexed, status.not_loaded) == (4, 1, 3)


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
    assert results[0].thumbnail_url == (
        f"https://i.ytimg.com/vi/{results[0].video_id}/mqdefault.jpg"
    )


def test_search_finds_video_without_transcript_via_metadata() -> None:
    store = InMemoryVectorStore()
    _ingest(store).run()
    results = SearchLikedVideos(HashingEmbedder(), store).run("lofi beats study", k=1)
    assert results[0].video_id == "music"
    assert results[0].start_seconds is None
    assert results[0].match_kind is ChunkKind.METADATA


def test_search_explains_match_with_other_moments() -> None:
    store = InMemoryVectorStore()
    chunks = [
        Chunk(chunk_id=f"v:t{i}", video_id="v", title="Talk", channel="Ch",
              kind=ChunkKind.TRANSCRIPT, text=text, start_seconds=start)
        for i, (text, start) in enumerate(
            [("attention heads explained", 10.0), ("more attention heads", 95.0),
             ("attention again", 300.0), ("unrelated cooking", 400.0)]
        )
    ]
    embedder = HashingEmbedder()
    store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))

    [match] = SearchLikedVideos(embedder, store).run("attention heads explained", k=1)

    assert match.match_kind is ChunkKind.TRANSCRIPT
    assert match.start_seconds == 10.0
    assert match.matched_chunks == 4
    assert [o.start_seconds for o in match.other_moments] == [95.0, 300.0, 400.0]
    scores = [o.score for o in match.other_moments]
    assert scores == sorted(scores, reverse=True) and all(s <= match.score for s in scores)


def test_search_music_filter() -> None:
    store = InMemoryVectorStore()
    embedder = HashingEmbedder()
    chunks = [
        Chunk(chunk_id=f"{vid}:meta", video_id=vid, title=title, channel=channel,
              kind=ChunkKind.METADATA, text=f"{title} {channel}")
        for vid, title, channel in [
            ("song", "Rainy night piano", "Some Artist - Topic"),
            ("talk", "Rainy night photography tips", "Photo Club"),
        ]
    ]
    store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))
    search = SearchLikedVideos(embedder, store)

    assert {m.video_id for m in search.run("rainy night")} == {"song", "talk"}
    assert [m.video_id for m in search.run("rainy night", music=MusicFilter.EXCLUDE)] == ["talk"]
    only = search.run("rainy night", music=MusicFilter.ONLY)
    assert [m.video_id for m in only] == ["song"] and only[0].is_music


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
