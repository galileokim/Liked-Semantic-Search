from datetime import UTC, datetime
from pathlib import Path

from likes_search.adapters.outbound.ingest_state import JsonIngestStateRepository
from likes_search.domain.models import FailedVideo, IngestState


def test_json_ingest_state_round_trips(tmp_path: Path) -> None:
    repo = JsonIngestStateRepository(tmp_path / "data" / "ingest_state.json")
    assert repo.load() == IngestState()

    state = IngestState(
        total_liked=12,
        last_run_at=datetime(2026, 10, 25, tzinfo=UTC),
        failed=(
            FailedVideo(video_id="v", title="T", channel="C", reason="IpBlocked", blocked=True),
        ),
    )
    repo.save(state)
    assert repo.load() == state


def test_corrupt_ingest_state_reads_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "ingest_state.json"
    path.write_text("{not json")
    assert JsonIngestStateRepository(path).load() == IngestState()
