from __future__ import annotations

import time
from collections.abc import Sequence

from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from likes_search.domain.models import Transcript, TranscriptSegment
from likes_search.ports import TranscriptFetchError, TranscriptProvider


class YouTubeTranscriptApiProvider(TranscriptProvider):
    """Uses youtube-transcript-api (scrapes the player's caption track; not the official
    Data API, which only serves captions for videos you own).

    YouTube throttles/blocks IPs that fetch too fast, hence the delay between requests.
    """

    def __init__(self, languages: Sequence[str] = ("en",), delay_seconds: float = 1.0) -> None:
        self._api = YouTubeTranscriptApi()
        self._languages = list(languages)
        self._delay_seconds = delay_seconds

    def fetch(self, video_id: str) -> Transcript | None:
        time.sleep(self._delay_seconds)
        try:
            fetched = self._api.fetch(video_id, languages=self._languages)
        except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
            return None
        except CouldNotRetrieveTranscript as e:
            raise TranscriptFetchError(type(e).__name__) from e

        return Transcript(
            video_id=video_id,
            language=fetched.language_code,
            segments=tuple(
                TranscriptSegment(text=s.text, start=s.start, duration=s.duration)
                for s in fetched.snippets
            ),
        )
