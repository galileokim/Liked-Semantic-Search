from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from likes_search.domain.models import Video
from likes_search.ports import LikedVideoSource

SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]
_PAGE_SIZE = 50  # API maximum


def load_credentials(client_secrets_file: Path, token_file: Path) -> Credentials:
    """Cached token -> refresh -> browser consent, in that order.

    With the OAuth consent screen in "Testing" mode (you as the only test user) no Google
    verification is needed, but refresh tokens expire after 7 days, so expect to
    re-consent weekly.
    """
    creds: Credentials | None = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)  # type: ignore[no-untyped-call]

    if creds is not None and not creds.valid and creds.refresh_token:
        try:
            creds.refresh(Request())  # type: ignore[no-untyped-call]
        except RefreshError:
            creds = None

    if creds is None or not creds.valid:
        if not client_secrets_file.exists():
            raise FileNotFoundError(
                f"OAuth client secrets not found at {client_secrets_file}. See README setup."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets_file), SCOPES)
        creds = flow.run_local_server(port=0)

    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(creds.to_json())
    return creds


class YouTubeApiLikedVideoSource(LikedVideoSource):
    def __init__(self, credentials: Credentials) -> None:
        self._youtube = build("youtube", "v3", credentials=credentials, cache_discovery=False)

    def iter_liked_videos(self, limit: int | None = None) -> Iterator[Video]:
        yielded = 0
        page_token: str | None = None
        while True:
            response: dict[str, Any] = (
                self._youtube.videos()
                .list(part="snippet", myRating="like", maxResults=_PAGE_SIZE, pageToken=page_token)
                .execute()
            )
            for item in response.get("items", []):
                yield _to_video(item)
                yielded += 1
                if limit is not None and yielded >= limit:
                    return
            page_token = response.get("nextPageToken")
            if not page_token:
                return


def _to_video(item: dict[str, Any]) -> Video:
    snippet: dict[str, Any] = item["snippet"]
    return Video(
        video_id=item["id"],
        title=snippet.get("title", ""),
        channel=snippet.get("channelTitle", ""),
        description=snippet.get("description", ""),
        tags=tuple(snippet.get("tags", [])),
        published_at=snippet.get("publishedAt"),
    )
