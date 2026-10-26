from __future__ import annotations

from importlib import resources
from typing import Annotated

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from likes_search.bootstrap import Container
from likes_search.domain.models import LibraryStatus, MusicFilter, VideoMatch

_STATIC = resources.files("likes_search.adapters.inbound").joinpath("static")
_INDEX_HTML = _STATIC.joinpath("index.html").read_text()


def create_app(container: Container, live_html: bool = False) -> FastAPI:
    """`live_html` re-reads the page on every request, so edits show on refresh."""
    app = FastAPI(title="Liked Videos Search")
    search = container.search()
    library_status = container.library_status()

    @app.get("/api/search")
    def search_endpoint(
        q: Annotated[str, Query(min_length=1)],
        k: Annotated[int, Query(ge=1, le=50)] = 10,
        music: MusicFilter = MusicFilter.INCLUDE,
    ) -> list[VideoMatch]:
        return search.run(q, k, music)

    @app.get("/api/status")
    def status_endpoint(music: MusicFilter = MusicFilter.INCLUDE) -> LibraryStatus:
        # Also lets the UI tell "nothing matched" apart from "nothing ingested yet".
        return library_status.run(music)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _STATIC.joinpath("index.html").read_text() if live_html else _INDEX_HTML

    return app


def create_dev_app() -> FastAPI:
    """Zero-argument factory for `likes serve --reload`: uvicorn's reloader re-imports
    the app by name in a fresh process on each Python change."""
    return create_app(Container(), live_html=True)
