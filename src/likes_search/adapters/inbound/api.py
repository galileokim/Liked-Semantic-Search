from __future__ import annotations

from importlib import resources
from typing import Annotated

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from likes_search.bootstrap import Container
from likes_search.domain.models import LibraryStatus, VideoMatch

_STATIC = resources.files("likes_search.adapters.inbound").joinpath("static")
_INDEX_HTML = _STATIC.joinpath("index.html").read_text()


def create_app(container: Container) -> FastAPI:
    app = FastAPI(title="Liked Videos Search")
    search = container.search()
    library_status = container.library_status()

    @app.get("/api/search")
    def search_endpoint(
        q: Annotated[str, Query(min_length=1)],
        k: Annotated[int, Query(ge=1, le=50)] = 10,
    ) -> list[VideoMatch]:
        return search.run(q, k)

    @app.get("/api/status")
    def status_endpoint() -> LibraryStatus:
        # Also lets the UI tell "nothing matched" apart from "nothing ingested yet".
        return library_status.run()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML

    return app
