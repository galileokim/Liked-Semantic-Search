from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from likes_search.bootstrap import Container
from likes_search.domain.models import VideoMatch


def create_app(container: Container) -> FastAPI:
    app = FastAPI(title="Liked Videos Search")
    search = container.search()

    @app.get("/api/search")
    def search_endpoint(
        q: Annotated[str, Query(min_length=1)],
        k: Annotated[int, Query(ge=1, le=50)] = 10,
    ) -> list[VideoMatch]:
        return search.run(q, k)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML

    return app


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Liked Videos Search</title>
<style>
  body { font: 16px/1.5 system-ui, sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; }
  form { display: flex; gap: .5rem; }
  input { flex: 1; padding: .6rem .8rem; font-size: 1rem; }
  button { padding: .6rem 1rem; font-size: 1rem; }
  .r { margin: 1.25rem 0; } .r a { font-weight: 600; } .m { color: #666; font-size: .875rem; }
  .s { color: #333; font-size: .9rem; }
</style>
</head>
<body>
<h1>Search your liked videos</h1>
<form id="f"><input id="q" placeholder="that video where a guy explains transformers with a whiteboard" autofocus><button>Search</button></form>
<div id="out"></div>
<script>
const f = document.getElementById("f"), q = document.getElementById("q"), out = document.getElementById("out");
const esc = s => s.replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
f.onsubmit = async e => {
  e.preventDefault();
  if (!q.value.trim()) return;
  out.textContent = "Searching…";
  const res = await fetch("/api/search?k=10&q=" + encodeURIComponent(q.value));
  const items = await res.json();
  out.innerHTML = items.length ? items.map(m => `
    <div class="r">
      <a href="${esc(m.url)}" target="_blank" rel="noopener">${esc(m.title)}</a>
      <div class="m">${esc(m.channel)} · score ${m.score.toFixed(3)}${m.start_seconds != null ? " · jumps to " + Math.floor(m.start_seconds/60) + ":" + String(Math.floor(m.start_seconds%60)).padStart(2,"0") : ""}</div>
      <div class="s">${esc(m.snippet.slice(0, 240))}${m.snippet.length > 240 ? "…" : ""}</div>
    </div>`).join("") : "No results.";
};
</script>
</body>
</html>
"""
