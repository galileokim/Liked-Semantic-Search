from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from likes_search.application.evaluate import evaluate, load_cases
from likes_search.bootstrap import Container
from likes_search.domain.models import MusicFilter

app = typer.Typer(help="Semantic search over your YouTube liked videos.", no_args_is_help=True)


@app.command()
def auth() -> None:
    """Run the Google OAuth flow and cache the token."""
    Container().authenticate()
    typer.echo("Authenticated.")


@app.command()
def ingest(
    limit: Annotated[int | None, typer.Option(help="Only process the N most recent likes.")] = None,
) -> None:
    """Fetch liked videos + transcripts and index them."""
    report = Container().ingest().run(limit=limit, on_progress=typer.echo)
    typer.echo(
        f"\nseen={report.seen} indexed={report.indexed} "
        f"skipped_existing={report.skipped_existing} "
        f"no_transcript={report.without_transcript} failed={len(report.failed)} "
        f"(blocked={report.blocked})"
    )
    if report.failed:
        typer.echo("Failed videos will be retried on the next run.")


@app.command()
def search(
    query: str,
    k: Annotated[int, typer.Option("-k", help="Number of videos to return.")] = 10,
    music: Annotated[
        MusicFilter, typer.Option(help="Include, exclude, or show only music.")
    ] = MusicFilter.INCLUDE,
) -> None:
    """Search your liked videos by meaning."""
    for i, m in enumerate(Container().search().run(query, k, music), start=1):
        snippet = m.snippet.replace("\n", " ")
        typer.echo(f"{i:>2}. [{m.score:.3f}] {m.title} — {m.channel}")
        typer.echo(f"    {m.url}")
        typer.echo(f"    {snippet[:160]}{'…' if len(snippet) > 160 else ''}\n")


@app.command(name="eval")
def eval_cmd(
    cases_file: Path,
    k: Annotated[int, typer.Option("-k")] = 10,
) -> None:
    """Measure retrieval quality against a JSONL file of {query, expected_video_id}."""
    result = evaluate(Container().search(), load_cases(cases_file), k)
    typer.echo(f"cases={result.cases} recall@{k}={result.recall_at_k:.3f} mrr={result.mrr:.3f}")
    for q in result.misses:
        typer.echo(f"  miss: {q}")


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: Annotated[
        bool, typer.Option(help="Restart on Python changes; serve HTML edits on refresh.")
    ] = False,
) -> None:
    """Start the web UI + JSON API."""
    import uvicorn

    if reload:
        import likes_search

        uvicorn.run(
            "likes_search.adapters.inbound.api:create_dev_app",
            factory=True,
            host=host,
            port=port,
            reload=True,
            reload_dirs=[str(Path(likes_search.__file__).parent)],
            log_level="debug",
        )
        return

    from likes_search.adapters.inbound.api import create_app

    uvicorn.run(create_app(Container()), host=host, port=port)


if __name__ == "__main__":
    app()
