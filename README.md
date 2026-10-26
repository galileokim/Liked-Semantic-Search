# likes-search

Semantic search over your YouTube liked videos — find "that video where someone explains
X" when you don't remember the title. Results deep-link to the moment in the video where
the matching transcript passage starts.

<table>
  <tr>
    <td width="50%"><img alt="Liked Videos Search" src="https://github.com/user-attachments/assets/a81e4af0-dde4-4e01-bf7b-4e3c1e89687e" /></td>
    <td width="50%"><img alt="Screenshot 2" src="https://github.com/user-attachments/assets/0562d6b3-3f7f-4263-8b6c-7dd6c333e31a" /></td>
  </tr>
  <tr>
    <td width="50%"><img alt="Screenshot 3" src="https://github.com/user-attachments/assets/38eacf83-43f3-4604-904b-8a07fe28eed9" /></td>
    <td width="50%"><img alt="Screenshot 4" src="https://github.com/user-attachments/assets/cfc73052-28e4-4836-8c78-d1d8404262d9" /></td>
  </tr>
</table>

## How it works

```
liked videos ──► transcripts ──► chunk ──► embed ──► Chroma
 (YouTube API)   (youtube-transcript-api)  (BGE, local)   (local, persistent)

query ──► embed (with query prefix) ──► top chunks ──► group by video (max score) ──► results
```

- **Chunks:** every video gets one *metadata* chunk (title, channel, tags, description) so
  it's searchable even without a transcript, plus ~200-word overlapping *transcript*
  windows, each remembering its start timestamp.
- **Embeddings:** `BAAI/bge-small-en-v1.5` via sentence-transformers, run locally. BGE is
  asymmetric, so queries get an instruction prefix and documents don't — which is why the
  `Embedder` port has separate `embed_query`/`embed_documents`.
- **Ingest is idempotent:** already-indexed videos are skipped; videos whose transcript
  fetch failed (rate limit / IP block) are *not* indexed, so the next run retries them.

## Setup

### 1. Google OAuth client (one-time, ~5 min, no Google review needed)

Liked videos are private, so an API key isn't enough — you need OAuth. Keeping the app
in **Testing** mode with yourself as the only test user avoids Google's verification.

1. [Google Cloud Console](https://console.cloud.google.com/) → create a project.
2. *APIs & Services → Library* → enable **YouTube Data API v3**.
3. *OAuth consent screen* → User type **External** → fill the required fields → add
   your Google account under **Test users**. Leave publishing status as **Testing**.
4. *Credentials → Create credentials → OAuth client ID* → Application type
   **Desktop app** → download the JSON and save it as `client_secret.json` in the
   project root (it's gitignored).

> In Testing mode refresh tokens expire after 7 days — `likes auth` re-opens the browser
> consent when that happens.

### 2. Install

Requires Python 3.12–3.14. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Usage

```bash
uv run likes auth                      # browser consent, caches token in .data/
uv run likes ingest --limit 50         # start small; re-run anytime to add new likes
uv run likes search "guy explains attention with a whiteboard"
uv run likes serve                     # web UI at http://127.0.0.1:8000
uv run likes serve --reload            # dev: restarts on Python changes, HTML edits show on refresh
```

### Evaluating retrieval quality

Write a few realistic queries for videos you know are in your likes:

```jsonl
{"query": "neural network recognizing handwritten digits", "expected_video_id": "aircAruvnKk"}
```

```bash
uv run likes eval eval_cases.jsonl -k 10   # prints recall@k, MRR, and missed queries
```

Use this to compare changes (embedding model, chunk size, hybrid search, ...) with numbers
instead of vibes.

## Development

```bash
uv run pytest
uv run mypy src tests      # strict mode
uv run ruff check .
```

## Configuration

All settings in `config.py` can be overridden via env vars or `.env`, e.g.:

```bash
LIKES_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
LIKES_TRANSCRIPT_LANGUAGES='["en","es"]'
LIKES_TRANSCRIPT_DELAY_SECONDS=2
```

Changing the embedding model requires re-indexing (delete `.data/chroma`).

## Known limitations

- `youtube-transcript-api` is unofficial; YouTube may throttle or block your IP if you
  ingest thousands of videos quickly. Keep the delay, ingest in batches.
- Only transcripts in `LIKES_TRANSCRIPT_LANGUAGES` are fetched; others fall back to
  metadata-only.
- A single long video can crowd out candidates for a query (results are grouped from the
  top `k × 5` chunks).
