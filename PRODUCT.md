# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **The owner, as daily user.** One person searching their own YouTube liked videos on their own laptop. They either remember a video only vaguely ("that one where someone explains X on a whiteboard") or want to see everything they've liked on a topic.
- **Interviewers and recruiters, as evaluators.** The project is also a portfolio piece. Evaluators see it demoed or read its code, so the interface and the engineering both get judged.

## Product Purpose

Search your own liked videos by meaning instead of by title. It pulls your likes and their transcripts, then finds matches even when the query shares no words with the video. Each result opens the video at the moment the matching passage starts.

Success means the half-remembered video shows up near the top and opens at the right spot, and that topic searches turn up the relevant videos you'd forgotten about. Retrieval quality is measured with recall@k and MRR (`likes eval`), not judged by feel.

## Positioning

Searches only your own likes, inside the transcripts and not just the titles, and runs entirely on your machine. YouTube's own search covers all of YouTube, matches titles and metadata, and has no deep links into a passage.

## Operating Context

- Runs locally. `uv run likes serve` serves the web UI at `http://127.0.0.1:8000`; there is also a CLI (`likes auth`, `likes ingest`, `likes search`, `likes eval`).
- Ingest runs every so often in batches; search is the interactive, everyday surface.
- Google OAuth runs in Testing mode, so the owner re-authorizes about every 7 days.

## Capabilities and Constraints

- Two equally important jobs: finding one specific video, and exploring a topic across your likes.
- Results show one entry per video: title, channel, similarity score, a transcript or metadata snippet, a thumbnail, and a deep link to the timestamp (when the match came from the transcript).
- Videos with no transcript can still be found by title, channel, tags and description.
- **Local only, permanently:** the embedding model (BGE), the Chroma index and OAuth tokens stay on the owner's machine. No hosting or multi-user plans.
- Transcripts come from the unofficial `youtube-transcript-api`, which YouTube may throttle. Only the configured languages are fetched (English by default).
- Terms: "liked videos", "ingest", "chunk" (metadata chunk vs transcript chunk), "score" (cosine similarity).

## Evidence on Hand

- A real personal index in `.data/` (gitignored, private). Don't present its contents as public data.
- No users, testimonials, benchmarks or published eval results exist. Don't invent any.

## Product Principles

1. **Find it, then land on the moment.** A result is only useful if it opens where the match is.
2. **Serve both jobs.** Pinpoint recall and topic exploration matter equally; neither gets buried.
3. **Private by construction.** Nothing leaves the machine, and nothing in the product suggests otherwise.
4. **Honest about relevance.** Show why a result matched (snippet, score, timestamp) instead of hiding it.
5. **Demo-worthy craft.** It's a portfolio piece, so the experience and the code should both hold up when someone looks closely.
