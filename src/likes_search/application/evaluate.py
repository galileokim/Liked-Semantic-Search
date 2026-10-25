from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from likes_search.application.search import SearchLikedVideos


class EvalCase(BaseModel):
    """A query you'd realistically type, and the liked video you were looking for."""

    query: str
    expected_video_id: str


class EvalResult(BaseModel):
    k: int
    cases: int
    recall_at_k: float
    mrr: float
    misses: list[str]


def load_cases(path: Path) -> list[EvalCase]:
    """JSONL: one {"query": ..., "expected_video_id": ...} per line."""
    lines = path.read_text().splitlines()
    return [EvalCase.model_validate(json.loads(line)) for line in lines if line.strip()]


def evaluate(search: SearchLikedVideos, cases: Sequence[EvalCase], k: int = 10) -> EvalResult:
    """Recall@k: did the target show up at all? MRR: how high did it rank?"""
    hits = 0
    reciprocal_ranks = 0.0
    misses: list[str] = []
    for case in cases:
        ids = [m.video_id for m in search.run(case.query, k)]
        if case.expected_video_id in ids:
            hits += 1
            reciprocal_ranks += 1.0 / (ids.index(case.expected_video_id) + 1)
        else:
            misses.append(case.query)
    n = len(cases)
    return EvalResult(
        k=k,
        cases=n,
        recall_at_k=hits / n if n else 0.0,
        mrr=reciprocal_ranks / n if n else 0.0,
        misses=misses,
    )
