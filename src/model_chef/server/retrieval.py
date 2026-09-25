"""Hybrid retrieval: dense + BM25 fused with RRF, then weighted by provenance.

This is the reference implementation of ranking; the Worker API mirrors it.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import date

from model_chef.config import RankingWeights
from model_chef.schemas import Chunk, OutputPolicy, SearchResult
from model_chef.stores import DocumentStore, FilterSpec, Vector, VectorStore

type Ranking = Sequence[tuple[str, float]]


def rrf(rankings: Iterable[Ranking], k: int) -> dict[str, float]:
    """Reciprocal rank fusion. Uses rank only, so BM25 and cosine scales never mix."""
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (chunk_id, _) in enumerate(ranking, start=1):
            fused[chunk_id] += 1.0 / (k + rank)
    return dict(fused)


def recency_factor(published: date | None, today: date, weights: RankingWeights) -> float:
    if published is None:
        return weights.recency_floor
    age_days = max(0, (today - published).days)
    return max(weights.recency_floor, math.exp(-age_days / weights.recency_tau_days))


def provenance_weight(chunk: Chunk, today: date, weights: RankingWeights) -> float:
    """Tier * recency * section weight. Zero means the chunk is never returned."""
    if chunk.section_type in weights.dropped_sections:
        return 0.0
    return (
        weights.tier[chunk.venue_tier]
        * recency_factor(chunk.published_date, today, weights)
        * weights.section.get(chunk.section_type, 1.0)
    )


def rank(
    fused: dict[str, float],
    chunks: dict[str, Chunk],
    today: date,
    weights: RankingWeights,
) -> list[tuple[Chunk, float]]:
    scored = [
        (chunk, fused[chunk_id] * w)
        for chunk_id, chunk in chunks.items()
        if (w := provenance_weight(chunk, today, weights)) > 0
    ]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)


def search(
    query: str,
    query_vector: Vector,
    vectors: VectorStore,
    documents: DocumentStore,
    *,
    k: int,
    today: date,
    filters: FilterSpec | None = None,
    policy: OutputPolicy = "snippet_only",
    overfetch_factor: int = 3,
    weights: RankingWeights = RankingWeights(),  # noqa: B008 - frozen, safe to share
) -> list[SearchResult]:
    limit = k * overfetch_factor
    fused = rrf(
        [
            vectors.query(query_vector, limit, filters),
            documents.search_lexical(query, limit, filters),
        ],
        weights.rrf_k,
    )
    ranked = rank(fused, documents.get(list(fused)), today, weights)
    return [SearchResult.from_chunk(chunk, score, policy) for chunk, score in ranked[:k]]
