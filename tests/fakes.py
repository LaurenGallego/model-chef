"""In-memory stores: the reference for adapter behaviour, and the backing for pipeline
and retrieval tests so neither needs network access.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from model_chef.schemas import Chunk
from model_chef.stores import FilterSpec, Vector


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.vectors: dict[str, tuple[Chunk, Vector]] = {}

    def upsert(self, chunks: Iterable[tuple[Chunk, Vector]]) -> int:
        items = list(chunks)
        for chunk, vec in items:
            self.vectors[chunk.chunk_id] = (chunk, vec)
        return len(items)

    def query(
        self, vector: Vector, limit: int, filters: FilterSpec | None = None
    ) -> list[tuple[str, float]]:
        hits = []
        for chunk_id, (chunk, vec) in self.vectors.items():
            if filters and any(chunk.filter_payload().get(k) != v for k, v in filters.items()):
                continue
            hits.append((chunk_id, _cosine(vector, vec)))
        return sorted(hits, key=lambda h: -h[1])[:limit]

    def delete(self, chunk_ids: Iterable[str]) -> int:
        return sum(self.vectors.pop(cid, None) is not None for cid in chunk_ids)


class InMemoryDocumentStore:
    def __init__(self) -> None:
        self.chunks: dict[str, Chunk] = {}

    def put(self, chunks: Iterable[Chunk]) -> int:
        items = list(chunks)
        for chunk in items:
            self.chunks[chunk.chunk_id] = chunk
        return len(items)

    def get(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        return {cid: self.chunks[cid] for cid in chunk_ids if cid in self.chunks}

    def search_lexical(
        self, query: str, limit: int, filters: FilterSpec | None = None
    ) -> list[tuple[str, float]]:
        terms = set(query.lower().split())
        hits = []
        for chunk_id, chunk in self.chunks.items():
            if filters and any(chunk.filter_payload().get(k) != v for k, v in filters.items()):
                continue
            overlap = len(terms & set(chunk.text.lower().split()))
            if overlap:
                hits.append((chunk_id, float(overlap)))
        return sorted(hits, key=lambda h: -h[1])[:limit]

    def delete(self, chunk_ids: Iterable[str]) -> int:
        return sum(self.chunks.pop(cid, None) is not None for cid in chunk_ids)

    def doc_chunk_ids(self, doc_id: str) -> list[str]:
        return [cid for cid, c in self.chunks.items() if c.doc_id == doc_id]


def _cosine(a: Vector, b: Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0
