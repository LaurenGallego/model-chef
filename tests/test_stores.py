"""Store protocol conformance.

The in-memory implementations here are the reference for adapter behaviour and
back the pipeline tests, so ingestion can be exercised without network access.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from model_chef.schemas import Chunk
from model_chef.stores import DocumentStore, FilterSpec, Vector, VectorStore
from tests.conftest import DOC_ID, build_chunk


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


class TestConformance:
    def test_in_memory_stores_satisfy_the_protocols(self):
        assert isinstance(InMemoryVectorStore(), VectorStore)
        assert isinstance(InMemoryDocumentStore(), DocumentStore)


class TestVectorStore:
    def test_upsert_is_idempotent_on_chunk_id(self):
        store = InMemoryVectorStore()
        chunk = build_chunk()
        store.upsert([(chunk, [1.0, 0.0])])
        store.upsert([(chunk, [1.0, 0.0])])
        assert len(store.vectors) == 1

    def test_query_respects_filters(self):
        store = InMemoryVectorStore()
        store.upsert([(build_chunk(doc_id="a" * 24, source_type="paper"), [1.0, 0.0])])
        store.upsert(
            [(build_chunk(doc_id="b" * 24, source_type="blog", venue_tier="blog"), [1.0, 0.0])]
        )
        hits = store.query([1.0, 0.0], limit=10, filters={"source_type": "blog"})
        assert len(hits) == 1


class TestDocumentStore:
    def test_get_omits_missing_ids_rather_than_raising(self):
        store = InMemoryDocumentStore()
        chunk = build_chunk()
        store.put([chunk])
        found = store.get([chunk.chunk_id, "missing-id"])
        assert set(found) == {chunk.chunk_id}

    def test_doc_chunk_ids_groups_by_document(self):
        """Backs whole-document eviction when a provisional admission lapses."""
        store = InMemoryDocumentStore()
        store.put([build_chunk(chunk_index=i, text=f"body {i}") for i in range(3)])
        store.put([build_chunk(doc_id="other" * 4 + "abcd", text="elsewhere")])
        assert len(store.doc_chunk_ids(DOC_ID)) == 3
