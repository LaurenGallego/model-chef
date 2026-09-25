"""Store protocol conformance, checked against the in-memory reference stores."""

from __future__ import annotations

from model_chef.stores import DocumentStore, VectorStore
from tests.conftest import DOC_ID, build_chunk
from tests.fakes import InMemoryDocumentStore, InMemoryVectorStore


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
