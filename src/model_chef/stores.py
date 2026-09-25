from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from model_chef.schemas import Chunk

type Vector = Sequence[float]
type FilterSpec = dict[str, str | int | bool]


@runtime_checkable
class VectorStore(Protocol):
    """Dense ANN index. Holds vectors plus the short filterable scalars only."""

    def upsert(self, chunks: Iterable[tuple[Chunk, Vector]]) -> int:
        """Insert or replace. Idempotent on ``chunk_id``. Returns count written."""
        ...

    def query(
        self,
        vector: Vector,
        limit: int,
        filters: FilterSpec | None = None,
    ) -> list[tuple[str, float]]:
        """Return ``(chunk_id, similarity)`` ordered by descending similarity."""
        ...

    def delete(self, chunk_ids: Iterable[str]) -> int:
        """Remove chunks, e.g. when a provisional document is evicted."""
        ...


@runtime_checkable
class DocumentStore(Protocol):
    """Chunk text, full payload, and the lexical (BM25) index."""

    def put(self, chunks: Iterable[Chunk]) -> int:
        """Insert or replace by ``chunk_id``. Returns count written."""
        ...

    def get(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        """Batch fetch. Missing ids are absent from the result rather than an error."""
        ...

    def search_lexical(
        self,
        query: str,
        limit: int,
        filters: FilterSpec | None = None,
    ) -> list[tuple[str, float]]:
        """BM25 over chunk text. Returns ``(chunk_id, score)`` descending."""
        ...

    def delete(self, chunk_ids: Iterable[str]) -> int: ...

    def doc_chunk_ids(self, doc_id: str) -> list[str]:
        """All chunk ids for a document, for whole-document eviction."""
        ...
