"""Chunking, run over the parsed fixture document."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, date, datetime, timedelta

import pytest

from model_chef.ingestion.chunk.chunker import MAX_CHARS, chunk_document
from model_chef.ingestion.parse.arxiv_html import parse_arxiv_html
from model_chef.schemas import AdmittedVia, EmbeddableChunk, ParsedDocument, TriageVerdict
from tests.test_parse_arxiv_html import FIXTURE, NOW, _candidate


@pytest.fixture(scope="module")
def doc() -> ParsedDocument:
    candidate = _candidate().model_copy(update={"published_date": date(2025, 3, 26)})
    return parse_arxiv_html(FIXTURE.read_text(), candidate, parsed_at=NOW)


def _verdict(doc_id: str, via: AdmittedVia = "merit", admit: bool = True) -> TriageVerdict:
    return TriageVerdict(
        doc_id=doc_id,
        admit=admit,
        gate="centroid",
        admitted_via=via if admit else None,
        signal_score=0.6,
        reason="test",
        decided_at=NOW,
    )


def _chunk(
    doc: ParsedDocument, via: AdmittedVia = "merit", max_chars: int = MAX_CHARS
) -> list[EmbeddableChunk]:
    return chunk_document(
        doc,
        _verdict(doc.doc_id, via),
        embedding_model="BAAI/bge-m3",
        ingested_at=NOW,
        max_chars=max_chars,
    )


class TestChunking:
    def test_rechunking_yields_identical_ids(self, doc):
        """The property that makes re-ingestion an upsert, not a duplicate."""
        first = [e.chunk.chunk_id for e in _chunk(doc)]
        assert first == [e.chunk.chunk_id for e in _chunk(doc)]
        assert len(set(first)) == len(first)

    def test_prose_chunks_respect_the_size_limit(self, doc):
        prose = [e.chunk for e in _chunk(doc) if e.chunk.chunk_kind != "table"]
        assert prose
        assert all(len(c.text) <= MAX_CHARS for c in prose)

    def test_packing_loses_no_text(self, doc):
        for section in doc.sections:
            if section.section_type == "references":
                continue
            packed = " ".join(
                e.chunk.text
                for e in _chunk(doc)
                if e.chunk.heading_path == section.heading_path
                and e.chunk.chunk_kind != "table"
            )
            assert Counter(packed.split()) == Counter(section.text.split())

    def test_references_are_not_indexed(self, doc):
        assert all(e.chunk.section_type != "references" for e in _chunk(doc))


class TestTables:
    def test_each_table_is_exactly_one_chunk_whatever_its_size(self, doc):
        tiny_limit = 200
        tables = [
            e.chunk for e in _chunk(doc, max_chars=tiny_limit) if e.chunk.chunk_kind == "table"
        ]
        assert len(tables) == len(doc.tables)
        assert all(len(t.text) > tiny_limit for t in tables)

    def test_table_embeds_headers_not_bulk_cell_values(self, doc):
        big = max(
            (e for e in _chunk(doc) if e.chunk.chunk_kind == "table"),
            key=lambda e: len(e.chunk.text),
        )
        assert big.embedding_text.startswith(" > ".join(big.chunk.heading_path))
        assert len(big.embedding_text) < len(big.chunk.text) / 2


class TestProvenance:
    def test_chunks_carry_candidate_provenance(self, doc):
        chunk = _chunk(doc)[0].chunk
        assert chunk.license == "cc-by-4.0"
        assert chunk.text_depth == "full_html"
        assert chunk.venue_tier == "preprint"
        assert chunk.chunk_kind == "abstract"

    def test_provisional_admission_schedules_a_rescore(self, doc):
        chunk = _chunk(doc, via="provisional")[0].chunk
        assert chunk.provisional
        assert chunk.rescore_after == date(2025, 3, 26) + timedelta(days=90)
        assert _chunk(doc)[0].chunk.rescore_after is None

    def test_refuses_to_chunk_without_an_admitting_verdict(self, doc):
        with pytest.raises(ValueError, match="no admitting verdict"):
            chunk_document(
                doc,
                _verdict(doc.doc_id, admit=False),
                embedding_model="BAAI/bge-m3",
                ingested_at=datetime(2026, 9, 25, tzinfo=UTC),
            )
