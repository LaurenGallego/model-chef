"""The schema contract. These tests guard the two properties the pipeline relies on:
idempotent re-ingestion, and that full source text cannot leak through a tool response.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from model_chef.schemas import (
    MAX_SNIPPET_CHARS,
    SearchResult,
    Table,
    content_hash,
    make_chunk_id,
)
from tests.conftest import build_chunk


class TestIdentity:
    def test_chunk_id_is_deterministic(self):
        a = make_chunk_id("doc1", 3, "hash")
        b = make_chunk_id("doc1", 3, "hash")
        assert a == b

    def test_chunk_id_varies_with_every_component(self):
        base = make_chunk_id("doc1", 3, "hash")
        assert make_chunk_id("doc2", 3, "hash") != base
        assert make_chunk_id("doc1", 4, "hash") != base
        assert make_chunk_id("doc1", 3, "other") != base

    def test_reingesting_identical_text_yields_the_same_id(self):
        """The property that makes the weekly cron upsert instead of duplicate."""
        first = build_chunk(text="identical body text")
        second = build_chunk(text="identical body text")
        assert first.chunk_id == second.chunk_id

    def test_edited_source_yields_a_new_content_hash(self):
        assert content_hash("original") != content_hash("edited")

    def test_content_hash_ignores_surrounding_whitespace(self):
        assert content_hash("  body  ") == content_hash("body")

    def test_hand_written_chunk_id_is_rejected(self):
        """Guards against a source module inventing ids and breaking idempotency."""
        with pytest.raises(ValidationError, match="not derived from"):
            build_chunk(chunk_id="definitely-not-a-derived-id")


class TestOutputPolicy:
    def test_snippet_only_truncates_permissive_sources_too(self):
        """The uniform policy: licence does not matter under snippet_only."""
        chunk = build_chunk(text="Sentence one. " * 200, license="cc-by-4.0")
        result = SearchResult.from_chunk(chunk, score=0.9, policy="snippet_only")
        assert result.truncated
        assert len(result.text) <= MAX_SNIPPET_CHARS

    def test_license_aware_serves_full_text_for_permissive_sources(self):
        body = "Sentence one. " * 200
        chunk = build_chunk(text=body, license="cc-by-4.0")
        result = SearchResult.from_chunk(chunk, score=0.9, policy="license_aware")
        assert not result.truncated
        assert result.text == body

    def test_license_aware_still_truncates_unlicensed_sources(self):
        chunk = build_chunk(text="Sentence one. " * 200, license=None)
        result = SearchResult.from_chunk(chunk, score=0.9, policy="license_aware")
        assert result.truncated
        assert len(result.text) <= MAX_SNIPPET_CHARS

    def test_unknown_licence_is_treated_as_all_rights_reserved(self):
        chunk = build_chunk(text="Sentence one. " * 200, license="some-custom-eula")
        result = SearchResult.from_chunk(chunk, score=0.9, policy="license_aware")
        assert result.truncated

    def test_short_text_is_not_marked_as_truncated_content(self):
        chunk = build_chunk(text="Short body.")
        result = SearchResult.from_chunk(chunk, score=0.5, policy="snippet_only")
        assert result.text == "Short body."

    def test_truncated_result_cannot_carry_full_text(self):
        """Structural guarantee: the model refuses an over-long snippet outright."""
        payload = SearchResult.from_chunk(build_chunk(), score=0.5).model_dump()
        payload["text"] = "x" * (MAX_SNIPPET_CHARS + 1)
        with pytest.raises(ValidationError, match="MAX_SNIPPET_CHARS"):
            SearchResult.model_validate(payload)

    def test_snippet_prefers_a_sentence_boundary(self):
        chunk = build_chunk(text="word " * 80 + "ends here. " + "more " * 200)
        result = SearchResult.from_chunk(chunk, score=0.5)
        assert result.text.endswith("ends here.")

    def test_snippet_ignores_a_sentence_boundary_too_near_the_start(self):
        """Otherwise a short opening sentence would throw away most of the budget."""
        chunk = build_chunk(text="Short. " + "filler " * 200)
        result = SearchResult.from_chunk(chunk, score=0.5)
        assert len(result.text) > MAX_SNIPPET_CHARS // 2

    def test_snippet_falls_back_to_a_word_boundary(self):
        chunk = build_chunk(text="nopunctuation " * 200)
        result = SearchResult.from_chunk(chunk, score=0.5)
        assert result.text.endswith("…")
        assert not result.text.rstrip("…").endswith("nopunctuatio")

    def test_citation_survives_truncation(self):
        """A truncated result is still attributable — that is the whole policy."""
        chunk = build_chunk(text="Sentence one. " * 200)
        result = SearchResult.from_chunk(chunk, score=0.5)
        assert result.citation.url == chunk.url
        assert result.citation.title == chunk.title
        assert result.citation.venue_tier == "peer_reviewed"


class TestFilterPayload:
    def test_stays_within_vectorize_metadata_index_budget(self):
        """Vectorize allows at most 10 metadata indexes."""
        assert len(build_chunk().filter_payload()) <= 10

    def test_every_value_fits_the_64_byte_index_limit(self):
        for key, value in build_chunk().filter_payload().items():
            assert len(str(value).encode()) <= 64, key

    def test_missing_publication_date_degrades_to_year_zero(self):
        assert build_chunk(published_date=None).filter_payload()["year"] == 0


class TestTable:
    def test_embedding_text_uses_context_not_bulk_cell_values(self):
        table = Table(
            caption="GRPO ablation over KL coefficient.",
            heading_path=("Experiments", "Ablations"),
            markdown="| model | kl | score |\n|---|---|---|\n"
            + "\n".join(f"| m{i} | 0.{i} | {i} |" for i in range(50)),
            n_rows=50,
            n_cols=3,
        )
        text = table.embedding_text()
        assert "GRPO ablation" in text
        assert "Experiments > Ablations" in text
        assert "| model | kl | score |" in text
        assert "m49" not in text, "bulk rows must not dominate the embedding"

    def test_handles_an_empty_table_without_raising(self):
        table = Table(caption=None, heading_path=(), markdown="", n_rows=0, n_cols=0)
        assert table.embedding_text() == ""
