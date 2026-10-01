"""Ranking properties. These assert orderings, not magic numbers, so the weights in
`RankingWeights` can be tuned without rewriting the tests.
"""

from __future__ import annotations

from datetime import date

from model_chef.config import RankingWeights
from model_chef.schemas import Chunk
from model_chef.server.retrieval import provenance_weight, rank, recency_factor, rrf, search
from tests.conftest import build_chunk
from tests.fakes import InMemoryDocumentStore, InMemoryVectorStore

TODAY = date(2026, 9, 25)
W = RankingWeights()


def _rank_equal_relevance(*chunks: Chunk) -> list[str]:
    fused = {c.chunk_id: 1.0 for c in chunks}
    return [c.doc_id for c, _ in rank(fused, {c.chunk_id: c for c in chunks}, TODAY, W)]


class TestFusion:
    def test_agreement_between_retrievers_outranks_a_single_hit(self):
        fused = rrf([[("both", 0.9), ("dense_only", 0.8)], [("both", 12.0)]], k=60)
        assert fused["both"] > fused["dense_only"]

    def test_fusion_ignores_score_scale(self):
        a = rrf([[("x", 0.9), ("y", 0.1)]], k=60)
        b = rrf([[("x", 900.0), ("y", 0.0001)]], k=60)
        assert a == b


class TestProvenance:
    def test_at_equal_relevance_peer_reviewed_outranks_preprint_outranks_blog(self):
        order = _rank_equal_relevance(
            build_chunk(doc_id="blog", venue_tier="blog", source_type="blog"),
            build_chunk(doc_id="peer", venue_tier="peer_reviewed"),
            build_chunk(doc_id="pre", venue_tier="preprint"),
        )
        assert order == ["peer", "pre", "blog"]

    def test_recent_outranks_old_at_equal_relevance(self):
        order = _rank_equal_relevance(
            build_chunk(doc_id="old", published_date=date(2023, 1, 1)),
            build_chunk(doc_id="new", published_date=date(2026, 8, 1)),
        )
        assert order == ["new", "old"]

    def test_recency_has_a_floor_so_strong_old_work_is_not_buried(self):
        ancient = recency_factor(date(2017, 6, 12), TODAY, W)
        assert ancient == W.recency_floor
        assert recency_factor(None, TODAY, W) == W.recency_floor

    def test_related_work_is_down_weighted_against_method(self):
        order = _rank_equal_relevance(
            build_chunk(doc_id="related", section_type="related_work"),
            build_chunk(doc_id="method", section_type="method"),
        )
        assert order == ["method", "related"]

    def test_references_are_never_returned(self):
        assert provenance_weight(build_chunk(section_type="references"), TODAY, W) == 0
        assert _rank_equal_relevance(build_chunk(section_type="references")) == []


class TestSearch:
    def _stores(self) -> tuple[InMemoryVectorStore, InMemoryDocumentStore]:
        vectors, documents = InMemoryVectorStore(), InMemoryDocumentStore()
        corpus = [
            (build_chunk(doc_id="grpo", text="GRPO drops the critic entirely."), [0.0, 1.0]),
            (build_chunk(doc_id="ppo", text="PPO clips the policy ratio."), [1.0, 0.0]),
            (build_chunk(doc_id="dpo", text="DPO needs no reward model."), [0.9, 0.1]),
        ]
        vectors.upsert(corpus)
        documents.put(c for c, _ in corpus)
        return vectors, documents

    def test_exact_token_match_surfaces_even_when_dense_misses(self):
        """The reason for hybrid search: dense retrieval is weak on tokens like `GRPO`."""
        vectors, documents = self._stores()
        query_vector = [1.0, 0.0]
        dense_top2 = {cid for cid, _ in vectors.query(query_vector, limit=2)}
        assert "grpo" not in {c.doc_id for c in documents.get(list(dense_top2)).values()}

        results = search(
            "GRPO", query_vector, vectors, documents, k=2, today=TODAY, overfetch_factor=1
        )
        assert "grpo" in {r.citation.doc_id for r in results}

    def test_returns_at_most_k_results_under_the_output_policy(self):
        vectors, documents = self._stores()
        results = search("policy", [1.0, 0.0], vectors, documents, k=2, today=TODAY)
        assert len(results) == 2
        assert all(r.truncated for r in results)
