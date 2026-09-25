from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pytest

from model_chef.schemas import Chunk, content_hash, make_chunk_id

DOC_ID = "doc0123456789abcdef012345"


def build_chunk(
    text: str = "A reward model overfits when the KL penalty is too low.", **kw: Any
) -> Chunk:
    """Construct a valid Chunk with a correctly derived chunk_id."""
    doc_id = kw.pop("doc_id", DOC_ID)
    chunk_index = kw.pop("chunk_index", 0)
    digest = content_hash(text)
    defaults: dict[str, Any] = {
        "chunk_id": make_chunk_id(doc_id, chunk_index, digest),
        "doc_id": doc_id,
        "chunk_index": chunk_index,
        "content_hash": digest,
        "text": text,
        "chunk_kind": "prose",
        "section_type": "method",
        "heading_path": ("Training", "Reward modeling"),
        "title": "Scaling Laws for Reward Model Overoptimization",
        "url": "https://arxiv.org/abs/2210.10760",
        "authors": ("Gao", "Schulman", "Hilton"),
        "source_type": "paper",
        "venue": "ICML 2023",
        "venue_tier": "peer_reviewed",
        "published_date": date(2022, 10, 19),
        "license": "cc-by-4.0",
        "text_depth": "full_html",
        "signal_score": 0.82,
        "admitted_via": "merit",
        "provisional": False,
        "rescore_after": None,
        "embedding_model": "BAAI/bge-m3",
        "ingested_at": datetime(2026, 9, 18, tzinfo=UTC),
    }
    defaults.update(kw)
    return Chunk(**defaults)


@pytest.fixture
def chunk() -> Chunk:
    return build_chunk()
