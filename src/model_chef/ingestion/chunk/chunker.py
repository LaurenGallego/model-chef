"""ParsedDocument -> Chunks. Prose is packed by paragraph; tables are never split."""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import datetime, timedelta

from model_chef.schemas import (
    Chunk,
    ChunkKind,
    EmbeddableChunk,
    ParsedDocument,
    Section,
    TriageVerdict,
    content_hash,
    make_chunk_id,
)

#: ~450 tokens: long enough to hold an argument, short enough to stay on one topic.
MAX_CHARS = 2000
PROVISIONAL_RESCORE_AFTER = timedelta(days=90)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def chunk_document(
    doc: ParsedDocument,
    verdict: TriageVerdict,
    *,
    embedding_model: str,
    ingested_at: datetime,
    max_chars: int = MAX_CHARS,
) -> list[EmbeddableChunk]:
    if not verdict.admit or verdict.admitted_via is None or verdict.doc_id != doc.doc_id:
        raise ValueError(f"no admitting verdict for {doc.doc_id}")

    candidate = doc.candidate
    provisional = verdict.admitted_via == "provisional"
    rescore_from = candidate.published_date or ingested_at.date()

    pieces: list[tuple[str, str, ChunkKind, Section]] = []
    for section in doc.sections:
        if section.section_type == "references":
            continue
        kind: ChunkKind = "abstract" if section.section_type == "abstract" else "prose"
        context = "\n".join((candidate.title, " > ".join(section.heading_path)))
        for text in _pack(section.text, max_chars):
            pieces.append((text, f"{context}\n\n{text}", kind, section))
        for table in section.tables:
            text = "\n\n".join(p for p in (table.caption, table.markdown) if p)
            pieces.append((text, table.embedding_text(), "table", section))

    chunks = []
    for index, (text, embedding_text, kind, section) in enumerate(pieces):
        digest = content_hash(text)
        chunk = Chunk(
            chunk_id=make_chunk_id(doc.doc_id, index, digest),
            doc_id=doc.doc_id,
            chunk_index=index,
            content_hash=digest,
            text=text,
            chunk_kind=kind,
            section_type=section.section_type,
            heading_path=section.heading_path,
            title=candidate.title,
            url=candidate.url,
            authors=candidate.authors,
            source_type=candidate.source_type,
            venue=candidate.venue,
            venue_tier=candidate.venue_tier,
            published_date=candidate.published_date,
            license=candidate.license,
            text_depth=doc.text_depth,
            signal_score=verdict.signal_score or 0.0,
            admitted_via=verdict.admitted_via,
            provisional=provisional,
            rescore_after=rescore_from + PROVISIONAL_RESCORE_AFTER if provisional else None,
            embedding_model=embedding_model,
            ingested_at=ingested_at,
        )
        chunks.append(EmbeddableChunk(chunk=chunk, embedding_text=embedding_text))
    return chunks


def _pack(text: str, max_chars: int) -> Iterator[str]:
    """Greedily join paragraphs up to ``max_chars``, splitting oversized ones by sentence."""
    buffer = ""
    for unit in _units(text, max_chars):
        if buffer and len(buffer) + 2 + len(unit) > max_chars:
            yield buffer
            buffer = ""
        buffer = f"{buffer}\n\n{unit}" if buffer else unit
    if buffer:
        yield buffer


def _units(text: str, max_chars: int) -> Iterator[str]:
    for paragraph in filter(None, (p.strip() for p in text.split("\n\n"))):
        if len(paragraph) <= max_chars:
            yield paragraph
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            # A single sentence can exceed the limit when it holds a long equation.
            for start in range(0, len(sentence), max_chars):
                yield sentence[start : start + max_chars]
