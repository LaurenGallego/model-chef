from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SourceType = Literal["cookbook", "paper", "blog"]
VenueTier = Literal["peer_reviewed", "preprint", "official_docs", "blog"]
ChunkKind = Literal["prose", "table", "figure_caption", "abstract", "code"]
TextDepth = Literal["full_html", "ar5iv", "pdf_docling", "abstract_only"]
AdmittedVia = Literal["merit", "provisional", "manual"]

SectionType = Literal[
    "abstract",
    "introduction",
    "method",
    "experimental_setup",
    "results",
    "analysis",
    "related_work",
    "conclusion",
    "references",
    "appendix",
    "unknown",
]


OutputPolicy = Literal["snippet_only", "license_aware"]

#: Licences under which `license_aware` policy will serve full chunk text.
PERMISSIVE_LICENSES = frozenset(
    {
        "cc-by-4.0",
        "cc-by-sa-4.0",
        "cc0-1.0",
        "mit",
        "apache-2.0",
        "bsd-3-clause",
    }
)

MAX_SNIPPET_CHARS = 480


class Frozen(BaseModel):
    """Base for contracts that cross a module boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class QualitySignals(Frozen):
    """Triage inputs. Every field is optional: sources populate what they have."""

    citation_count: int | None = None
    citation_velocity: float | None = Field(
        default=None, description="Citations per month since publication."
    )
    influential_citations: int | None = None
    review_score_mean: float | None = Field(
        default=None, description="Mean reviewer rating, OpenReview venues only."
    )
    review_decision: str | None = None
    social_upvotes: int | None = Field(
        default=None, description="HuggingFace Daily Papers upvotes."
    )
    has_code_link: bool = False
    affiliation_hits: tuple[str, ...] = ()


class Candidate(Frozen):
    """Discovery output: metadata only, never full text."""

    external_id: str = Field(description="arXiv id, DOI, or canonical URL.")
    source_type: SourceType
    title: str
    abstract: str | None = None
    authors: tuple[str, ...] = ()
    url: str
    published_date: date | None = None
    venue: str | None = None
    venue_tier: VenueTier
    license: str | None = Field(
        default=None, description="None means assume all-rights-reserved."
    )
    signals: QualitySignals = QualitySignals()
    discovered_by: str = Field(description="Discovery module name, e.g. 'arxiv'.")
    discovered_at: datetime

    @property
    def doc_id(self) -> str:
        """Stable document identity across re-discovery by different sources."""
        return _digest(self.source_type, self.external_id)[:24]


class TriageVerdict(Frozen):
    """Why a candidate was admitted or rejected. Persisted for audit and tuning."""

    doc_id: str
    admit: bool
    gate: Literal["hard_filter", "centroid", "signal", "manual"] = Field(
        description="The gate that decided. Rejections name the gate that stopped it."
    )
    centroid_score: float | None = None
    signal_score: float | None = None
    admitted_via: AdmittedVia | None = None
    reason: str
    decided_at: datetime


class Table(Frozen):
    """A table kept intact. Never split across chunks."""

    caption: str | None
    heading_path: tuple[str, ...]
    markdown: str
    n_rows: int
    n_cols: int

    def embedding_text(self) -> str:
        """Text used to embed this table.

        Deliberately excludes the bulk of the cell values: what makes a table
        findable is its caption, position and column headers, not its numbers.
        """
        header, *rows = self.markdown.splitlines() or [""]
        parts = [
            " > ".join(self.heading_path),
            self.caption or "",
            header,
            *rows[:3],
        ]
        return "\n".join(p for p in parts if p)


class Section(Frozen):
    heading_path: tuple[str, ...]
    section_type: SectionType
    text: str
    tables: tuple[Table, ...] = ()


class ParsedDocument(Frozen):
    """A fetched and parsed document, before chunking."""

    doc_id: str
    candidate: Candidate
    text_depth: TextDepth
    sections: tuple[Section, ...]
    parsed_at: datetime

    @property
    def tables(self) -> tuple[Table, ...]:
        return tuple(t for s in self.sections for t in s.tables)


class Chunk(Frozen):
    """The unit of storage and retrieval.

    Split across two stores at write time: the vector plus the short filterable
    scalars go to the vector index, everything else to the document store.
    """

    # identity
    chunk_id: str
    doc_id: str
    chunk_index: int
    content_hash: str

    # content
    text: str
    chunk_kind: ChunkKind
    section_type: SectionType
    heading_path: tuple[str, ...] = ()

    # provenance
    title: str
    url: str
    authors: tuple[str, ...] = ()
    source_type: SourceType
    venue: str | None
    venue_tier: VenueTier
    published_date: date | None
    license: str | None
    text_depth: TextDepth

    # lifecycle
    signal_score: float
    admitted_via: AdmittedVia
    provisional: bool
    rescore_after: date | None
    embedding_model: str
    ingested_at: datetime

    @model_validator(mode="after")
    def _check_chunk_id(self) -> Self:
        expected = make_chunk_id(self.doc_id, self.chunk_index, self.content_hash)
        if self.chunk_id != expected:
            raise ValueError(
                f"chunk_id {self.chunk_id!r} is not derived from "
                f"(doc_id, chunk_index, content_hash); expected {expected!r}. "
                "Use make_chunk_id() so re-ingestion is idempotent."
            )
        return self

    @property
    def is_permissively_licensed(self) -> bool:
        return self.license is not None and self.license.lower() in PERMISSIVE_LICENSES

    def filter_payload(self) -> dict[str, str | int | bool]:
        """The subset stored on the vector, for pre-filtering."""

        return {
            "source_type": self.source_type,
            "venue_tier": self.venue_tier,
            "chunk_kind": self.chunk_kind,
            "section_type": self.section_type,
            "year": self.published_date.year if self.published_date else 0,
            "provisional": self.provisional,
        }


class EmbeddableChunk(Frozen):
    """A chunk paired with the text its vector is computed from. Only ``chunk`` is stored.

    The two differ on purpose: tables embed caption and headers rather than cell
    values, and prose embeds with its title and heading path as context.
    """

    chunk: Chunk
    embedding_text: str


class Citation(Frozen):
    """Everything needed to attribute a result."""

    doc_id: str
    title: str
    url: str
    authors: tuple[str, ...] = ()
    venue: str | None = None
    venue_tier: VenueTier
    published_date: date | None = None
    license: str | None = None


class SearchResult(Frozen):
    """The wire model returned by MCP tools.

    There is deliberately no field on this model that can carry unbounded source
    text: ``text`` is length-checked against the output policy at construction.
    """

    chunk_id: str
    text: str
    truncated: bool
    chunk_kind: ChunkKind
    section_type: SectionType
    heading_path: tuple[str, ...]
    score: float
    citation: Citation

    @model_validator(mode="after")
    def _enforce_length(self) -> Self:
        if self.truncated and len(self.text) > MAX_SNIPPET_CHARS:
            raise ValueError(
                f"truncated result carries {len(self.text)} chars, "
                f"exceeding MAX_SNIPPET_CHARS={MAX_SNIPPET_CHARS}"
            )
        return self

    @classmethod
    def from_chunk(
        cls,
        chunk: Chunk,
        score: float,
        policy: OutputPolicy = "snippet_only",
    ) -> SearchResult:
        full = policy == "license_aware" and chunk.is_permissively_licensed
        text = chunk.text if full else _snippet(chunk.text)
        return cls(
            chunk_id=chunk.chunk_id,
            text=text,
            truncated=not full,
            chunk_kind=chunk.chunk_kind,
            section_type=chunk.section_type,
            heading_path=chunk.heading_path,
            score=score,
            citation=Citation(
                doc_id=chunk.doc_id,
                title=chunk.title,
                url=chunk.url,
                authors=chunk.authors,
                venue=chunk.venue,
                venue_tier=chunk.venue_tier,
                published_date=chunk.published_date,
                license=chunk.license,
            ),
        )


def _digest(*parts: str | int) -> str:
    joined = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode()).hexdigest()


def content_hash(text: str) -> str:
    """Hash of chunk text. Drives idempotent upsert and cross-source dedupe."""
    return _digest(text.strip())[:32]


def make_chunk_id(doc_id: str, chunk_index: int, content_hash: str) -> str:
    """Deterministic chunk id, so re-running ingestion upserts rather than duplicates."""
    return _digest(doc_id, chunk_index, content_hash)[:32]


def _snippet(text: str, limit: int = MAX_SNIPPET_CHARS) -> str:
    """Truncate on a sentence boundary where one is available, else on a word."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    window = text[:limit]
    for terminator in (". ", "? ", "! "):
        cut = window.rfind(terminator)
        if cut > limit // 2:
            return window[: cut + 1].rstrip()
    cut = window.rfind(" ")
    return (window[:cut] if cut > 0 else window).rstrip() + "…"
