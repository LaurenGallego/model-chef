"""Parse LaTeXML-generated HTML (arXiv HTML and ar5iv) into a ParsedDocument.

LaTeXML gives real structure: sections are nested ``<section>`` elements, tables are
``ltx_tabular`` grids, and maths carries its TeX source in ``alttext``.
"""

from __future__ import annotations

from datetime import datetime

import structlog
from selectolax.parser import HTMLParser, Node

from model_chef.schemas import Candidate, ParsedDocument, Section, SectionType, Table, TextDepth

log = structlog.get_logger(__name__)

_SECTION_CLASSES = ("ltx_section", "ltx_appendix", "ltx_bibliography")
_SUBSECTION_CLASSES = ("ltx_subsection", "ltx_subsubsection", "ltx_paragraph")

#: First match wins, so more specific phrases precede the words they contain.
_SECTION_KEYWORDS: tuple[tuple[SectionType, tuple[str, ...]], ...] = (
    ("related_work", ("related work", "prior work", "background")),
    ("introduction", ("introduction",)),
    ("experimental_setup", ("setup", "setting", "implementation detail", "hyperparameter")),
    ("analysis", ("analysis", "ablation", "discussion")),
    ("results", ("result", "experiment", "evaluation", "benchmark")),
    ("method", ("method", "approach", "algorithm", "preliminar", "formulation")),
    ("conclusion", ("conclusion", "closing remark", "limitation", "future work")),
)


def parse_arxiv_html(
    html: str,
    candidate: Candidate,
    *,
    parsed_at: datetime,
    text_depth: TextDepth = "full_html",
) -> ParsedDocument:
    tree = HTMLParser(html)
    root = tree.css_first("article") or tree.body
    if root is None:
        raise ValueError(f"no document body in HTML for {candidate.external_id}")

    for math in root.css("math"):
        math.replace_with(f"${math.attributes.get('alttext') or ''}$")

    sections: list[Section] = []
    abstract = root.css_first("div.ltx_abstract")
    if abstract is not None:
        _drop_titles(abstract)
        sections.append(
            Section(heading_path=("Abstract",), section_type="abstract", text=_text(abstract))
        )

    for node in root.css(", ".join(f"section.{c}" for c in _SECTION_CLASSES)):
        sections.extend(_walk(node, (), _section_type_of(node)))

    log.info(
        "parsed_html",
        doc_id=candidate.doc_id,
        sections=len(sections),
        tables=sum(len(s.tables) for s in sections),
    )
    return ParsedDocument(
        doc_id=candidate.doc_id,
        candidate=candidate,
        text_depth=text_depth,
        sections=tuple(sections),
        parsed_at=parsed_at,
    )


def _walk(node: Node, parent_path: tuple[str, ...], inherited: SectionType) -> list[Section]:
    """One Section per (sub)section node, holding only its own direct content."""
    path = (*parent_path, _title(node))
    section_type = inherited
    if parent_path and inherited not in ("appendix", "references"):
        section_type = _classify(path[-1]) or inherited

    paragraphs: list[str] = []
    tables: list[Table] = []
    children: list[Section] = []
    for child in node.iter():
        if _has_class(child, _SUBSECTION_CLASSES):
            children.extend(_walk(child, path, section_type))
        elif _has_class(child, ("ltx_table",)):
            tables.extend(_tables(child, path))
        elif _has_class(child, ("ltx_figure",)):
            if (caption := child.css_first("figcaption")) is not None:
                paragraphs.append(_text(caption))
        elif not _has_class(child, ("ltx_title",)):
            for nested in child.css("figure.ltx_table"):
                tables.extend(_tables(nested, path))
                nested.decompose()
            if text := _text(child):
                paragraphs.append(text)

    own = Section(
        heading_path=path,
        section_type=section_type,
        text="\n\n".join(paragraphs),
        tables=tuple(tables),
    )
    return [own, *children] if own.text or own.tables else children


def _tables(figure: Node, heading_path: tuple[str, ...]) -> list[Table]:
    caption_node = figure.css_first("figcaption")
    caption = _text(caption_node) if caption_node is not None else None
    tables = []
    # Tables inside \resizebox arrive as <span class="ltx_tabular">, not <table>.
    for grid in figure.css(".ltx_tabular"):
        rows = [
            [_text(cell).replace("|", r"\|") for cell in row.css(".ltx_td")]
            for row in grid.css(".ltx_tr")
        ]
        rows = [r for r in rows if any(r)]
        if not rows:
            continue
        n_cols = max(len(r) for r in rows)
        rows = [r + [""] * (n_cols - len(r)) for r in rows]
        lines = [_md_row(rows[0]), _md_row(["---"] * n_cols), *(_md_row(r) for r in rows[1:])]
        tables.append(
            Table(
                caption=caption,
                heading_path=heading_path,
                markdown="\n".join(lines),
                n_rows=len(rows),
                n_cols=n_cols,
            )
        )
    return tables


def _section_type_of(node: Node) -> SectionType:
    if _has_class(node, ("ltx_bibliography",)):
        return "references"
    if _has_class(node, ("ltx_appendix",)):
        return "appendix"
    return _classify(_title(node)) or "unknown"


def _classify(title: str) -> SectionType | None:
    lowered = title.lower()
    for section_type, keywords in _SECTION_KEYWORDS:
        if any(k in lowered for k in keywords):
            return section_type
    return None


def _title(node: Node) -> str:
    """Heading text without its number tag ('3.2', 'Appendix A')."""
    heading = next((c for c in node.iter() if _has_class(c, ("ltx_title",))), None)
    if heading is None:
        return ""
    tag = heading.css_first(".ltx_tag")
    full = _text(heading)
    if tag is not None:
        full = full.removeprefix(_text(tag)).strip()
    return full


def _drop_titles(node: Node) -> None:
    for title in node.css(".ltx_title"):
        title.decompose()


def _has_class(node: Node, classes: tuple[str, ...]) -> bool:
    return any(c in classes for c in (node.attributes.get("class") or "").split())


def _text(node: Node) -> str:
    return " ".join(node.text(separator=" ").split())


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"
