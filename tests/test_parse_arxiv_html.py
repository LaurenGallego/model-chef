"""arXiv HTML parsing, against a recorded LaTeXML page (CC BY 4.0, see fixture header)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from model_chef.ingestion.parse.arxiv_html import parse_arxiv_html
from model_chef.schemas import Candidate, ParsedDocument, Section

FIXTURE = Path(__file__).parent / "fixtures" / "arxiv_html" / "2503.20783v2.html"
NOW = datetime(2026, 9, 25, tzinfo=UTC)


def _candidate() -> Candidate:
    return Candidate(
        external_id="2503.20783",
        source_type="paper",
        title="Understanding R1-Zero-Like Training: A Critical Perspective",
        url="https://arxiv.org/abs/2503.20783",
        venue_tier="preprint",
        license="cc-by-4.0",
        discovered_by="arxiv",
        discovered_at=NOW,
    )


@pytest.fixture(scope="module")
def doc() -> ParsedDocument:
    return parse_arxiv_html(FIXTURE.read_text(), _candidate(), parsed_at=NOW)


def _section(doc: ParsedDocument, title: str) -> Section:
    return next(s for s in doc.sections if s.heading_path[-1] == title)


class TestSections:
    def test_abstract_comes_first(self, doc):
        assert doc.sections[0].section_type == "abstract"
        assert doc.sections[0].text.startswith("DeepSeek-R1-Zero has shown")

    def test_section_types_derive_from_headings(self, doc):
        assert _section(doc, "Introduction").section_type == "introduction"
        assert _section(doc, "Closing Remarks").section_type == "conclusion"
        assert _section(doc, "References").section_type == "references"
        assert _section(doc, "Policy Gradient Derivations").section_type == "appendix"

    def test_subsections_nest_and_drop_number_tags(self, doc):
        grpo = _section(doc, "GRPO Leads to Biased Optimization")
        assert grpo.heading_path == (
            "Analysis on Reinforcement Learning",
            grpo.heading_path[-1],
        )

    def test_subsections_inherit_type_when_their_heading_is_uninformative(self, doc):
        parent = _section(doc, "Analysis on Reinforcement Learning")
        child = _section(doc, "Dr. GRPO: Group Relative Policy Optimization Done Right")
        assert child.section_type == parent.section_type

    def test_maths_keeps_its_tex_source(self, doc):
        """Exact tokens like `\\mathcal{J}_{GRPO}` must survive for lexical search."""
        grpo = _section(doc, "GRPO Leads to Biased Optimization")
        assert r"\mathcal{J}_{GRPO}" in grpo.text


class TestTables:
    def test_tables_are_extracted_with_caption_and_structure(self, doc):
        assert len(doc.tables) == 5
        table = doc.tables[0]
        assert table.caption is not None and table.caption.startswith("Table 1:")
        assert table.n_cols == 7
        assert table.markdown.splitlines()[0].startswith("| Base model + Template | AIME24")

    def test_table_cells_do_not_leak_into_section_prose(self, doc):
        section = _section(
            doc, "Qwen-2.5 Models Unlock the Best Performance When Discarding Template"
        )
        assert section.tables
        assert "(4-shot prompting)" not in section.text

    def test_span_tabular_from_resizebox_is_parsed(self):
        html = """<article><section class="ltx_section"><h2 class="ltx_title">
          <span class="ltx_tag">4</span>Results</h2>
          <figure class="ltx_table"><figcaption>Table 2: KL sweep.</figcaption>
            <span class="ltx_transformed_inner"><span class="ltx_tabular">
              <span class="ltx_tr">
                <span class="ltx_td">beta</span><span class="ltx_td">Acc</span></span>
              <span class="ltx_tr">
                <span class="ltx_td">0.1</span><span class="ltx_td">51.2</span></span>
            </span></span>
          </figure></section></article>"""
        doc = parse_arxiv_html(html, _candidate(), parsed_at=NOW)
        (table,) = doc.tables
        assert table.heading_path == ("Results",)
        assert table.markdown.splitlines() == [
            "| beta | Acc |",
            "| --- | --- |",
            "| 0.1 | 51.2 |",
        ]
