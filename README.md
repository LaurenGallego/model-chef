# model-chef

A curated, provenance-tagged retrieval index over **LLM post-training knowledge** —
RLHF, GRPO, DPO, PPO, reward modelling — exposed as an **MCP server** so any harness
(Claude Code, Claude Desktop, Cursor) can query it as a tool.

> **Status: phase 1 of 7.** Packaging, contracts and CI are in place. Ingestion and the
> server are not built yet. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Why the scope is narrow — on purpose

This covers **post-training only**. Not pretraining, not SFT, not ML broadly.

That is a deliberate design decision, not an unfinished roadmap. A retrieval index is
only as useful as its curation, and curation does not generalise: the signals that
identify a good RLHF paper (reviewer scores, citation velocity within a fast-moving
subfield, whether the recipe is reproducible) are not the signals that identify a good
pretraining paper. A narrow index with a defensible admission policy beats a broad one
with none.

## What makes it more than a citation search

- **Full text, including tables.** Papers are parsed from LaTeX-derived arXiv HTML where
  available, so result tables survive with their structure intact. Tables are indexed as
  atomic chunks with their caption and section context.
- **Curated admission, with the decision log in git.** Candidates pass a cascade of cheap
  gates; the ambiguous band is reviewed by a human weekly and the verdicts are committed.
- **The index is mutable.** Recent work is admitted provisionally on social signal, then
  re-scored at 90 days and evicted if it did not pan out.
- **Hybrid retrieval.** Dense embeddings plus BM25, fused — because dense retrieval alone
  fails on exact technical tokens like `GRPO`, `Qwen3-32B` or `β=0.1`.

## Architecture

```
DISCOVER  ~2,000/wk   metadata only, many APIs       arXiv · OpenReview · OpenAlex
    ↓                                                 S2 · HF Papers · blog RSS
TRIAGE    → ~150      gate cascade + human review
    ↓
PARSE                 arXiv HTML → ar5iv → Docling (PDF)
    ↓
CHUNK → EMBED → UPSERT

                      Vectorize (vectors)  ·  D1 (text + BM25)  ·  R2 (archive)
                                        ↑
harness → MCP (this package) → HTTPS → Worker API (auth, embed, search, RRF)
```

The MCP server **never calls an LLM provider**. It returns retrieved, cited chunks;
generation happens in whatever harness you are running. That is what makes it work with
Claude, GPT, or a local model without any per-provider code.

## Install

Not published yet. For development:

```bash
git clone https://github.com/LaurenGallego/model-chef
cd model-chef
uv sync --group dev
uv run pytest
```

## Configuration

Copy `.env.example` to `.env`. Required to use the server once it exists:

| Variable | Purpose |
|---|---|
| `MODEL_CHEF_API_URL` | Endpoint of the hosted index |
| `MODEL_CHEF_API_KEY` | Key for the model-chef API — **not** an LLM provider key |
| `MODEL_CHEF_OUTPUT_POLICY` | `snippet_only` (default) or `license_aware` |

Ingestion variables are maintainer-only and not needed to use the package.

## How the ingest works

Discovery runs daily across the source APIs and costs almost nothing, because it pulls
metadata only. Triage narrows roughly 2,000 weekly candidates to ~150 using hard filters,
topical similarity against hand-written centroids, and free quality signals (OpenReview
reviewer scores, citation velocity, social signal). Only survivors get fetched and
parsed. A venue calendar triggers bulk backfills when ICLR/NeurIPS/ICML decisions become
public.

## Known limitations

- **Freshness is bounded by the ingest cadence** — daily discovery, weekly deep pass.
- **Coverage is tiered.** Papers before Dec 2023 often have no arXiv HTML and fall back
  to ar5iv or PDF parsing; some are abstract-only. Every chunk records its `text_depth`
  and the breakdown is reported, so shallow coverage is visible rather than hidden.
- **Blog coverage is a hand-curated allowlist**, never open crawling. It is small by
  construction.
- **Pretraining and SFT are out of scope.** See above — this is deliberate.
- **Single-vendor storage.** The index lives on Cloudflare. Ingestion writes through
  provider-agnostic protocols and the corpus is rebuildable from source, so the exit is
  a new adapter plus a reindex — but it is not zero.
- **Output policy is unresolved.** The default truncates every result to a snippet plus
  citation. Whether permissively licensed sources should return full text is an open
  decision (ARCHITECTURE.md §8).

## Licence

MIT
