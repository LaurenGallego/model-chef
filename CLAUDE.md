# model-chef

## What this is

A curated, provenance-tagged, continuously-updated retrieval index over LLM
**post-training** knowledge — papers, official cookbooks, and a hand-curated allowlist
of company blogs — exposed as an **MCP server** so any harness (Claude Code, Claude
Desktop, Cursor) can query it as a tool.

The goal is a specialist research assistant with access to current, high-signal work:
full papers including tables and results, not a citation lookup. It is hosted on the
maintainer's own infrastructure and other people connect their own harness to it.

**Full design lives in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).** Read it before
making architectural changes. This file is the working agreement; that file is the design.

## Scope

**In scope:** RL / post-training — RLHF, RLAIF, GRPO, DPO, PPO, reward modelling,
verifiable rewards, post-training evaluation.

**Out of scope for now:** pretraining and SFT. Note them as future work in the README
rather than building them.

Source types:
- Official cookbooks/docs with permissive licences (OpenAI, Anthropic, HF, TRL)
- Papers from arXiv, OpenReview (ICLR/NeurIPS/ICML/COLM), ACL Anthology
- A short, explicitly curated allowlist of company blogs — every entry added by hand,
  never auto-discovered

**Do not silently expand scope — flag it first.** This project has already grown once,
deliberately, from a weeks-long portfolio piece to a hosted service with a real
monthly cost. Further growth should be a decision, not a drift.

## Architecture in one screen

```
src/model_chef/
  schemas.py        → Pydantic contracts. The file everything depends on.
  stores.py         → VectorStore / DocumentStore protocols (provider-agnostic)
  config.py         → settings, env-overridable
  ingestion/        → discover → triage → fetch → parse → chunk → embed → upsert
    discovery/      → one module per API; emits Candidate (metadata only, cheap)
    triage/         → the gate cascade + topic centroids
    fetch/ parse/ chunk/ embed/
  server/           → the MCP server (thin — no generation, no LLM calls)
    tools.py        → MCP tool definitions
    retrieval.py    → hybrid query + venue/recency/section-weighted ranking
  data/             → centroids.toml, venues.toml, blogs.toml — curation as reviewed data
scripts/            → backfill, re-embed, reconcile, evict
docs/ARCHITECTURE.md
```

### Rules that are load-bearing

- **The server never generates text and never calls an LLM provider.** It returns
  retrieved, cited chunks. Generation happens in the user's harness — that is what makes
  "works with Claude/GPT/local" true for free instead of something we build.
- **Ingestion never imports a provider SDK directly.** It writes through the protocols
  in `stores.py`. Only the serving layer is Cloudflare-specific. This is what keeps the
  exit open and lets tests run without network.
- **`chunk_id` is always derived via `make_chunk_id()`.** `schemas.py` enforces it.
  It is what makes re-ingestion idempotent instead of duplicating.
- **Triage has no LLM stage.** The ambiguous band goes to a human review queue whose
  verdicts are checked into git. That is the curation claim, and it is also a labelled
  set for tuning gate 1.
- **Populate `venue_tier`, `published_date`, `section_type`, `heading_path` and
  `text_depth` at parse time or not at all.** They are the most expensive fields to
  backfill.

## Code quality bar — this is the point of the project

This is a CV artifact. Prioritise demonstrable engineering discipline over feature count.

- **No over-engineering.** Quality means effective, simple, readable — not exhaustive
  corner-case handling.
- **Type hints**, `mypy --strict` passing in CI. Reasonable, not necessarily everywhere.
- **Minimal comments**: minimal explanations and comments, for particular classes & functions  
  that may require it, but not general for files, decisions, ...
- **Pydantic** at every data boundary — no loose dicts crossing module boundaries.
- **`ruff`** for lint + format, via pre-commit.
- **Tests**: ranking logic and schema validation are the highest-value tests here.
  The ingestion pipeline is integration-tested against a small fixture set rather than
  mocked at every layer.
- **CI** on every PR: lint, format, typecheck, test. Worth more to a reader of the repo
  than most features.
- **Structured logging** (`structlog`) in ingestion specifically — when a scheduled
  scrape silently breaks six months from now, logs are how anyone diagnoses it.
- **`pyproject.toml`** as the single source of packaging config, `uv` for dependencies.
- **No secrets in the repo.** `.env.example` committed, `.env` gitignored.
- **README** must include: what it is, why the scope is narrow (say so explicitly, so a
  reader does not assume it is unfinished), architecture diagram, how ingest works,
  install/config, and an honest "known limitations" section.

## Legal / licensing — not a "later" item

- **Output policy is an open decision.** `snippet_only` (current default) truncates
  every result uniformly. `license_aware` serves full text for permissively licensed
  sources only. Both are implemented in `SearchResult.from_chunk`; the conservative one
  is the default until the call is made. See ARCHITECTURE.md §8.
- Respect each source's robots.txt / ToS. Rate-limit politely — ar5iv in particular is a
  small academic service, not an API.
- Record licence **per document**, from source metadata. arXiv licences vary per paper.
- Attribute always, whatever the output policy.
- The blog allowlist is manually curated specifically so licensing can be eyeballed per
  source before it is added.

## Current state

Phase 1 complete: packaging, CI, `schemas.py`, `stores.py`, `config.py`, 26 tests,
`ruff` + `mypy --strict` green.

Next: phase 2 — arXiv HTML ingestion, gates 0–1, Vectorize/D1 adapters, hybrid search.

## Open decisions (flag before assuming an answer)

- **Output policy** — `snippet_only` vs `license_aware` (ARCHITECTURE.md §8)
- **Corpus size target** — 1M chunks assumed for costing; D1 caps at ~5M
- **Key distribution** — manual issuance to start; a signup flow is not built
- Exact gate-1 centroid threshold — to be tuned against the manual review log
