# model-chef architecture

A curated, provenance-tagged retrieval index over LLM **post-training** knowledge —
papers, official cookbooks, and a hand-curated allowlist of company blogs — exposed
as an MCP server so any harness can query it.

The design goal is a specialist research assistant with access to current, high-signal
work, not a citation lookup. That means full-text parsing including tables, and an
index that keeps up with new venues and releases.

---

## 1. The shape: a funnel

Discovery is nearly free, deep parsing is expensive, and the ratio between them is
roughly 100:1. Everything follows from that.

```
DISCOVER   ~2,000 items/week   metadata only, many APIs      ~$0
    ↓
TRIAGE     ~2,000 → ~150       cascade of cheap gates        ~$0
    ↓
PARSE      ~150 items/week     full text, tables, sections   ~1 CPU-hour/week
    ↓
CHUNK → EMBED → UPSERT
```

The **generic spine** is triage → parse → chunk → embed → upsert, written once.
Source-specific code exists only in **discovery** and **fetch**. Adding a source is
one discovery module plus an allowlist entry.

---

## 2. Triage

A cascade, cheapest signal first, so expensive signals only run on survivors.

### Gate 0 — hard filters (free)
Category allowlist (`cs.LG`, `cs.CL`, `cs.AI`), date window, dedupe against the index
by DOI/arXiv id, language. Removes ~70%.

### Gate 1 — topical relevance (~1ms)
Embed title + abstract, cosine against ~15 hand-written **topic centroids** — short
paragraphs describing what model-chef covers (verifiable-reward post-training,
preference optimisation objectives, reward modelling and overoptimisation, RL
infrastructure for LLMs, …). Take max similarity, threshold it.

The centroids live in `src/model_chef/data/centroids.toml`. They **are** the curation
policy, expressed as reviewable data.

### Gate 2 — quality signal (free, metadata only)
Weighted score over signals already fetched during discovery:

| Signal | Source | Why |
|---|---|---|
| Reviewer scores + decision | OpenReview | Best free quality signal available; exists before citations do |
| Citation velocity | OpenAlex / S2 | Works for anything older than ~6 months |
| Influential citation count | Semantic Scholar | Filters perfunctory citations |
| Daily Papers upvotes | HuggingFace | Fast social signal, days after release |
| Author affiliation | OpenAlex | Against a curated org list |
| Linked code | arXiv / GitHub | Proxy for reproducibility |
| Venue tier | OpenReview / ACL Anthology | Peer review status |

### Gate 3 — manual review queue (free)
There is **no LLM adjudication stage**. Items in the ambiguous band append to
`review_queue.jsonl` with title, abstract, signals and link. A human skims it weekly
and verdicts are recorded in `curation_decisions.toml`, checked into git.

This is deliberately a human step rather than a model call:

- It costs nothing and removes an external dependency from ingestion.
- It makes the project's central claim literally true — every borderline admission was
  reviewed by a person, and the decision log is in version control.
- The accumulated verdicts are a **labelled set**, so gate 1's threshold can be tuned
  against real decisions and its precision/recall reported.

Expected volume: 30–60 items/week.

### "Latest" and "best" conflict — two admission paths

A paper published yesterday has no citations, so admission has two routes:

- **Merit** — high citation velocity, or accepted at a strong venue with good reviews.
- **Provisional** — recent, with strong social or affiliation signal. Admitted with
  `provisional=true` and `rescore_after = published_date + 90d`.

A weekly **re-scoring and eviction pass** revisits provisional documents: those that
accumulated real signal are promoted, those that did not are evicted. The index is
mutable and self-cleaning.

Because there is no automatic adjudicator, **gate 1 is biased toward recall**.
Admitting a dud is cheap and reversible; never discovering a paper is invisible and
permanent.

---

## 3. Discovery sources

| API | Auth | Limits | Role |
|---|---|---|---|
| arXiv OAI-PMH | none | polite; 429s reported in 2026 | Bulk metadata firehose, daily |
| arXiv API | none | 3 req/s | Targeted search |
| OpenReview v2 (`openreview-py`) | none | generous | Venue papers + reviewer scores + decisions |
| OpenAlex | none | 100k/day, 10/s | Citations, DOI resolution, institutions |
| Semantic Scholar | free key | **1 req/s** — batch endpoints only | Influential citations, reference graph |
| HuggingFace Daily Papers | none | — | Trending / social signal |
| ACL Anthology | none | — | ACL/EMNLP/NAACL |
| GitHub | token | 5k/hr | Cookbook repos, code-link signal |
| Blog RSS/Atom | none | — | Company blogs, from the allowlist |

Semantic Scholar's 1 req/s is the binding constraint: use batch endpoints only, and
only for already-admitted documents. OpenAlex is the primary citation source.

### Venue-triggered ingestion

`src/model_chef/data/venues.toml` holds each conference with its OpenReview id and
expected decision date. A daily cron checks whether any venue is inside its poll
window and, if so, queries OpenReview for public decisions; when they land it enqueues
a bulk backfill. Outside the window it is one API call. Adding ICML/ICLR/COLM is a
config-file PR.

---

## 4. Parsing, and why tables work

**arXiv HTML beats PDF parsing, and it isn't close.** arXiv HTML is LaTeXML-derived,
so tables arrive as real `<table>` markup with correct cell structure, sections are
real headings, and maths is MathML. PDF table extraction — borderless tables, merged
cells, multi-column layouts — is the genuinely hard case. Avoid it where possible.

Fetch cascade, recorded per document in `text_depth`:

1. `full_html` — arXiv HTML (papers from Dec 2023 onward)
2. `ar5iv` — older papers, best-effort
3. `pdf_docling` — **Docling** (MIT, CPU-viable, RT-DETR layout + TableFormer)
4. `abstract_only` — nothing else available

**Docling, not Marker.** Marker is GPL-3.0 with RAIL-M weights restricting commercial
use above a revenue threshold. Docling is MIT and emits a structured document built
for RAG. MinerU is strongest on CJK, which is not this corpus.

### Table handling

Tables are **atomic chunks, never split** (`chunk_kind="table"`). The embedding text
is deliberately *not* a dump of cell values — see `Table.embedding_text()`. What makes
a table findable is its caption, heading path and column headers, not its numbers. The
full serialised markdown lives in the document store.

### Section typing

Every chunk carries a `section_type`. Users want methods and results; nobody wants
related-work prose. Down-weighting `related_work` and dropping `references` is a few
lines in ranking and improves every query. It comes free from HTML heading structure.

---

## 5. Storage

### Why not a managed vector database

Managed vector DBs price on **RAM**, because HNSW must be memory-resident to be fast —
so you rent an always-on VM. At ~$57/GB-RAM/month, Qdrant Cloud wanted $30–60/month.
The actual data is ~650MB. At 1M vectors with light traffic that is paying two orders
of magnitude over the odds for idle capacity.

### The Cloudflare stack

| Store | Holds |
|---|---|
| **Vectorize** | 1024-dim vectors + 6 short filterable scalars |
| **D1** (SQLite) | chunk text, serialised tables, full payload, FTS5/BM25 index |
| **R2** | archive of parsed documents, pre-chunking |
| **KV** | issued `MODEL_CHEF_API_KEY`s |
| **Workers** | the API: auth, rate limit, query embedding, search, fusion |

Three things make this work:

1. **`@cf/baai/bge-m3` on Workers AI solves query embedding.** A Worker cannot run a
   0.6B PyTorch model, so hosted embedding was always required. BGE-M3 is open-weight,
   8192-token context, and emits dense, sparse and ColBERT representations. Ingestion
   embeds **locally** with `sentence-transformers` for free; only queries hit Workers AI.
   *A CI test must assert local and hosted embeddings agree within tolerance* — silent
   version skew is invisible and corrupts retrieval quality.
2. **D1 fixes Vectorize's metadata limits.** Vectorize caps metadata at 10KiB/vector,
   10 metadata indexes, and 64 indexed bytes per field. Text and full payload live in
   D1; Vectorize holds vectors plus `source_type`, `venue_tier`, `chunk_kind`,
   `section_type`, `year`, `provisional`. See `Chunk.filter_payload()`.
3. **Hybrid search via FTS5.** Vectorize has no sparse vector support, which matters
   because dense retrieval fails on exact technical tokens (`GRPO`, `Qwen3-32B`,
   `β=0.1`). D1 is SQLite, so BM25 is an FTS5 virtual table. Query both, fuse with RRF
   in the Worker.

**The honest cost:** two stores that can drift. A reconciliation script and a nightly
consistency check are required, not optional.

### Cost at 1M chunks, ~10k queries/month

| Line item | Cost |
|---|---|
| Workers Paid (fixed floor) | $5.00 |
| Vectorize storage — 1.02B stored dims, 10M included | $0.51 |
| Vectorize queried — (10k queries + ~90k new chunks) × 1024, 50M included | $0.52 |
| D1 — ~2GB, within 5GB included | $0.00 |
| R2 — ~20GB archive | $0.30 |
| Workers AI — query embedding | ~$0.00 |
| **Total** | **≈ $6.33/month** |

Scaling notes:
- **Traffic barely matters.** 1M queries/month ≈ $16/month.
- **A full re-embed costs ~$10 one-time.** The number to remember before changing model.
- **D1 caps at 10GB per database and cannot be raised** — roughly 5M chunks. Beyond
  that, shard by `source_type` or year, or move text to R2.
- **Abuse is the real risk to the bill.** Per-key rate limiting in the Worker is
  load-bearing.

### Keeping the exit open

Ingestion writes through the `VectorStore` / `DocumentStore` protocols in `stores.py`
and never imports a provider SDK. Only the serving layer is Cloudflare-specific, so
porting to self-hosted Qdrant is a new adapter plus a reindex. R2 holds the parsed
documents and ingestion is reproducible from source APIs, so the corpus is always
rebuildable.

---

## 6. Serving

```
harness (Claude Code / Desktop / Cursor)
   ↓ MCP stdio
model-chef package  (tiny — no model weights, no DB driver)
   ↓ HTTPS
model-chef API (Worker)  → auth, per-key rate limit, embed, search, RRF, rerank
   ↓
Vectorize + D1
```

**A Qdrant/Vectorize key can never ship inside the package** — it would be public, and
one retry loop would run up the maintainer's bill. The Worker is what makes "connect
your own harness" safe.

**Query embedding happens server-side** for two reasons: the package stays small
instead of pulling a 1.2GB model on first run, and index-time and query-time
embeddings are guaranteed to use the same model.

The server **never calls an LLM provider**. It returns retrieved, cited chunks;
generation happens in whatever harness the user is running. That is what makes
"works with Claude/GPT/local" true for free.

### Tool surface

- `search_recipes(query, k=5, min_venue_tier=None)` — main entrypoint
- `get_source(chunk_id)` — full citation metadata
- `list_recent_updates(since_date)` — transparency into what ingest added

---

## 7. Ranking

```
score = cosine × tier_weight × recency_factor × section_weight

tier_weight:     peer_reviewed 1.00, official_docs 1.00, preprint 0.90, blog 0.85
recency_factor:  max(0.70, exp(-age_days / 540))
section_weight:  results/method 1.00, related_work 0.75, references dropped
```

The floor at 0.70 stops a strong 2022 paper being buried under a mediocre recent blog
post. Weights live in one config module; the tests assert **ordering properties**
("given equal similarity, peer-reviewed outranks blog") rather than pinning magic
numbers.

Venue tier is derived, not curated: arXiv `journal_ref` and comments ("Accepted at
NeurIPS 2025") and OpenReview decisions determine it. No hand-maintained venue ranking.

---

## 8. Output policy — **open decision**

`snippet_only` (current default) truncates every result to `MAX_SNIPPET_CHARS`,
uniformly, regardless of licence. Safe and easy to explain.

`license_aware` serves full chunk text for permissively licensed sources (CC-BY papers,
MIT/Apache cookbooks) and snippet + link for everything else, with attribution always.

**This is unresolved.** The goal of parsing tables so an agent can reason over results
is in real tension with snippet-only — a truncated table is useless. Both paths are
implemented and tested in `SearchResult.from_chunk`; the default is the conservative
one until the call is made. It is a licensing decision, not a technical one.

---

## 9. Roadmap

Each phase ends demoable.

1. **Scaffold** — packaging, CI, `schemas.py`, store protocols. ✅
2. **arXiv HTML ingestion + gates 0–1 + Vectorize/D1 + hybrid search**
3. **Tables + section typing** — what makes it more than a citation tool
4. **OpenReview + venue calendar** — reviewer scores into triage
5. **Blogs/news + social signals + provisional admission & eviction**
6. **Worker API + auth + packaging** — the point others can use it
7. **Structured results extraction** — parse result tables into typed rows
   (`model`, `task`, `metric`, `value`, `config`) so the agent can answer *"what KL
   coefficient did people use for GRPO on 7B models"*. Design table chunks now so this
   is additive.

## 10. Known limitations

- Index freshness is bounded by the ingest cadence (daily discovery, weekly deep pass).
- `abstract_only` documents are retrievable but shallow; the `text_depth` breakdown is
  reported so coverage is honest.
- Pretraining and SFT are out of scope. Post-training only.
- Blog coverage is an explicitly curated allowlist, never open crawling.
