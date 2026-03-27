# Ask the Docs — V3 Plan: Production-Grade RAG

## Scope decision — V3 vs V4

V3 makes the RAG pipeline production-grade: better retrieval quality, proper observability,
reliability, evaluation, and all the PROD NOTEs that are achievable without a fundamental
architectural change.

V4 goes further into **Agentic RAG** — multi-step reasoning, tool use, self-correction, and
query routing. That is a different architectural paradigm and belongs in its own plan.

---

## Problem 1: Sparse encoder is TF-only — no IDF

**Current:** `sparse-encoder.ts` computes raw term frequency. The word "component" in a Next.js
doc scores the same as the word "useRouter" even though "component" appears in every chunk and
says nothing about which one to pick.

**Why it matters:** Without IDF, common terms dominate sparse scores. The sparse side of hybrid
search is weaker than it should be.

**V3 fix: BM25 with a running IDF table**

Two-pass ingest:
- Pass 1 (happens at the end of every ingest run): walk all chunks in Qdrant, count how many
  documents contain each term → store in `data/idf-table.json`
- Pass 2 (at encode time): use the IDF table to weight terms

```
BM25(term, chunk) = TF(term, chunk) × IDF(term)
IDF(term) = log((N - df + 0.5) / (df + 0.5) + 1)
  where N = total chunks, df = chunks containing the term
```

IDF table is rebuilt after every ingest. During query time the table is loaded once and cached
in memory for the process lifetime.

Libraries: `wink-bm25-text-search` was considered but manages a full document corpus internally
(feed it documents, query against it). Its API is designed for standalone search, not for
producing `{ indices, values }` sparse vectors for Qdrant. The BM25 IDF formula is a single
line — implemented manually in `idf-table.ts` with no extra dependency.

---

## Problem 2: No re-ranking after retrieval

**Current:** The top-5 chunks from hybrid search go directly into the prompt. Hybrid search
is good at recall (finding the right chunks) but imprecise at precision (ranking them correctly).

**Why it matters:** The LLM only reads the chunks in order. If the most relevant chunk is ranked
#4, the answer quality drops. A cross-encoder re-ranker reads the (query, chunk) pair together
and scores relevance much more accurately than embedding similarity alone.

**V3 fix: Cross-encoder re-ranking**

After hybrid search returns 20 candidates, pass each (question, chunk) pair through a
cross-encoder model. Sort by cross-encoder score. Take top 5 for the prompt.

```
hybrid search → 20 candidates → cross-encoder scores each → top 5 → prompt
```

Cross-encoder options:
- `Xenova/ms-marco-MiniLM-L-6-v2` via `@xenova/transformers` — runs in Node.js, no Python
- Cohere Rerank API — hosted, one API call, no local model needed
- `cross-encoder/ms-marco-electra-base` — better quality, heavier

The Xenova option keeps everything local and TypeScript. The Cohere option is the simplest
integration for production if you already have API access.

---

## Problem 3: Retrieval misses when query wording differs from doc wording

**Current:** The raw user question is embedded and searched. If the user asks "how do I handle
404s?" but the docs say "not found pages", the embedding might miss it.

**V3 fix: Query expansion**

Generate N alternative phrasings of the question using the LLM, run retrieval for each, merge
results (deduplicate by chunk ID), then re-rank the merged pool.

```
question → LLM generates 3 variants → 4 parallel searches → merge + deduplicate → re-rank → top 5
```

This costs one extra LLM call per query (small, fast with a cheap model) but significantly
improves recall for queries where the user's vocabulary differs from the docs.

Alternative: **HyDE (Hypothetical Document Embeddings)** — instead of expanding the question,
use the LLM to write a hypothetical answer, then embed that answer and search with it. The
hypothesis lives in the same semantic space as the docs. Works well when queries are short and
vague.

---

## Problem 4: Contextual retrieval — chunks lack surrounding context

**Current:** Each chunk is embedded as-is. A chunk that says "This only works in Server
Components" has no context about *what* only works in Server Components — that's in the heading
or the paragraph above.

**Why it matters:** Without context, the embedding encodes the wrong meaning. The chunk is
unsearchable for the concept it actually covers.

**V3 fix: LLM-generated context prefix per chunk (Anthropic's contextual retrieval)**

At ingest time, for each chunk, call a small fast LLM to generate one sentence of context:

```
prompt: "Here is a section from Next.js docs. Write one sentence describing what this section
covers in relation to the full document. Be specific about API names and feature names."

output: "This chunk describes how to configure the Image component's lazy loading behaviour
in the App Router."
```

Prepend that sentence to the chunk before embedding. The embedding now carries the right
meaning even for chunks that are mid-section fragments.

Reported result: ~49% reduction in retrieval failures (Anthropic, 2024).

Cost: one LLM call per chunk at ingest time. Amortized across queries, this is cheap.
Only re-runs for changed chunks (incremental ingest already handles this).

---

## Problem 5: Conversation history is truncated, not summarized

**Current:** Last 3 turns are included verbatim. Early context in long conversations is lost.

**V3 fix: Progressive summarization**

When history exceeds N turns, summarize the oldest turns into a single paragraph:

```
[Summary of earlier conversation: The user asked about App Router layouts and learned that
nested layouts are defined by nesting layout.tsx files...]
[Turn 4 — verbatim]
[Turn 5 — verbatim]
[Turn 6 — verbatim]
```

The summary is stored in the frontend state alongside the messages and sent with each request.
Summarization happens client-side via one LLM call when the turn count crosses the threshold.

---

## Problem 6: No metadata on chunks — no filtered search

**Current:** Chunks only store `source`, `title`, `sectionTitle`, `chunkIndex`. No date, no
doc type, no URL.

**Why it matters:** Users cannot ask "show me only App Router docs" or "find recent changes".
Qdrant supports filtered search (filter by payload field before vector search) but we have
nothing to filter on.

**V3 fix: Enrich chunk metadata at ingest time**

Add to every chunk:
```ts
lastModified: string   // fs.statSync().mtime.toISOString()
docType: string        // "guide" | "api-reference" | "error" — inferred from path
url: string            // https://nextjs.org/docs/{path without number prefixes}
```

And in `similaritySearch`, support an optional `filter` parameter:
```ts
similaritySearch(embedding, sparse, topK, filter?: QdrantFilter)
```

The API route can accept a `filter` from the frontend (e.g. `{ docType: "guide" }`) and pass
it through. The UI can expose a doc type selector.

---

## Problem 7: Chunker does not handle MDX structure

**Current:** The chunker splits on `##` / `###` headings but does not handle:
- Front-matter (`--- yaml ---` blocks at the top of MDX files)
- Code blocks (a split inside a code block produces broken fragments)
- Tables (a half-table is meaningless as a chunk)

**V3 fix: MDX-aware chunker**

- Strip front-matter before splitting (simple regex: `/^---[\s\S]*?---\n/`)
- Detect code fences (`` ``` ``) and never split inside one
- Detect table rows (`| col |`) and keep entire tables in one chunk

This is achievable with careful regex without adding a full MDX parser dependency.

---

## Problem 8: No observability

**Current:** Console.log only. No structured logs, no latency tracking, no request IDs.
No way to know if retrieval is slow or if Qdrant calls are failing silently.

**V3 fix: Structured logging + latency tracking**

Add a thin logger utility:
```ts
log.info({ requestId, event: 'retrieval', durationMs: 142, chunkCount: 5, scores: [...] })
```

Track and log latency for each stage separately:
- Embedding latency
- Qdrant search latency
- Re-ranking latency
- LLM first-token latency

Wire `healthCheck()` into a `/api/health` route that returns Qdrant status + uptime. This is
what a load balancer or Kubernetes calls before routing traffic.

---

## Problem 9: No eval pipeline

**Current:** We have no way to measure if retrieval is getting better or worse. Every change
to the chunker, embedder, or sparse encoder is a guess.

**V3 fix: Eval pipeline with golden dataset**

Build a `scripts/eval.ts` that:
1. Loads a golden dataset: `data/eval.json` — a list of `{ question, expected_sources[], acceptable_answers[] }`
2. Runs each question through the full pipeline
3. Computes metrics:
   - **Context recall**: did the expected source appear in the retrieved chunks?
   - **Faithfulness**: does the answer contradict any retrieved chunk? (LLM judge)
   - **Answer relevance**: does the answer address the question? (LLM judge)
4. Outputs a report: overall scores + per-question breakdown

The golden dataset starts small (20-30 questions) and grows over time. Every retrieval change
is validated against it before merging.

---

## Problem 10: Reliability gaps

**Current:**
- Retry has no jitter (thundering herd risk)
- No distinction between retryable and non-retryable errors
- No circuit breaker
- No rate limiting on `/api/chat`

**V3 fix:**

Replace the hand-rolled retry with `p-retry`:
```ts
await pRetry(() => client.query(...), {
  retries: 3,
  onFailedAttempt: (err) => { if (err.statusCode === 400) throw err }, // don't retry bad requests
  minTimeout: 500,
  randomize: true, // adds jitter
})
```

Add rate limiting to `/api/chat` with `lru-cache` as a simple in-memory store:
```
100 requests per IP per minute → 429 Too Many Requests
```

---

## Problem 11: Parallel ingest

**Current:** Files are processed sequentially. For large doc sets this is slow.

**V3 fix:** Use `p-limit` to process N files concurrently:
```ts
const limit = pLimit(5) // 5 concurrent files
await Promise.all(files.map((f) => limit(() => processFile(f))))
```

Embedding batches within each file can also run concurrently with a separate semaphore.

---

## Problem 12: Ingest cache is file-level — one changed byte re-embeds the whole file

**Current:** `ingest-cache.json` stores `{ source → MD5(fileContent) }`. If any byte in a file
changes, all chunks for that file are deleted and re-embedded from scratch.

For small docs pages this is fine. For large files (long reference pages, generated API docs),
a single paragraph edit re-embeds hundreds of sections — hundreds of unnecessary embedding API calls.

**Why it matters:** At scale (large files, frequent doc updates, paid embedding APIs) this wastes
money and time. A 500-section file edited in one place should re-embed 1 section, not 500.

**V3 fix: Section-level cache**

Promote the cache from file-level to section-level:

```
Before:
  cache: { "routing/intro.md": "a3f9c1..." }

After:
  cache: {
    "routing/intro.md": {
      "App Router > Introduction":    { hash: "a3f...", chunkIds: ["uuid1"] },
      "App Router > Getting Started": { hash: "b72...", chunkIds: ["uuid2", "uuid3"] }
    }
  }
```

On each ingest run for a changed file:
1. Re-split the file into sections (same `splitIntoSections` logic)
2. Hash each section's content independently
3. Compare against the cached section hashes
4. **Skip** sections whose hash is unchanged — no embedding call, no Qdrant write
5. For changed/new sections: embed and upsert only those chunks
6. For removed sections (in cache but not in new split): delete only those chunk IDs

This makes ingest O(changed sections) instead of O(all sections in changed file).

**Why delete only the stale chunk IDs instead of deleteChunksBySource?**

With section-level tracking, the cache knows exactly which chunk IDs belonged to each section.
On a section change, delete only those IDs → upsert new ones. The rest of the file is untouched.
`deleteChunksBySource` is the nuclear option — reserved for full re-ingests only.

**Tradeoff:** The cache file grows larger (section entries instead of one hash per file) and
the ingest logic is more complex. Worth it once file sizes or update frequency make file-level
re-embedding expensive.

---

## Problem 13: Context generation is sequential — ingest takes 15+ minutes with no timing visibility

**Current:** `embedAndStore` generates a context prefix for each chunk one at a time
(`ingester.ts:139-143`). Each call is an LLM round-trip — ~1–3 s on Ollama. With 15 chunks
per file and hundreds of files, this dominates ingest time. There is also no per-stage timing
output, so it is impossible to know which step is slow without a profiler.

**Why it matters:** A full ingest that takes 15+ minutes discourages doc updates and makes
debugging retrieval quality painful.

**V3 fix: Parallel context generation with `p-limit` + ingest timing**

Replace the sequential loop with concurrent calls controlled by a semaphore:

```ts
import pLimit from 'p-limit'

const limit = pLimit(Number(process.env.CONTEXT_CONCURRENCY ?? 3))

const contextualContents = await Promise.all(
  chunks.map((chunk) =>
    limit(async () => {
      const ctx = await generateContextPrefix(doc.title, chunk.sectionTitle, chunk.content)
      return ctx ? `${ctx}\n\n${chunk.content}` : chunk.content
    })
  )
)
```

Default concurrency:
- Ollama local: `3` (CPU/GPU bound — higher values thrash the model server)
- OpenAI/GitHub API: `20` (network-bound — set via `CONTEXT_CONCURRENCY=20`)

Add wall-clock timing at each stage so slow steps are visible in the output:
```
[install.mdx] context: 4.2s | embed: 0.8s | upsert: 0.2s → 18 chunks
=== Ingestion complete in 3m 12s ===
```

**Files to change:**
- `lib/rag/ingester.ts` — parallel context loop + per-file and total timing
- `package.json` — add `p-limit` as a direct dependency
- `.env.example` — document `CONTEXT_CONCURRENCY`

---

## V3 File Changes

```
lib/
  rag/
    sparse-encoder.ts     ← REWRITE: BM25 with IDF table
    idf-table.ts          ← NEW: build + load IDF table from corpus
    chunker.ts            ← UPDATE: front-matter strip, code block + table preservation
    ingester.ts           ← UPDATE: contextual prefix per chunk, parallel processing, metadata, section-level cache
    ingest-cache.ts       ← UPDATE: section-level hashing (source → section → { hash, chunkIds })
    retriever.ts          ← UPDATE: query expansion, re-ranking, summarized history
    reranker.ts           ← NEW: cross-encoder re-ranking (Xenova or Cohere)
    query-expander.ts     ← NEW: LLM-based query expansion
    context-builder.ts    ← NEW: LLM-generated context prefix per chunk
    eval.ts               ← NEW: evaluation metrics

  vectordb/
    qdrant.client.ts      ← UPDATE: metadata filter support, better retry (p-retry), gRPC option

  logger.ts               ← NEW: structured logging utility

app/
  api/
    chat/route.ts         ← UPDATE: rate limiting, request IDs, structured logging
    health/route.ts       ← NEW: /api/health → Qdrant status + uptime

scripts/
  eval.ts                 ← NEW: npm run eval — run golden dataset, print metrics

data/
  idf-table.json          ← Generated (gitignored)
  eval.json               ← Checked in — golden Q&A dataset
```

## Implementation Order

### Step 1 — MDX-aware chunker + metadata enrichment
Improves chunk quality and unlocks filtered search. No new dependencies.

### Step 2 — BM25 sparse encoder with IDF table
Biggest retrieval quality improvement after re-ranking. Pure TypeScript with `wink-bm25-text-search`.

### Step 3 — Contextual retrieval (LLM context prefix)
High-impact, self-contained ingest change. Only affects ingest, not query time.

### Step 4 — Re-ranking
Add cross-encoder re-ranking with Xenova. Independently testable.

### Step 5 — Query expansion
Add query variant generation. Needs the re-ranker to handle the larger candidate pool well.

### Step 6 — Conversation summarization
Frontend + retriever change. Independently testable with long conversations.

### Step 7 — Observability + reliability
Logger, health endpoint, rate limiting, p-retry. Unglamorous but essential.

### Step 8 — Section-level ingest cache
Upgrade `ingest-cache.ts` to track section hashes and chunk IDs. Update `ingester.ts` to diff
at section granularity. Run with a large file to verify only changed sections are re-embedded.

### Step 9 — Eval pipeline
Build golden dataset. Run eval before and after each step above to measure actual impact.
