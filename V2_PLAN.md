# Ask the Docs — V2 Plan

## What we're improving and why

---

## Problem 1: Flat JSON is not a vector database

**V1**: All chunks + embeddings stored in `data/vector-store.json`. Loaded entirely into memory on every query. Every search scans every vector (O(n)).

**Why it breaks at scale**: 500 docs × 10 chunks = 5,000 chunks. Each embedding is ~6,144 bytes (1536 floats × 4 bytes). That's ~30MB loaded on every single request.

**V2 fix**: Replace `lib/vectordb/vector-store.ts` with Qdrant.
- Runs locally via Docker — no cloud account needed
- Stores vectors on disk, indexes with HNSW (O(log n) search)
- Supports hybrid search natively (dense + sparse in one query)
- TypeScript client: `@qdrant/js-client-rest`
- Same public API shape — swap the implementation, nothing else changes

```bash
# One command to start
docker run -p 6333:6333 qdrant/qdrant
```

---

## Problem 2: Character chunking cascades on any edit

**V1**: `RecursiveCharacterTextSplitter` splits every 1500 chars. Adding content anywhere shifts every subsequent chunk boundary. Chunk IDs are positional (`source-0`, `source-1`...) so they become stale silently.

**Why it breaks**: Add 2000 chars to a doc → every chunk after the insertion is a new mix of old+new content → all IDs stay the same → wrong embeddings stored under correct-looking IDs. The store is silently corrupted.

**V2 fix**: Two-pass structural chunking.
1. Split on `##` / `###` headings first → each section is an independent unit
2. If a section exceeds 1500 chars, sub-split it with character splitter
3. ID = `md5(source + content)` — content-addressed, not positional

Result: editing one section only affects that section's chunks. All other sections have stable IDs and are skipped during re-ingest.

---

## Problem 3: No incremental ingestion — re-embeds everything every run

**V1**: Every ingest call re-embeds every chunk from every file, even unchanged ones. Orphan chunks from deleted/shortened docs are never removed.

**Why it matters**: Embedding APIs cost money and time. Re-embedding 500 unchanged files to pick up 1 changed file is wasteful. Orphan chunks cause the LLM to cite deleted content.

**V2 fix**: File-hash cache + delete-before-upsert strategy.

```
data/ingest-cache.json  ← { "source/file.mdx": "md5-of-file-content" }
```

On each ingest run per file:
1. Hash the file content
2. Compare against cache — if unchanged, skip entirely
3. If changed: delete all existing chunks for this source, re-chunk, re-embed, upsert
4. If file deleted: delete its chunks, remove from cache

New command flags:
```bash
npm run ingest           # incremental — skip unchanged files
npm run ingest --full    # force re-embed everything (for model changes)
npm run ingest --clear   # wipe store and re-ingest all
```

---

## Problem 4: Cosine-only retrieval misses exact terms

**V1**: Only semantic (dense) search. "What is the `useRouter` hook?" — the word `useRouter` might score lower than a semantically similar chunk that doesn't mention it.

**Why it matters**: Technical docs have exact API names, function signatures, config keys. Semantic search alone misses them. Keyword search alone misses meaning. You need both.

**V2 fix**: Hybrid search in Qdrant.
- Dense vector: embedding similarity (meaning)
- Sparse vector: BM25 keyword score (exact terms)
- Qdrant fuses both scores with Reciprocal Rank Fusion (RRF)
- Single query, single round trip

---

## Problem 5: Each question is independent — no conversation memory

**V1**: The API receives `{ question }` only. Follow-up questions like "how do I use that with TypeScript?" have no context.

**V2 fix**: Pass conversation history to the API.
- Frontend sends `{ question, history: Message[] }` (last N turns)
- Retriever builds prompt with history before the current question
- Simple, no new dependencies

---

## Problem 6: Score threshold — retrieving irrelevant chunks

**V1**: Always returns top 5 chunks regardless of score. A score of 0.15 means essentially unrelated, but it gets sent to the LLM anyway.

**V2 fix**: Filter chunks below a minimum score threshold before building the prompt.

```ts
const MIN_SCORE = 0.30
const relevant = results.filter(r => r.score >= MIN_SCORE)
if (relevant.length === 0) return "I couldn't find relevant information."
```

---

## What we're NOT doing in V2 (and why)

| Feature | Why skipped |
|---|---|
| Re-ranking (cross-encoder) | Adds latency + another model dependency; hybrid search covers most of the gain |
| Query rewriting | Adds an extra LLM call before every query; good for prod, overkill for learning |
| Eval pipeline | Needs a golden dataset; worth a dedicated V3 |
| Auth / rate limiting | Out of scope for a learning project |

---

## V2 File Structure

```
lib/
  llm/
    types.ts          ← unchanged
    factory.ts        ← unchanged
    ollama.client.ts  ← unchanged
    openai.client.ts  ← unchanged
  rag/
    chunker.ts        ← REWRITE: structural chunking + content-hash IDs
    ingester.ts       ← REWRITE: file-hash cache + delete-before-upsert
    retriever.ts      ← UPDATE: score threshold + conversation history
  vectordb/
    qdrant.client.ts  ← NEW: replaces vector-store.ts
    vector-store.ts   ← DELETE (replaced)

scripts/
  ingest.ts           ← UPDATE: --full / --clear flags

app/
  api/
    chat/
      route.ts        ← UPDATE: accept history[] from frontend
  page.tsx            ← UPDATE: send history with each message

data/
  docs/               ← unchanged
  ingest-cache.json   ← NEW: generated, gitignored
  (vector-store.json) ← DELETE: no longer used
```

---

## Implementation Order

Each step is independently testable. Do them in order.

### Step 1 — Qdrant vector DB
- Start Qdrant with Docker
- Install `@qdrant/js-client-rest`
- Write `lib/vectordb/qdrant.client.ts` with the same public API:
  `upsertChunks`, `similaritySearch`, `deleteChunksBySource`, `getChunkCount`, `clearStore`
- Update `ingester.ts` and `retriever.ts` imports
- Run ingest, run a query — verify it works

### Step 2 — Structural chunking + content-hash IDs
- Rewrite `lib/rag/chunker.ts`:
  - Split on `##` / `###` headings first (LangChain `MarkdownHeaderTextSplitter`)
  - Sub-split large sections with character splitter
  - Return `sectionTitle` in chunk metadata
- Change chunk ID to `md5(source + content)` in `ingester.ts`
- Re-ingest, verify chunk count and quality

### Step 3 — Incremental ingestion
- Add `data/ingest-cache.json` (file hash → last-ingested content hash)
- In `ingester.ts`: before processing a file, check cache. Skip if unchanged.
- Add `deleteChunksBySource()` call before any file's upsert
- Add `--full` flag to bypass cache
- Test: run ingest twice, second run should report 0 files re-embedded

### Step 4 — Hybrid search
- Enable sparse vectors in Qdrant collection config
- Update `ingester.ts` to compute sparse (BM25) vectors alongside dense
- Update `qdrant.client.ts` `similaritySearch` to use hybrid query with RRF
- Verify search quality improves for exact API names

### Step 5 — Score threshold + conversation history
- Add `MIN_SCORE = 0.30` filter in `retriever.ts`
- Update `route.ts` to accept `history: Message[]`
- Update `retriever.ts` `buildPrompt` to include history turns
- Update `page.tsx` to send `history` with each request
- Test multi-turn: "How does useRouter work?" → "Can I use that in a Server Component?"
