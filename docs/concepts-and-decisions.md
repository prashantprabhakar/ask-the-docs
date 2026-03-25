# Concepts and Design Decisions

This document explains the concepts behind the RAG pipeline in this project and the reasoning
behind every major decision. If you are reading the code and wondering "why did they do it this
way?" or "what even is TF / IDF / RRF?", the answer is here — with examples.

---

## How docs get into the system — the full pipeline

Before any search can happen, three steps must run in order:

```
npm run fetch-docs    ← pull raw .mdx files from the Next.js GitHub repo
npm run ingest        ← chunk, embed, store in Qdrant
npm run dev           ← the app can now answer questions
```

**Why a fetch step at all?**

The documentation lives in the `vercel/next.js` GitHub repository. We do not copy it into
this repo — that would mean manually updating it every time Next.js releases a new version.
Instead, `fetch-docs` pulls the latest directly from GitHub so the knowledge base stays
current.

**How the download works — tarball streaming**

The naive approach would be to use the GitHub REST API to list every file in the `docs/`
folder and download each one individually. This fails immediately: the API allows 60 requests
per hour without authentication, and the Next.js docs directory has 200+ files. One fetch
run would exhaust the quota.

Instead, `fetch-docs` downloads a single tarball of the entire repo:

```
https://codeload.github.com/vercel/next.js/tar.gz/refs/heads/canary
```

GitHub exposes this endpoint for any repo and branch. It has no per-file rate limit because
it is one request, not hundreds.

The tarball is ~100MB for the full Next.js repo. We do not write it to disk. Instead, the
download stream is piped directly into a tar extractor which filters to only entries inside
`docs/` — so only the documentation files ever touch disk.

**Why handle HTTP redirects manually?**

Node's built-in `https.get` does not follow redirects automatically. GitHub's codeload
endpoint redirects once to a CDN before streaming the file. The script follows up to 5
redirects explicitly so the download works without any third-party HTTP library.

**A subtle tar gotcha — filter receives the original path, not the stripped path**

The `tar` package's `filter` option is called with the entry path as it appears in the
archive — *before* any `strip` is applied. So even though `strip: 1` turns
`next.js-canary/docs/foo.mdx` into `docs/foo.mdx` for extraction purposes, the filter
still sees `next.js-canary/docs/foo.mdx`. The filter must match against the full original
path, not the stripped one.

---

## What is RAG?

RAG stands for Retrieval-Augmented Generation.

An LLM is trained on a fixed snapshot of the internet. It does not know about your private
documentation, your internal codebase, or anything published after its training cutoff. RAG
solves this by adding a retrieval step before generation:

1. **Retrieve** — embed the user's question and search a vector database for the most relevant
   chunks of documentation.
2. **Augment** — insert those chunks into the prompt as context.
3. **Generate** — the LLM answers using only the provided context.

The LLM never needs to be retrained. You update the vector database when your docs change.

---

## What is a vector embedding?

An embedding is a list of floating-point numbers that represents the *meaning* of a piece of
text. The key property: texts with similar meaning end up with vectors that are close together
in that high-dimensional space, even if they share no words.

Example with Next.js docs:

```
"How do I navigate between pages?"        → [0.21, -0.54, 0.87, ...]
"What is the Link component used for?"    → [0.19, -0.51, 0.85, ...]  ← close
"What is the weather today?"              → [-0.73, 0.12, -0.44, ...] ← far
```

The first two are asking about the same thing in different words. An embedding model encodes
that semantic overlap. A keyword search would miss it entirely because "navigate" and "Link"
share zero words.

We embed every document chunk at ingest time and store those vectors in Qdrant. When a user
asks a question, we embed the question the same way and search for the closest vectors.

**Why cosine similarity and not Euclidean distance?**

Cosine similarity measures the *angle* between two vectors, not the distance between their
endpoints. This makes it independent of vector magnitude — a long document and a short one
about the same topic will have similar angles even if their raw vectors are very different sizes.

---

## What is a sparse vector?

A dense vector (the embedding above) has a value at every position — 768 floats, all non-zero.
A sparse vector has values at only a handful of positions — most are zero.

Think of it like this: a document has thousands of possible words. A sparse vector has one slot
per word. Most slots are 0 (word not present). The slots where a word *does* appear get a
non-zero weight.

Example — a chunk that says "useRouter returns the current router object":

```
sparse vector: {
  "userouter" → position 48291: 0.25
  "returns"   → position 12044: 0.25
  "current"   → position 7831:  0.25
  "router"    → position 39102: 0.25
  "object"    → position 55310: 0.25
  (everything else is 0)
}
```

When you search with the query "useRouter", its sparse vector has a non-zero value at position
48291. The dot product between query and document is non-zero only where both have values in
the same positions — i.e. where they share words. This is keyword matching, implemented as
vector math.

---

## What is TF? What is IDF? What is BM25?

These three terms are the building blocks of classical keyword search — the same ideas that
power Google's earliest ranking algorithm and modern search engines.

### TF — Term Frequency

TF answers: "how important is this word *within this document*?"

```
TF(word, document) = count of word in document / total words in document
```

Example: the chunk "useRouter returns the router. Call useRouter to get the router." has 8
words. "useRouter" appears 2 times → TF = 2/8 = 0.25. "router" appears 2 times → TF = 0.25.
"the" appears 1 time → TF = 0.125.

This is exactly what our sparse encoder computes. It is the simplest possible weight.

**The problem with TF alone:** the word "the" scores similarly to "useRouter". Both might
appear the same number of times. But "the" appears in *every* document, so it tells you
nothing about which document to pick. "useRouter" is rare — finding it is meaningful.

### IDF — Inverse Document Frequency

IDF answers: "how rare is this word *across all documents*?"

```
IDF(word) = log(total documents / documents containing word)
```

If "useRouter" appears in only 3 out of 500 documents:
  IDF = log(500 / 3) = 5.1  ← high, this word is rare and informative

If "component" appears in 400 out of 500 documents:
  IDF = log(500 / 400) = 0.22  ← low, this word is everywhere and says little

### BM25 — the combination

BM25 = TF × IDF, with some extra length normalization thrown in.

```
BM25(word, document) ≈ TF × IDF × length_normalization
```

It gives high scores to words that appear often *in this document* (TF) but rarely *across
all documents* (IDF). That combination perfectly captures "this document is specifically about
useRouter".

**Why we use TF and not BM25:**

BM25 requires IDF, and IDF requires knowing how many documents contain each word — across the
entire corpus. You cannot compute IDF for a single document in isolation.

This conflicts with incremental ingestion. When one file changes, we re-ingest only that file.
But computing BM25 correctly would require recomputing IDF for the entire corpus, which means
re-processing every document. You either give up incremental ingestion or you give up correct
IDF. We kept incremental ingestion and accepted the simpler TF-only approach.

---

## How the sparse encoder works — step by step

Here is what happens when `encodeSparse("useRouter returns the current router object")` runs:

**Step 1: Tokenize**

Split on whitespace and punctuation, lowercase everything, strip short words and stopwords:

```
input:  "useRouter returns the current router object"
tokens: ["userouter", "returns", "current", "router", "object"]
        ("the" is a stopword → removed)
```

**Step 2: Count term frequencies**

```
userouter → 1 occurrence
returns   → 1 occurrence
current   → 1 occurrence
router    → 1 occurrence
object    → 1 occurrence
total tokens: 5
```

**Step 3: Normalize to TF**

Divide each count by total tokens:

```
userouter → 1/5 = 0.20
returns   → 1/5 = 0.20
current   → 1/5 = 0.20
router    → 1/5 = 0.20
object    → 1/5 = 0.20
```

**Step 4: Hash each term to an integer index**

Qdrant sparse vectors use integer indices, not string keys. We cannot store "userouter" as-is.
We run a DJB2 hash on each term and map it into a space of 500,000 integers:

```
"userouter" → hash → 48291
"returns"   → hash → 12044
"current"   → hash → 7831
"router"    → hash → 39102
"object"    → hash → 55310
```

The same word always hashes to the same integer, so query vectors and document vectors share
the same index space. When both contain "userouter", both have a non-zero value at index 48291
— the dot product picks this up.

**Step 5: Output the sparse vector**

```ts
{ indices: [48291, 12044, 7831, 39102, 55310],
  values:  [0.20,  0.20,  0.20, 0.20,  0.20] }
```

This is stored in Qdrant alongside the dense embedding. At query time, the user's question goes
through the exact same steps and produces its own sparse vector. Qdrant computes the dot product
between the two sparse vectors — non-zero only where they share words — and that dot product is
the sparse search score.

---

## Do we write the sparse encoder manually? Are there libraries?

Yes, we wrote it manually. The whole encoder is in `lib/rag/sparse-encoder.ts` and is about
50 lines of actual logic.

There are libraries, but they have tradeoffs:

| Option | Quality | Works in TypeScript? | Notes |
|---|---|---|---|
| **Our TF encoder** (this project) | Basic | Yes | Simplest, no deps, incremental-friendly |
| **BM25 libs** (`wink-bm25-text-search`, `okapibm25`) | Good | Yes | Need full corpus for IDF — conflicts with incremental ingest |
| **SPLADE** (via Qdrant FastEmbed) | Best | No (Python only) | Neural model, learns which terms matter — what Qdrant recommends for prod |

SPLADE produces sparse vectors that are far better than any hand-crafted formula. It is a
neural model trained to assign weights to terms, not just count them. But FastEmbed is
Python-only, which would mean running a separate Python sidecar service. For a TypeScript
project that runs fully locally, that is too much complexity for a learning project.

The manual TF encoder is the right call here: simple, zero dependencies, and sufficient to
demonstrate how hybrid search works.

---

## What is hybrid search? Why do you need both dense and sparse?

Hybrid search runs a dense (semantic) search and a sparse (keyword) search simultaneously and
combines the results.

Each approach has a fundamental blind spot:

**Dense search alone fails on exact terms.**

Query: "What does `useRouter` return?"

The embedding model might surface chunks about "how routing works in Next.js" because they are
semantically similar to the query — they are both about routing. But the chunk that literally
documents the `useRouter` API might rank lower because the embedding captures meaning, not
exact words.

**Sparse search alone fails on meaning.**

Query: "How do I move between pages in my app?"

The word "navigate" or "move" might not appear in any chunk. The relevant chunk talks about
`<Link href="/about">` — technically about navigation, but uses completely different vocabulary.
A keyword search finds nothing. A semantic search finds it easily.

**Together:**

- Dense finds things that *mean* what you asked, even in different words.
- Sparse finds things that *literally contain* what you typed.
- The combined result covers both cases.

This matters a lot for technical documentation, which mixes human-language explanations with
exact API names, function signatures, and config keys.

---

## What is RRF? Why not just average the two scores?

**RRF** stands for Reciprocal Rank Fusion. It is the algorithm that merges the two ranked
result lists into one.

The naive idea would be: get a dense score and a sparse score for each document, average them,
sort. The problem is that dense scores and sparse scores live on completely different scales.

Dense scores are cosine similarities: they range from 0 to 1.
Sparse scores are dot products of TF weights: they range from 0 to some unbounded positive number,
depending on how many shared terms there are and how frequent they are.

If you average `0.82` (dense) and `4.71` (sparse), the sparse score dominates completely. The
dense score barely matters. You could normalize both to [0, 1] first, but normalization requires
knowing the maximum possible score in advance — which you do not.

**RRF works on ranks instead of scores.**

After dense search returns its ranked list and sparse search returns its ranked list, RRF
assigns each document a combined score based purely on its position in each list:

```
rrf_score = 1 / (rank_in_dense_list + 60)
          + 1 / (rank_in_sparse_list + 60)
```

The constant 60 is a dampening factor. Without it, rank #1 would get score 1.0 and rank #2
would get 0.5 — a huge cliff. With 60, rank #1 gets 1/61 ≈ 0.016 and rank #2 gets 1/62 ≈
0.016 — much more gradual, so documents that are consistently good across both lists beat
documents that are exceptional in one but absent in the other.

**Worked example:**

| Document | Dense rank | Sparse rank | RRF score |
|---|---|---|---|
| Chunk A | #1 | #3 | 1/61 + 1/63 = 0.0321 |
| Chunk B | #2 | #1 | 1/62 + 1/61 = 0.0323 |
| Chunk C | #1 | not in list | 1/61 + 0 = 0.0164 |

Chunk B wins even though it was not #1 in either list — because it was near the top of both.
Chunk C was #1 in dense but invisible to sparse, so it loses to documents that showed up in
both lists.

This is exactly the behavior you want: reward documents that are relevant from multiple angles.

**Why prefetch 20 from each search instead of just 5?**

If dense search only returns 5 results, a relevant document ranked #6 in dense is invisible —
even if it was #1 in sparse. By fetching 20 candidates from each search, RRF has a large enough
pool to surface documents that are good in *both* lists. The final `limit: 5` cuts down to what
the caller actually needs.

---

## Why Qdrant? Why not FAISS, Chroma, Pinecone, Weaviate?

**FAISS** is a library, not a database. It keeps everything in memory and on a flat file. No
metadata filtering, no hybrid search, no persistence beyond manually serializing arrays. Fine
for research, not for a real pipeline.

**Chroma** is easy to set up but designed for prototypes. It does not support named vectors or
native hybrid search. Switching to hybrid search later means rewriting the store layer.

**Pinecone** is managed cloud-only — requires an account, API key, and costs money at scale.
Not suitable for a project that should run locally with no external dependencies.

**Weaviate** is a strong production option but heavier to operate and its TypeScript client is
less mature.

**Qdrant** wins here because:
- Runs locally with one Docker command: `docker run -p 6333:6333 qdrant/qdrant`
- Named vectors natively — one stored point holds a dense and a sparse vector side by side
- Hybrid search + RRF is a first-class feature, not a workaround
- HNSW index means search is O(log n), not O(n)
- Clean REST API with a well-typed TypeScript client

---

## Why structural chunking? What was wrong with character chunking?

The v1 approach split documents every 1500 characters. The problem is that chunk boundaries
carry no meaning — a chunk might start mid-sentence and end mid-paragraph.

Worse: adding a paragraph anywhere in a document shifts every chunk boundary after that point.
If chunk IDs are positional (`source-0`, `source-1`...), adding 200 characters to page 1 makes
every chunk on page 2 onwards a slightly different mix of content — but the IDs are the same.
The store is silently corrupted: wrong embeddings stored under correct-looking IDs.

Structural chunking splits on Markdown headings first (`##`, `###`). Each section is an
independent unit. If a section is too large for one chunk, it is then sub-split by character
count.

Benefits:
- Editing section 3 does not affect sections 4, 5, 6 at all — their content and IDs are stable
- Each chunk corresponds to a meaningful piece of documentation
- The full heading path (`App Router > Layouts > Nested Layouts`) is stored in metadata and
  surfaced in citations — the user knows exactly where the answer came from

---

## Why UUID v5 for chunk IDs? Why not random UUIDs?

UUID v5 is deterministic: given the same inputs, it always produces the same UUID. We derive it
from `namespace + source_path + content`.

Re-running ingest on an unchanged file produces the same UUIDs → Qdrant's upsert is a no-op.
Change the content → new UUID → the old chunk is an orphan, cleaned up by `deleteChunksBySource`
before the next upsert.

Random UUIDs would create a new point on every ingest, duplicating every chunk in the store
indefinitely.

Source is included in the ID so that two different files with identical content (e.g. both say
"See the official docs") get different IDs, preserving their separate metadata.

---

## Why a file-hash cache for incremental ingestion?

Embedding is the most expensive step — an API call per batch. Re-embedding 500 unchanged files
to pick up 1 changed one is wasteful.

The cache (`data/ingest-cache.json`) stores an MD5 hash of each file's content. On each
ingest run, if the hash has not changed, the file is skipped entirely — no chunking, no API
calls, no writes to Qdrant.

If the hash changed:
1. Delete all existing chunks for that source from Qdrant
2. Re-chunk, re-embed, upsert fresh chunks

Deleting before upserting is critical. If a section is removed from the file, its chunk gets a
new content hash → new UUID → the old UUID becomes an orphan that would be cited forever.
Deleting by source first guarantees a clean slate.

---

## File-level cache vs section-level cache — what is the tradeoff?

The current cache operates at **file granularity**: one hash per file. If any byte in the file
changes, all chunks for that file are deleted and re-embedded.

This is fine when files are small. But for large files — long API reference pages, generated
docs — a one-paragraph edit triggers re-embedding of hundreds of sections.

**Section-level caching** pushes the granularity down to the section (heading boundary):

```
File-level cache:
  { "routing/intro.md": "a3f9c1..." }
  → one byte changes → re-embed all 50 sections

Section-level cache:
  {
    "routing/intro.md": {
      "App Router > Introduction":    { hash: "a3f...", chunkIds: ["uuid1"] },
      "App Router > Getting Started": { hash: "b72...", chunkIds: ["uuid2", "uuid3"] },
      ...
    }
  }
  → one section changes → re-embed that section only, skip the other 49
```

The ingest flow becomes:
1. Re-split the file into sections
2. Hash each section independently
3. Skip sections whose hash matches the cache
4. For changed/new sections: embed and upsert only those chunks
5. For removed sections (in cache but not in new split): delete only those chunk IDs

**Why this is correct and not just an optimisation:**

Because chunk IDs are content-addressed (UUID v5 of source + content), an unchanged section
produces identical chunk IDs on every run. The Qdrant upsert would be a no-op for those IDs
anyway. Section-level caching just avoids the embedding API call that would tell us that.

The only reason the current code uses `deleteChunksBySource` (delete everything for the file)
is to handle orphan chunks from removed or merged sections. Section-level tracking knows exactly
which chunk IDs belong to each section, so it can delete precisely and skip the nuclear option.

**When does section-level caching matter?**

- Large files (100+ sections) where most edits touch a small fraction of sections
- Frequent doc updates (running ingest on every commit/deploy)
- Paid embedding APIs where each call costs money

For small doc sets with infrequent updates, file-level is sufficient and simpler.

---

## Why not use LangChain?

LangChain provides abstractions for chains, agents, document loaders, vector store connectors,
and more.

We avoided it deliberately:

1. **Abstraction hides learning.** This project exists to understand how a RAG pipeline works.
   LangChain wraps every step behind interfaces. You end up knowing how to configure LangChain,
   not how retrieval actually works.

2. **Magic breaks in unexpected ways.** When something goes wrong (bad retrieval quality,
   wrong chunk sizes, silent errors), you need to understand the internals anyway. Better to own
   them from the start.

3. **Overkill for this scope.** Our pipeline is four steps: chunk, embed, store, retrieve.
   Writing those directly is less code than configuring LangChain to do the same.

In production at scale, LangChain or LlamaIndex can be a reasonable choice — they shine when
connecting many components quickly. For a learning project, writing from scratch is the point.

---

## Why not use the OpenAI Assistants API or a managed RAG product?

Products like OpenAI Assistants (file search) or Amazon Bedrock Knowledge Bases handle
chunking, embedding, storage, and retrieval for you. They are fine for shipping quickly.

We avoid them for the same reason as LangChain: they hide the implementation. You cannot
observe chunk quality, retrieval scores, or how hybrid search behaves. Understanding those
things is the goal here.

---

## Score threshold — why filter low-scoring results?

Qdrant always returns `topK` results, even when none of them are relevant. If the user asks
something that has no answer in the documentation, the search still returns the "least-bad"
chunks. An LLM given irrelevant context will hallucinate — it will construct a plausible-sounding
answer from the noise.

Filtering results below a minimum score gives the system a way to say "nothing is relevant" and
return an honest "I couldn't find that in the documentation" instead.

**One important nuance with RRF scores:** RRF scores are not cosine similarities. They are
rank-derived and typically fall in a much smaller range (0.01–0.05) rather than 0 to 1. So a
threshold of 0.30 (which would be sensible for cosine similarity) would reject everything. The
threshold must be calibrated against your actual score distributions by running representative
queries and observing the numbers.

The threshold value is empirical. You set it by running a set of known-irrelevant queries,
looking at what scores they produce, and picking a cutoff that rejects them reliably without
cutting off legitimate low-confidence results. There is no universal right answer — it depends
on your embedding model, your sparse encoder, and your documents.

---

## Conversation history — how multi-turn Q&A works

By default, each question is independent. The LLM sees only the retrieved context and the
current question. A follow-up like "how do I use that with TypeScript?" has no "that" to refer
to — the LLM has no memory of what was discussed.

The fix is to pass the prior conversation turns to the LLM alongside the new question.

**How it is structured in the prompt:**

```
system:    You are a helpful assistant...
user:      [turn 1 question]
assistant: [turn 1 answer]
user:      [turn 2 question]
assistant: [turn 2 answer]
user:      Context from the documentation:
           {retrieved chunks}
           ---
           Question: {current question}
```

The history comes *before* the retrieved context so the LLM first understands what has already
been discussed, then reads the new evidence, then answers.

**Why cap history at N turns?**

Every turn you add to the prompt costs tokens — which costs money (on paid APIs) and increases
latency. More importantly, LLMs have a context window limit. A long conversation would
eventually exceed it. We cap at the last 3 turns (6 messages: 3 user + 3 assistant), which is
enough for natural follow-up questions without bloating the prompt.

**Why not summarize old turns instead of truncating?**

Summarization is better but requires an extra LLM call before every query. For a learning
project the truncation approach is the right tradeoff. In production you would summarize old
turns progressively so early context is not lost entirely.

**What history is sent from the frontend?**

The frontend strips the `sources` field (which is UI-only metadata) before sending, keeping
only `role` and `content` per message. The server receives a clean `Message[]` that matches
the LLM's expected format directly.
