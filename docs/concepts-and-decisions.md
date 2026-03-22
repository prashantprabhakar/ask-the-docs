# Concepts and Design Decisions

This document explains the concepts behind the RAG pipeline in this project and the reasoning
behind every major decision. If you are reading the code and wondering "why did they do it this
way?", the answer is probably here.

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

## Why Qdrant? Why not FAISS, Chroma, Pinecone, Weaviate?

**FAISS** is a library, not a database. It keeps everything in memory and on a flat file. It
has no concept of filtering by metadata, no hybrid search, and no persistence story beyond
manually serializing numpy arrays. Fine for research notebooks, not for a real pipeline.

**Chroma** is easy to set up but is designed as an embedded store for prototypes. It does not
support named vectors or native hybrid search. Switching to hybrid search later means rewriting
the store layer.

**Pinecone** is managed cloud-only. It requires an account, an API key, and costs money at any
meaningful scale. Not suitable for a project that should run locally with no external
dependencies.

**Weaviate** is a strong production option, but heavier to operate than Qdrant and its
TypeScript client is less mature.

**Qdrant** wins here because:
- Runs locally with one Docker command: `docker run -p 6333:6333 qdrant/qdrant`
- Supports named vectors natively — one point can hold a dense and a sparse vector
- Hybrid search (dense + sparse + RRF fusion) is a first-class feature, not a workaround
- HNSW index means search is O(log n), not O(n)
- The REST client is simple and the API is well-documented

---

## What is a vector embedding?

An embedding is a list of floating-point numbers (a vector) that represents the meaning of a
piece of text. Texts with similar meaning have vectors that are close together in vector space.

Example: "How do I navigate between pages?" and "What is the Link component used for?" have
very different words but similar meaning — their embeddings will be close. "What is the weather
today?" will be far away.

The embedding model (e.g. `nomic-embed-text` via Ollama, or `text-embedding-3-small` via
OpenAI) is what converts text to these vectors. We embed both the documents (at ingest time)
and the user's question (at query time), then find the documents closest to the question.

**Why cosine similarity?** We measure closeness with cosine similarity — the angle between two
vectors, not their distance. This makes the score independent of vector magnitude, which matters
because document chunks vary in length.

---

## What is a sparse vector? What is TF?

A dense vector has a value at every position (e.g. 768 floats, all non-zero). A sparse vector
has values at only a small number of positions — most are zero.

In our sparse encoding, each unique word in a document maps to a position (an integer index),
and the value at that position is the term's weight. Documents with many shared keywords have
overlapping non-zero positions and score higher.

**TF** stands for Term Frequency. It is the simplest possible weight: how often does this word
appear in this chunk, divided by the total number of words. A word that appears 10 times in a
20-word chunk gets TF = 0.5.

**Why TF and not BM25?**

BM25 is better. It adds two improvements over TF:

1. **IDF (Inverse Document Frequency)** — common words across all documents (like "the" or
   "returns") get down-weighted. Rare words that appear in only a few documents get
   up-weighted. This is powerful: the word "useRouter" in a Next.js doc is much more
   informative than the word "component".

2. **Length normalization** — a word appearing 3 times in a 10-word chunk is more significant
   than the same word appearing 3 times in a 1000-word chunk. BM25 adjusts for this.

BM25 is skipped here because it requires IDF, and IDF requires knowing the frequency of every
term across the entire corpus. That conflicts with incremental ingestion — you would need to
recompute IDF every time a document changes, which requires either a two-pass ingest or a
running IDF table stored somewhere.

TF-only is "good enough for a learning project" and still improves retrieval over dense-only
search for exact keyword matches.

**Why not SPLADE?**

SPLADE is the state of the art for sparse vectors. It is a neural model that learns which terms
are important, producing sparse vectors that are far better than any hand-crafted TF or BM25
formula. Qdrant recommends SPLADE via their FastEmbed library for production.

SPLADE is skipped here because it requires a Python model (FastEmbed is Python-only), which
adds a service dependency. For a TypeScript project that runs fully locally, a hand-coded TF
encoder is the right tradeoff.

**How does the term-to-index mapping work?**

We cannot store the actual word strings in Qdrant's sparse vector — it only accepts integer
indices. So we hash each word using DJB2 into a space of 500,000 integers. The same word
always maps to the same integer, so query and document vectors share the same "vocabulary
space".

Hash collisions (two different words mapping to the same integer) are rare at this vocab size
and do not cause incorrect results — just slightly inflated scores for those two terms.

---

## What is hybrid search?

Hybrid search combines two retrieval strategies in a single query:

- **Dense retrieval** — find chunks semantically similar to the question (via embeddings)
- **Sparse retrieval** — find chunks that share exact keywords with the question (via TF/BM25)

Each strategy has blind spots:

- Dense alone: "What does `useRouter` return?" might surface chunks about routing in general,
  missing the chunk that literally explains `useRouter`, because semantically similar content
  may rank above it.
- Sparse alone: "How do I navigate between pages?" might miss the `Link` component docs because
  the word "navigate" doesn't appear — only "link" and "href" do.

Together they cover each other's gaps.

---

## What is RRF? Why not weighted score averaging?

**RRF** stands for Reciprocal Rank Fusion. It is the algorithm Qdrant uses to merge the two
ranked result lists from dense and sparse search.

Instead of combining scores directly, RRF works on ranks:

```
rrf_score(document) = 1 / (rank_in_dense_list + 60)
                    + 1 / (rank_in_sparse_list + 60)
```

The constant 60 dampens the impact of very high ranks — it ensures a document ranked #1 in
one list does not completely dominate a document ranked #2 in both lists.

**Why not just average the scores?**

Dense scores (cosine similarity, 0 to 1) and sparse scores (dot product of TF weights, 0 to N)
live on completely different scales. Averaging them directly is meaningless — the sparse score
would dominate simply because it can be larger. You would need to normalize both first, which
requires knowing the score distribution — a chicken-and-egg problem.

RRF avoids all of this. Ranks are always comparable: rank #3 in one list means the same thing
as rank #3 in another list, regardless of the raw scores.

**Why prefetch 20 from each instead of just fetching 5?**

If we only asked each search for 5 results, a relevant document might be ranked #6 in dense
and #4 in sparse — it would never appear in the fusion. By prefetching 20 from each, the
fusion has a large enough candidate pool to find documents that are good in both lists, even if
neither list has them in the very top results.

---

## Why structural chunking? What was wrong with character chunking?

The v1 approach split documents every 1500 characters. The problem is that chunk boundaries
have no meaning — a chunk might start mid-sentence and end mid-paragraph. Worse, adding a
paragraph anywhere in a document shifts every chunk boundary after it. If chunk IDs are
positional (`source-0`, `source-1`...), they silently point to wrong content.

Structural chunking splits on Markdown headings first (`##`, `###`). Each section becomes an
independent unit. If a section is too large, it is then sub-split by character count.

Benefits:
- Each chunk corresponds to a meaningful section of the docs
- Editing one section does not affect any other section's chunks
- The heading path (`App Router > Layouts > Nested Layouts`) is stored in metadata and shown
  in citations, so the user knows exactly where the answer came from

---

## Why UUID v5 for chunk IDs? Why not random UUIDs?

UUID v5 is a deterministic UUID: given the same inputs, it always produces the same UUID.
We generate it from `namespace + source_path + content`.

This means re-running ingest on an unchanged file produces the same UUIDs. Qdrant's `upsert`
is idempotent — it overwrites a point if the ID already exists, or creates it if not. So
re-ingesting unchanged content is a no-op at the database level.

Random UUIDs would create a new point on every ingest, duplicating every chunk in the store.

**Why include source in the ID?**

If two different files have an identical section (e.g. both say "See the official docs"), they
should have different IDs because their metadata (source, title) differs. Including the source
path in the ID ensures this.

---

## Why a file-hash cache for incremental ingestion?

Embedding is the most expensive step — it requires an API call (or local model inference) per
batch of chunks. Re-embedding unchanged files on every ingest run wastes time and money.

The cache (`data/ingest-cache.json`) stores a hash of each file's content. On each ingest run,
if the hash has not changed, the file is skipped entirely. If it has changed:

1. Delete all existing chunks for that source from Qdrant
2. Re-chunk, re-embed, and upsert the new chunks

Deleting before upserting is critical. If a section is removed, its old chunk has a new content
hash → new UUID → the old UUID becomes an orphan in Qdrant, never cleaned up, cited forever.
Deleting by source first guarantees a clean slate.

---

## Why not use LangChain?

LangChain is a framework that provides abstractions for chains, agents, memory, document
loaders, vector store connectors, and more.

We avoided it here deliberately:

1. **Abstraction hides learning** — the whole point of this project is to understand what a
   RAG pipeline actually does. LangChain wraps every step behind interfaces. You end up
   knowing how to call LangChain, not how RAG works.

2. **Magic breaks in unexpected ways** — LangChain's abstractions leak. When something goes
   wrong (wrong chunk sizes, bad retrieval scores, silent errors), you need to understand the
   internals anyway. Better to own them from the start.

3. **Overkill for this scope** — LangChain's value is in connecting many components quickly.
   Our pipeline has four steps: chunk, embed, store, retrieve. Writing those directly is less
   code than configuring LangChain to do the same thing.

In production at scale, LangChain (or LlamaIndex) can be a reasonable choice. Here, writing
it from scratch is the point.

---

## Why not use the OpenAI Assistants API or a managed RAG product?

Products like OpenAI Assistants (with file search) or Amazon Bedrock Knowledge Bases handle
chunking, embedding, storage, and retrieval for you. They are fine for shipping quickly.

We avoid them here for the same reason as LangChain: they hide the implementation. Understanding
hybrid search, RRF, and chunk quality requires seeing them directly.

---

## Score threshold — why filter low-scoring results?

Qdrant always returns `topK` results. If no chunk is relevant to the question, it still returns
the least-bad chunks. An LLM given irrelevant context will hallucinate — it will try to answer
using the context even when the context has nothing to do with the question.

Filtering results below a minimum score (e.g. 0.30) gives the retriever a way to say "nothing
is relevant" and return an honest "I couldn't find that in the documentation" instead of a
plausible-sounding wrong answer.

The threshold value is empirical — you tune it by looking at score distributions across a set
of representative queries.
