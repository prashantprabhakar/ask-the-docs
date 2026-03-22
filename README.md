# Ask the Docs

A learning project to understand **Retrieval-Augmented Generation (RAG)** from scratch — no vector DB servers, no frameworks hiding the details. Ask questions about Next.js documentation and get streamed, cited answers.

> Built with Next.js 16, Ollama (local), and a hand-rolled file-based vector store.

---

## What it does

Type a question like *"How does the App Router handle layouts?"* and the app:

1. Embeds your question into a vector using a local embedding model
2. Finds the most semantically similar chunks from the Next.js docs
3. Sends those chunks as context to an LLM with a grounded prompt
4. Streams the answer back token by token with source citations

No hallucinations from training data — the LLM only answers from what the docs actually say.

---

## How RAG works here

RAG has two distinct phases:

### Phase 1 — Ingest (run once, offline)

```
docs/*.md → chunk → embed → vector-store.json
```

**Step 1: Load** — [`lib/rag/ingester.ts`](lib/rag/ingester.ts) walks `data/docs/` and reads every `.md` / `.mdx` file. The first `#` heading becomes the document title used in citations.

**Step 2: Chunk** — [`lib/rag/chunker.ts`](lib/rag/chunker.ts) splits each document using `RecursiveCharacterTextSplitter` (1500 chars, 200 char overlap). The overlap is the key insight: if an answer straddles a chunk boundary, at least one chunk will contain the full relevant passage.

**Step 3: Embed** — Each chunk is passed to an embedding model (Ollama's `nomic-embed-text` or OpenAI's `text-embedding-3-small`). This converts text into a high-dimensional vector that encodes semantic meaning.

**Step 4: Store** — [`lib/vectordb/vector-store.ts`](lib/vectordb/vector-store.ts) persists chunks + embeddings to `data/vector-store.json`. Chunks get deterministic MD5 IDs so re-running ingest is safe (upsert, not duplicate).

### Phase 2 — Query (on every user question)

```
question → embed → similarity search → prompt → LLM → stream
```

**Retrieve** — The question is embedded with the same model used during ingest (same semantic space = comparable vectors). Cosine similarity is computed against every stored chunk. Only chunks scoring above 0.6 are considered.

**Augment** — The top-5 chunks are injected into a system prompt: *"Answer ONLY using the provided context."* This is the "augmented" part of RAG — giving the LLM private knowledge it wasn't trained on.

**Generate** — [`lib/rag/retriever.ts`](lib/rag/retriever.ts) sends the prompt to the LLM and returns an async token stream. The API route ([`app/api/chat/route.ts`](app/api/chat/route.ts)) forwards this as Server-Sent Events (SSE) to the browser, which renders tokens as they arrive.

### The vector math

Cosine similarity measures the angle between two vectors in embedding space:

```
similarity = dot(A, B) / (|A| × |B|)
```

Score near **1.0** → semantically identical. Score near **0.0** → unrelated. This works because the embedding model places semantically similar text in nearby directions regardless of exact wording.

---

## Project structure

```
app/
  page.tsx              # Chat UI (streaming SSE consumer)
  api/chat/route.ts     # POST /api/chat — SSE endpoint

lib/
  rag/
    ingester.ts         # Load docs, orchestrate ingest pipeline
    chunker.ts          # Split docs into overlapping chunks
    retriever.ts        # Embed query, search, build prompt, stream
  vectordb/
    vector-store.ts     # File-based store with cosine similarity
  llm/
    factory.ts          # Create LLM/embedding clients from env vars
    ollama.client.ts    # Ollama (local) implementation
    openai.client.ts    # OpenAI / GitHub Models implementation
    types.ts            # ILLMClient / IEmbeddingClient interfaces

scripts/
  fetch-docs.ts         # CLI entry point: npm run fetch-docs
  ingest.ts             # CLI entry point: npm run ingest

data/
  docs/                 # Generated — Next.js docs fetched from GitHub (gitignored)
  ingest-cache.json     # Generated — file hashes for incremental ingest (gitignored)
```

---

## Setup

### Prerequisites

- [Node.js](https://nodejs.org) 18+
- [Ollama](https://ollama.com) running locally (default), **or** an OpenAI / GitHub Models API key

### 1. Install dependencies

```bash
npm install
```

### 2. Configure environment

```bash
cp .env.example .env
```

Default config uses Ollama. Pull the required models:

```bash
ollama pull llama3.2          # LLM
ollama pull nomic-embed-text  # Embeddings
```

To use OpenAI instead, set in `.env`:

```
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

### 3. Fetch the docs

```bash
npm run fetch-docs
```

Downloads the Next.js documentation from the official GitHub repository into `data/docs/`. This must run before ingest. Re-run it whenever you want to pull the latest docs.

### 4. Ingest the docs

```bash
npm run ingest
```

Chunks, embeds, and stores everything in Qdrant. Only needs to run once (or after fetching updated docs).

To force a full re-embed from scratch (e.g. after switching embedding models):

```bash
npm run ingest:full
```

### 5. Run the app

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

---

## What could be improved

This project intentionally keeps things simple for learning. Here's what a production RAG system would add:

| Area | Current | Production alternative |
|------|---------|----------------------|
| **Vector store** | Flat JSON file, brute-force cosine scan | Chroma, Pgvector, Pinecone, Weaviate — indexed with HNSW for sub-millisecond search at millions of vectors |
| **Retrieval quality** | Embedding similarity only | Hybrid search: BM25 (keyword) + semantic, then re-rank with a cross-encoder |
| **Chunking** | Fixed character size with overlap | Semantic chunking (split on meaning, not length), or document-aware parsing that respects headings/code blocks |
| **Query understanding** | Raw question embedded as-is | HyDE (embed a hypothetical answer instead of the question), query expansion, or multi-query retrieval |
| **Context window** | Top-5 chunks concatenated | Re-ranking to fit more relevant content, or map-reduce for large doc sets |
| **Conversation** | Single-turn only | Multi-turn with history compression, so follow-up questions retain context |
| **Evaluation** | None | RAGAS, TruLens, or custom metrics (faithfulness, answer relevance, context precision) |
| **Freshness** | Manual re-ingest | Incremental updates tracked by file hash, webhook-triggered re-ingestion |

---

## Key concepts to take away

- **Embedding** turns text into a point in high-dimensional space. Nearby points = similar meaning.
- **Chunking with overlap** ensures answers that straddle boundaries are still retrievable.
- **The 0.6 score threshold** filters irrelevant chunks before they pollute the prompt — tune this for your domain.
- **Grounded prompting** ("answer ONLY from context") is what prevents hallucination; without it, the LLM falls back on training data.
- **SSE streaming** lets the UI feel responsive while the LLM generates — the sources are sent first as a separate event, then tokens follow.
