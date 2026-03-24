# Ask the Docs — V4 Plan: Agentic RAG

## What changes in V4

V3 makes a single-pass RAG pipeline production-grade. V4 changes the fundamental architecture:
the system stops being a fixed retrieve-then-generate pipeline and becomes an **agent** — a
loop where the LLM reasons about what it knows, decides what to look up, acts, observes the
result, and repeats until it has a complete answer.

This unlocks questions that single-pass RAG cannot handle:

> "Compare how App Router and Pages Router handle data fetching, and tell me which one I should
> use for a new project that needs real-time updates."

Single-pass RAG retrieves a few chunks and hopes they cover both routers. An agent retrieves
for App Router, reads the result, realises it needs Pages Router too, retrieves again, compares,
and then answers — with sources from both.

---

## Problem 1: Complex questions need multiple retrieval steps

**Current:** One question → one retrieval → one answer. If the answer requires combining
information from multiple sections or multiple retrieval passes, the system fails or hallucinates.

**V4 fix: Multi-hop retrieval with sub-question decomposition**

Before retrieving, the LLM decomposes the question into atomic sub-questions:

```
User: "How does caching work differently in App Router vs Pages Router?"

Decomposed:
  1. "How does caching work in App Router?"
  2. "How does caching work in Pages Router?"
  3. "What are the key differences?"
```

Each sub-question is retrieved independently. Results are merged, deduplicated, and passed
to the LLM with a synthesis prompt.

Implementation: `lib/rag/decomposer.ts` — takes a question, calls the LLM with a structured
prompt, returns `string[]` of sub-questions.

---

## Problem 2: The system cannot look things up mid-generation

**Current:** Retrieval happens once before generation. The LLM cannot say "wait, I need to
check one more thing" and go look it up.

**V4 fix: ReAct loop (Reason + Act)**

The LLM runs in a loop, alternating between reasoning and tool calls:

```
Thought: I need to find how Server Components handle data fetching.
Action: search("Server Components data fetching")
Observation: [retrieved chunks]
Thought: Now I need the Pages Router equivalent.
Action: search("Pages Router getServerSideProps")
Observation: [retrieved chunks]
Thought: I have enough to answer.
Answer: [final answer with citations]
```

Each iteration is one LLM call. The loop runs until the LLM emits `Answer:` or hits a maximum
iteration limit (default: 5).

Implementation: `lib/agent/react-loop.ts` — orchestrates the Thought/Action/Observation cycle.
Tools are registered functions the LLM can call by name.

---

## Problem 3: No tool ecosystem — retrieval is the only action

**Current:** The system can only retrieve from docs. It cannot filter by section, list available
topics, look up a specific page, or summarise a large chunk of documentation.

**V4 fix: Tool registry**

Define a set of tools the agent can call:

| Tool | Input | Output |
|------|-------|--------|
| `search(query)` | natural language query | top chunks with scores |
| `search_section(query, docType)` | query + filter | chunks filtered by doc type |
| `get_page(source)` | file path | full page content |
| `list_topics()` | none | list of all section titles in the index |
| `summarise(chunks[])` | chunk list | LLM-generated summary |

Tools are plain TypeScript functions. The LLM receives their names and descriptions in the
system prompt and emits structured tool calls (JSON). The agent runner parses them, calls the
function, and feeds the result back as an Observation.

This is the same pattern as OpenAI function calling or Claude tool use — but implemented
manually so the tool layer is not tied to any specific LLM API.

---

## Problem 4: No self-correction — the system never checks its own answer

**Current:** Whatever the LLM generates is returned as-is. If the answer contradicts the
retrieved context (a hallucination), there is no catch.

**V4 fix: Faithfulness verification**

After the LLM generates an answer, run a second LLM call:

```
prompt: "Given this context: {chunks}
         And this answer: {answer}
         Does the answer contradict or go beyond the provided context?
         Reply: FAITHFUL or UNFAITHFUL. If UNFAITHFUL, explain what was wrong."
```

If the answer is unfaithful, the agent loop runs again with an additional instruction:
"Your previous answer was unfaithful to the sources. Correct it using only the provided
context."

This is the core of self-correcting RAG (also called SELF-RAG in the literature).

---

## Problem 5: No routing — every question goes through full RAG

**Current:** Every question hits the full pipeline regardless of whether retrieval helps.
"What is React?" does not need RAG — the LLM knows this. But it still pays the retrieval
cost and gets noisy chunks mixed into the answer.

**V4 fix: Query router**

Before retrieval, classify the question:

```
NEEDS_DOCS    → run full agentic RAG
KNOWN         → answer directly from LLM training (no retrieval)
OUT_OF_SCOPE  → "This question is outside the scope of Next.js documentation."
AMBIGUOUS     → ask a clarifying question before proceeding
```

The router is a small fast LLM call (cheap model like `llama3.2` or a fine-tuned classifier).
This reduces latency for simple questions and avoids polluting the context with irrelevant chunks.

---

## Problem 6: Sparse encoder is still BM25 — not a neural model

**Current (after V3):** BM25 with IDF. Still hand-crafted weights.

**V4 fix: SPLADE-style learned sparse vectors**

SPLADE is a neural model that learns which terms are important across the corpus. It produces
sparse vectors that are far better than BM25 for technical documentation.

The challenge: Qdrant's FastEmbed (the recommended SPLADE integration) is Python-only.

Solutions for a TypeScript project:
1. **Sidecar service**: a small FastAPI Python service that exposes `/encode` → returns sparse
   vector. The TypeScript ingest/query code calls it over HTTP. Simple, decoupled.
2. **ONNX export**: export the SPLADE model to ONNX and run it with `onnxruntime-node`.
   No Python at runtime, just a model file. Harder to set up but fully TypeScript.
3. **Qdrant's built-in inference** (cloud): Qdrant Cloud can run FastEmbed server-side.
   Zero code on our end.

Option 1 (sidecar) is the pragmatic production choice. Option 2 is the right choice for
fully local TypeScript deployment.

---

## Problem 7: gRPC client — REST is too slow for production

**Current:** `@qdrant/js-client-rest` opens a new HTTP connection per request.

**V4 fix:** Switch to `@qdrant/js-client-grpc` which uses a persistent connection.
For high-QPS APIs this matters significantly. The gRPC client uses HTTP/2 multiplexing —
many requests share one connection with no per-request handshake.

---

## Problem 8: Zero-downtime embedding model migration

**Current:** Switching embedding models requires dropping the collection and re-ingesting
everything. During that time the app returns no answers.

**V4 fix: Collection aliases + dual-write migration**

Qdrant supports collection aliases — a pointer that can be atomically swapped.

Migration procedure:
1. Create new collection `ask-the-docs-v2` with the new embedding model
2. Ingest all documents into `ask-the-docs-v2`
3. Atomically swap the alias `ask-the-docs` to point at `ask-the-docs-v2`
4. Drop `ask-the-docs-v1`

The app always queries by alias. The swap is instant. Zero downtime.

---

## Problem 9: No knowledge graph — the system treats all chunks as independent

**Current:** Each chunk is a standalone unit. The system does not know that "App Router" and
"Server Components" are related concepts, or that `useRouter` is a hook in the Pages Router
not the App Router.

**V4 fix: Entity graph for multi-hop reasoning**

At ingest time, extract entities and relationships from each chunk using the LLM:
```
entities: ["App Router", "layout.tsx", "nested layouts"]
relationships: [("App Router", "uses", "layout.tsx"), ("layout.tsx", "enables", "nested layouts")]
```

Store in a lightweight graph (e.g. a JSON adjacency list, or Neo4j for production). During
agentic retrieval, the agent can traverse the graph to find related entities and retrieve
their chunks explicitly, rather than relying purely on embedding similarity.

This enables true multi-hop reasoning: "How does App Router's nested layout system interact
with Server Components?" — the graph connects these concepts explicitly.

---

## V4 Architecture

```
User question
    ↓
Router (classify question)
    ↓
[NEEDS_DOCS]          [KNOWN]        [OUT_OF_SCOPE]   [AMBIGUOUS]
    ↓                   ↓                  ↓               ↓
Decomposer          Direct LLM      "Out of scope"    Clarify
    ↓
Sub-questions[]
    ↓
ReAct Loop
  ┌─ Thought → choose tool
  ├─ search(query)
  ├─ search_section(query, filter)
  ├─ get_page(source)
  ├─ list_topics()
  └─ summarise(chunks)
    ↓
Faithfulness check
    ↓
Answer + citations
```

---

## V4 File Changes

```
lib/
  agent/
    react-loop.ts         ← NEW: Thought/Action/Observation orchestrator
    tools.ts              ← NEW: tool registry + implementations
    router.ts             ← NEW: query classifier
    decomposer.ts         ← NEW: sub-question decomposition
    faithfulness.ts       ← NEW: answer verification

  rag/
    sparse-encoder.ts     ← UPDATE: SPLADE via sidecar or ONNX
    graph-builder.ts      ← NEW: entity extraction + graph construction
    graph-retriever.ts    ← NEW: graph-aware multi-hop retrieval

  vectordb/
    qdrant.client.ts      ← UPDATE: switch to gRPC client, alias support

app/
  api/
    chat/route.ts         ← UPDATE: stream ReAct loop steps to UI
  page.tsx                ← UPDATE: show reasoning steps, tool calls in UI
```

## Implementation Order

### Step 1 — gRPC client + collection aliases
Infrastructure. No user-visible change but required before high-QPS use.

### Step 2 — Query router
Fast win. Reduces latency for simple questions. Independently testable.

### Step 3 — Sub-question decomposition + multi-hop retrieval
The first agentic step. No full loop yet — just decompose → retrieve × N → merge → answer.

### Step 4 — ReAct loop + tool registry
Full agentic loop. Built on top of Step 3.

### Step 5 — Faithfulness verification (SELF-RAG)
Add the self-correction layer on top of the loop.

### Step 6 — SPLADE (sidecar approach)
Ship the Python FastAPI sidecar, switch ingest + query to use it. Full re-ingest required.

### Step 7 — Knowledge graph
Most complex step. Requires entity extraction at ingest (costs LLM calls per chunk) and graph
traversal at query time. Best implemented after the agentic loop is stable.
