// ─── LLM ──────────────────────────────────────────────────────────────────────

export const llm = {
  provider: process.env.LLM_PROVIDER ?? 'ollama',
  model: process.env.LLM_MODEL ?? 'llama3.2',
  embeddingProvider: process.env.EMBEDDING_PROVIDER ?? 'ollama',
  embeddingModel: process.env.EMBEDDING_MODEL ?? 'nomic-embed-text',
  embeddingDim: parseInt(process.env.EMBEDDING_DIM ?? '768'),
}

// ─── Ollama ───────────────────────────────────────────────────────────────────

export const ollama = {
  baseURL: process.env.OLLAMA_BASE_URL ?? 'http://localhost:11434',
}

// ─── OpenAI / GitHub Models ───────────────────────────────────────────────────

export const openai = {
  provider: process.env.OPENAI_PROVIDER ?? 'openai',
  apiKey: process.env.OPENAI_API_KEY ?? '',
  githubToken: process.env.GITHUB_TOKEN ?? '',
  githubModel: process.env.GITHUB_MODEL ?? 'gpt-4o-mini',
  githubBaseURL: 'https://models.inference.ai.azure.com',
}

// ─── Qdrant ───────────────────────────────────────────────────────────────────

export const qdrant = {
  url: process.env.QDRANT_URL ?? 'http://localhost:6333',
  collection: 'ask-the-docs',
  denseVector: 'dense',
  sparseVector: 'sparse',
  /** Fixed namespace for deterministic chunk UUIDs — never change this. */
  chunkNamespace: 'b3d2e1f0-4a5b-6c7d-8e9f-0a1b2c3d4e5f',
}

// ─── Chunking ─────────────────────────────────────────────────────────────────

export const chunking = {
  maxChunkSize: 1500,
  chunkOverlap: 150,
}

// ─── Retrieval ────────────────────────────────────────────────────────────────

export const retrieval = {
  /** RRF scores below this are dropped before re-ranking. */
  minScore: 0.01,
  /** Candidates fetched per query variant before cross-encoder re-ranking. */
  candidates: 20,
  /** Final chunks included in the prompt after re-ranking. */
  topK: 5,
  /** Prior conversation turns injected into the prompt. */
  historyTurns: 3,
  /** Characters shown in the source excerpt. */
  excerptLength: 200,
  /** Candidates prefetched per vector space in Qdrant hybrid search. */
  prefetchLimit: 20,
}

// ─── Ingestion ────────────────────────────────────────────────────────────────

export const ingestion = {
  /** Embedding batch size — tune per provider (Ollama: 10, OpenAI API: 100+). */
  batchSize: 10,
  /** Max concurrent LLM calls for context prefix generation per file. */
  contextConcurrency: Number(process.env.CONTEXT_CONCURRENCY ?? 3),
  /** Max files processed in parallel. Ollama: 2–3 (GPU-bound). OpenAI API: 5–10. */
  fileConcurrency: Number(process.env.FILE_CONCURRENCY ?? 3),
}

// ─── Rate limiting ────────────────────────────────────────────────────────────

export const rateLimit = {
  windowMs: 60_000,
  maxRequests: 100,
  maxTrackedIPs: 5_000,
}

// ─── Retry ────────────────────────────────────────────────────────────────────

export const retry = {
  attempts: 3,
  minTimeoutMs: 500,
}

// ─── Conversation ─────────────────────────────────────────────────────────────

export const conversation = {
  /** Recent turns sent verbatim; older turns are compressed into the summary. */
  recentWindow: 3,
  /** Summarization kicks in once history exceeds this many turns. */
  summaryThreshold: 6,
}

// ─── Reranker ─────────────────────────────────────────────────────────────────

export const reranker = {
  model: 'Xenova/ms-marco-MiniLM-L-6-v2',
}
