import { createEmbeddingClient, createLLMClient } from '../llm/factory'
import { similaritySearch } from '../vectordb'
import type { ChunkFilter } from '../vectordb'
import { encodeSparse } from './sparse-encoder'
import { rerank } from './reranker'
import { expandQuery } from './query-expander'
import { logger } from '../logger'
import type { Message } from '../llm/types'
import { retrieval as retrievalConfig } from '../config'

const embedder = createEmbeddingClient()
const llm = createLLMClient()

const MIN_SCORE = retrievalConfig.minScore
const RETRIEVAL_CANDIDATES = retrievalConfig.candidates
const TOP_K = retrievalConfig.topK
const HISTORY_TURNS = retrievalConfig.historyTurns

export interface RetrievedSource {
  title: string
  sectionTitle: string
  source: string
  url: string
  score: number
  excerpt: string
}

export interface RAGResponse {
  answer: string
  sources: RetrievedSource[]
}

/**
 * Build the prompt that gets sent to the LLM.
 *
 * Structure:
 *   system: instructions
 *   user/assistant: prior conversation turns (last HISTORY_TURNS pairs)
 *   user: retrieved context + current question
 *
 * History comes before the context so the LLM understands what has already
 * been asked and answered before it reads the new evidence.
 */
function buildPrompt(question: string, contextChunks: string[], history: Message[], summary?: string): Message[] {
  const context = contextChunks.join('\n\n---\n\n')

  /**
   * Append the conversation summary to the system message when present.
   * Keeping it in the system message (rather than as a fake user/assistant
   * turn) avoids polluting the conversation structure and is always visible
   * to the model regardless of how many turns follow.
   */
  const systemContent = [
    `You are a helpful assistant that answers questions about technical documentation.
Answer ONLY using the provided context. If the context doesn't contain enough information to answer, say so clearly.
Do not make up information. Be concise and precise.`,
    summary ? `Earlier in this conversation: ${summary}` : '',
  ].filter(Boolean).join('\n\n')

  return [
    { role: 'system', content: systemContent },
    // Inject prior turns so follow-up questions like "how do I use that with TypeScript?"
    // have the context of what "that" refers to.
    ...history,
    {
      role: 'user',
      content: `Context from the documentation:

${context}

---

Question: ${question}`,
    },
  ]
}

/**
 * Expand the question into multiple query variants, search for each in
 * parallel, and return a deduplicated candidate pool.
 *
 * Deduplication key: chunk content. If the same chunk is returned by multiple
 * query variants, only the first occurrence is kept (scores are discarded
 * before re-ranking anyway — the cross-encoder re-scores everything).
 */
async function retrieveCandidates(question: string, requestId: string, filter?: ChunkFilter) {
  // ── Query expansion ────────────────────────────────────────────────────────
  const expandElapsed = logger.timer()
  const queries = await expandQuery(question)
  logger.info({ requestId, event: 'query_expand', queryCount: queries.length, durationMs: expandElapsed() })

  // ── Embedding ──────────────────────────────────────────────────────────────
  const embedElapsed = logger.timer()
  const embeddings = await embedder.embedBatch(queries)
  logger.info({ requestId, event: 'embed', queryCount: queries.length, durationMs: embedElapsed() })

  // ── Parallel search ────────────────────────────────────────────────────────
  const searchElapsed = logger.timer()
  const allResults = await Promise.all(
    queries.map((q, i) => similaritySearch(embeddings[i], encodeSparse(q), RETRIEVAL_CANDIDATES, filter))
  )
  logger.info({ requestId, event: 'search', queryCount: queries.length, durationMs: searchElapsed() })

  // ── Deduplicate ────────────────────────────────────────────────────────────
  const seen = new Set<string>()
  const merged = allResults.flat().filter((r) => {
    if (seen.has(r.content)) return false
    seen.add(r.content)
    return true
  })

  const candidates = merged.filter((r) => r.score >= MIN_SCORE)
  logger.info({ requestId, event: 'candidates', total: merged.length, aboveThreshold: candidates.length })

  return candidates
}

/**
 * The full RAG query pipeline — called on every user question.
 *
 * Retrieve → Re-rank → Augment → Generate
 */
export async function ragQuery(question: string, history: Message[] = [], summary?: string, filter?: ChunkFilter): Promise<RAGResponse> {
  const requestId = crypto.randomUUID()
  logger.info({ requestId, event: 'request_start', question })

  // Step 1: RETRIEVE — expand query, search variants in parallel, deduplicate
  const candidates = await retrieveCandidates(question, requestId, filter)

  if (candidates.length === 0) {
    logger.info({ requestId, event: 'no_candidates' })
    return {
      answer: "I couldn't find any relevant information in the documentation.",
      sources: [],
    }
  }

  // Step 2: RE-RANK — cross-encoder scores each (question, chunk) pair
  const rerankElapsed = logger.timer()
  const reranked = await rerank(question, candidates, TOP_K)
  logger.info({
    requestId,
    event: 'rerank',
    candidateCount: candidates.length,
    topK: reranked.length,
    scores: reranked.map((r) => r.rerankScore.toFixed(3)),
    durationMs: rerankElapsed(),
  })

  const trimmedHistory = history.slice(-(HISTORY_TURNS * 2))
  const messages = buildPrompt(question, reranked.map((r) => r.content), trimmedHistory, summary)

  // Step 3: GENERATE
  const llmElapsed = logger.timer()
  const answer = await llm.chat(messages)
  logger.info({ requestId, event: 'llm_complete', durationMs: llmElapsed() })

  return {
    answer,
    sources: reranked.map((r) => ({
      title: r.metadata.title,
      sectionTitle: r.metadata.sectionTitle,
      source: r.metadata.source,
      url: r.metadata.url,
      score: r.rerankScore,
      excerpt: r.content.slice(0, retrievalConfig.excerptLength) + '...',
    })),
  }
}

/**
 * Streaming version — yields answer tokens as they arrive.
 * Sources are returned separately after the stream ends.
 *
 * The stream is wrapped to track first-token latency — the time from when
 * the LLM call is made to when the first token arrives over the wire.
 * This is the metric users feel as "response lag".
 */
export async function ragQueryStream(question: string, history: Message[] = [], summary?: string, filter?: ChunkFilter): Promise<{
  stream: AsyncIterable<string>
  sources: RetrievedSource[]
}> {
  const requestId = crypto.randomUUID()
  logger.info({ requestId, event: 'request_start', question })

  const candidates = await retrieveCandidates(question, requestId, filter)

  if (candidates.length === 0) {
    logger.info({ requestId, event: 'no_candidates' })
    return {
      stream: (async function* () { yield "I couldn't find any relevant information in the documentation." })(),
      sources: [],
    }
  }

  const rerankElapsed = logger.timer()
  const reranked = await rerank(question, candidates, TOP_K)
  logger.info({
    requestId,
    event: 'rerank',
    candidateCount: candidates.length,
    topK: reranked.length,
    scores: reranked.map((r) => r.rerankScore.toFixed(3)),
    durationMs: rerankElapsed(),
  })

  const trimmedHistory = history.slice(-(HISTORY_TURNS * 2))
  const messages = buildPrompt(question, reranked.map((r) => r.content), trimmedHistory, summary)

  const llmStart = logger.timer()
  const rawStream = llm.streamChat(messages)

  /**
   * Wrap the raw stream to measure first-token latency.
   * First-token latency = time from LLM call to first byte of response.
   * This is what the user perceives as "time to start reading an answer".
   */
  async function* instrumentedStream(): AsyncIterable<string> {
    let firstToken = true
    for await (const token of rawStream) {
      if (firstToken) {
        logger.info({ requestId, event: 'llm_first_token', durationMs: llmStart() })
        firstToken = false
      }
      yield token
    }
    logger.info({ requestId, event: 'llm_complete', durationMs: llmStart() })
  }

  return {
    stream: instrumentedStream(),
    sources: reranked.map((r) => ({
      title: r.metadata.title,
      sectionTitle: r.metadata.sectionTitle,
      source: r.metadata.source,
      url: r.metadata.url,
      score: r.rerankScore,
      excerpt: r.content.slice(0, retrievalConfig.excerptLength) + '...',
    })),
  }
}
