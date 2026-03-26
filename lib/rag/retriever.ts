import { createEmbeddingClient, createLLMClient } from '../llm/factory'
import { similaritySearch } from '../vectordb'
import { encodeSparse } from './sparse-encoder'
import { rerank } from './reranker'
import { expandQuery } from './query-expander'
import type { Message } from '../llm/types'

const embedder = createEmbeddingClient()
const llm = createLLMClient()

/**
 * Chunks below this score are considered irrelevant and dropped before building
 * the prompt. Without this, Qdrant always returns topK results even when none
 * are relevant — and the LLM will hallucinate an answer from the noise.
 *
 * PROD NOTE — This value is empirical. Tune it by running known-irrelevant
 *   queries, observing their scores, and setting the threshold just above them.
 *   RRF scores are not cosine similarities — they are rank-derived and typically
 *   much smaller (often 0.01–0.05 range), so the threshold must be calibrated
 *   against your actual score distributions, not against intuition.
 */
const MIN_SCORE = 0.01

/**
 * How many candidates to retrieve from hybrid search before re-ranking.
 * Larger = better recall for the re-ranker to work with, at the cost of more
 * cross-encoder forward passes. 20 is a common production default.
 */
const RETRIEVAL_CANDIDATES = 20

/**
 * How many top-ranked chunks to include in the prompt after re-ranking.
 */
const TOP_K = 5

/**
 * How many prior conversation turns to include in the prompt.
 * Each turn = one user message + one assistant message.
 * More turns = better follow-up handling, larger prompt, more tokens.
 *
 * PROD NOTE — In production you would also summarize old turns rather than
 *   truncating them, so the LLM retains context from early in a long session.
 */
const HISTORY_TURNS = 3

export interface RetrievedSource {
  title: string
  sectionTitle: string
  source: string
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
 *
 * Expansion + parallel search run concurrently with Promise.all:
 *   - expandQuery fires the LLM for variants
 *   - embedBatch encodes all variants in one call
 *   - N similaritySearch calls run in parallel (one per variant)
 */
async function retrieveCandidates(question: string) {
  // Expand the query and embed all variants in a single batch call
  const queries = await expandQuery(question)
  const embeddings = await embedder.embedBatch(queries)

  // Parallel search — one per query variant
  const allResults = await Promise.all(
    queries.map((q, i) => similaritySearch(embeddings[i], encodeSparse(q), RETRIEVAL_CANDIDATES))
  )

  // Deduplicate by content — first-seen wins
  const seen = new Set<string>()
  const merged = allResults.flat().filter((r) => {
    if (seen.has(r.content)) return false
    seen.add(r.content)
    return true
  })

  return merged.filter((r) => r.score >= MIN_SCORE)
}

/**
 * The full RAG query pipeline — called on every user question.
 *
 * Retrieve → Re-rank → Augment → Generate
 */
export async function ragQuery(question: string, history: Message[] = [], summary?: string): Promise<RAGResponse> {
  // Step 1: RETRIEVE — expand query, search variants in parallel, deduplicate
  const candidates = await retrieveCandidates(question)

  if (candidates.length === 0) {
    return {
      answer: "I couldn't find any relevant information in the documentation.",
      sources: [],
    }
  }

  // Step 2: RE-RANK — cross-encoder scores each (question, chunk) pair
  // and re-orders by fine-grained relevance. This improves precision over
  // the bi-encoder scores returned by hybrid search.
  const reranked = await rerank(question, candidates, TOP_K)

  const trimmedHistory = history.slice(-(HISTORY_TURNS * 2))

  // Step 3: AUGMENT — build the prompt with re-ranked context
  const messages = buildPrompt(question, reranked.map((r) => r.content), trimmedHistory, summary)

  // Step 4: GENERATE — send to LLM and get the answer
  const answer = await llm.chat(messages)

  return {
    answer,
    sources: reranked.map((r) => ({
      title: r.metadata.title,
      sectionTitle: r.metadata.sectionTitle,
      source: r.metadata.source,
      score: r.rerankScore,
      excerpt: r.content.slice(0, 200) + '...',
    })),
  }
}

/**
 * Streaming version — yields answer tokens as they arrive.
 * Sources are returned separately after the stream ends.
 */
export async function ragQueryStream(question: string, history: Message[] = [], summary?: string): Promise<{
  stream: AsyncIterable<string>
  sources: RetrievedSource[]
}> {
  const candidates = await retrieveCandidates(question)

  if (candidates.length === 0) {
    return {
      stream: (async function* () { yield "I couldn't find any relevant information in the documentation." })(),
      sources: [],
    }
  }

  const reranked = await rerank(question, candidates, TOP_K)

  const trimmedHistory = history.slice(-(HISTORY_TURNS * 2))
  const messages = buildPrompt(question, reranked.map((r) => r.content), trimmedHistory, summary)

  return {
    stream: llm.streamChat(messages),
    sources: reranked.map((r) => ({
      title: r.metadata.title,
      sectionTitle: r.metadata.sectionTitle,
      source: r.metadata.source,
      score: r.rerankScore,
      excerpt: r.content.slice(0, 200) + '...',
    })),
  }
}
