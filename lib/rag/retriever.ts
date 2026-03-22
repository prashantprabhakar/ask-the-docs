import { createEmbeddingClient, createLLMClient } from '../llm/factory'
import { similaritySearch } from '../vectordb'
import type { Message } from '../llm/types'

const embedder = createEmbeddingClient()
const llm = createLLMClient()

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
 * The [CONTEXT] block is the "augmented" part of Retrieval-Augmented Generation.
 * We're giving the LLM private knowledge it wasn't trained on.
 */
function buildPrompt(question: string, contextChunks: string[]): Message[] {
  const context = contextChunks.join('\n\n---\n\n')

  return [
    {
      role: 'system',
      content: `You are a helpful assistant that answers questions about technical documentation.
Answer ONLY using the provided context. If the context doesn't contain enough information to answer, say so clearly.
Do not make up information. Be concise and precise.`,
    },
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
 * The full RAG query pipeline — called on every user question.
 *
 * Retrieve → Augment → Generate
 */
export async function ragQuery(question: string): Promise<RAGResponse> {
  // Step 1: RETRIEVE — embed the question and search for similar chunks
  const queryEmbedding = await embedder.embed(question)
  const results = await similaritySearch(queryEmbedding, 5)

  if (results.length === 0) {
    return {
      answer: "I couldn't find any relevant information in the documentation.",
      sources: [],
    }
  }

  // Step 2: AUGMENT — build the prompt with retrieved context
  const messages = buildPrompt(
    question,
    results.map((r) => r.content)
  )

  // Step 3: GENERATE — send to LLM and get the answer
  const answer = await llm.chat(messages)

  return {
    answer,
    sources: results.map((r) => ({
      title: r.metadata.title,
      sectionTitle: r.metadata.sectionTitle,
      source: r.metadata.source,
      score: r.score,
      excerpt: r.content.slice(0, 200) + '...',
    })),
  }
}

/**
 * Streaming version — yields answer tokens as they arrive.
 * Sources are returned separately after the stream ends.
 */
export async function ragQueryStream(question: string): Promise<{
  stream: AsyncIterable<string>
  sources: RetrievedSource[]
}> {
  const queryEmbedding = await embedder.embed(question)
  const results = await similaritySearch(queryEmbedding, 5)

  const messages = buildPrompt(
    question,
    results.map((r) => r.content)
  )

  return {
    stream: llm.streamChat(messages),
    sources: results.map((r) => ({
      title: r.metadata.title,
      sectionTitle: r.metadata.sectionTitle,
      source: r.metadata.source,
      score: r.score,
      excerpt: r.content.slice(0, 200) + '...',
    })),
  }
}
