/**
 * Contextual retrieval — generate a context sentence per chunk at ingest time.
 *
 * PROBLEM — Ambiguous chunks:
 *   A chunk that says "The default value is `true`." is meaningless in isolation.
 *   The embedding model encodes that text without knowing it refers to lazy
 *   loading in next/image. The resulting vector will not match queries like
 *   "is lazy loading on by default?" — even though the chunk answers it.
 *
 * SOLUTION — LLM-generated context prefix:
 *   Before embedding, ask a fast LLM to write one sentence describing what the
 *   chunk covers in relation to the full document. Prepend that sentence to the
 *   chunk content. The embedding now carries the right meaning.
 *
 *   Without context:
 *     "App Router > Image > Lazy Loading\n\nThe default value is `true`."
 *
 *   With context:
 *     "This chunk describes the default lazy loading behaviour of the next/image
 *     component in the App Router, where lazy loading is enabled by default.\n\n
 *     App Router > Image > Lazy Loading\n\nThe default value is `true`."
 *
 * COST — One LLM call per chunk, at ingest time only. Because ingest is
 *   incremental (skips unchanged files), this cost is amortised: pay once per
 *   chunk, never again unless the chunk changes.
 *
 * RESULT — Anthropic's 2024 paper reported a ~49% reduction in retrieval
 *   failures using this technique on their own documentation corpora.
 *
 * PROD NOTE — Use the cheapest/fastest model available for this task. The
 *   generation is mechanical ("what does this chunk cover?"), not reasoning.
 *   claude-haiku, gpt-4o-mini, or a small local Ollama model all work well.
 *   The context sentence quality matters less than you'd expect — even a rough
 *   description improves embeddings significantly over having no context.
 */

import { createLLMClient } from '../llm/factory'

const llm = createLLMClient()

/**
 * Generate a one-sentence context description for a chunk.
 *
 * @param docTitle  - Title of the source document (e.g. "Image Optimization")
 * @param sectionTitle - Full heading path (e.g. "App Router > Image > Lazy Loading")
 * @param chunkContent - The raw chunk text (without the section prefix)
 * @returns A context sentence, or empty string on failure (fail-open).
 */
export async function generateContextPrefix(
  docTitle: string,
  sectionTitle: string,
  chunkContent: string
): Promise<string> {
  try {
    const sentence = await llm.chat([
      {
        role: 'system',
        content: `You are a technical documentation indexer. Given a chunk of documentation, write ONE concise sentence (max 30 words) that describes what the chunk covers. Be specific — mention API names, component names, and feature names. Output only the sentence, no preamble, no punctuation at start.`,
      },
      {
        role: 'user',
        content: `Document: ${docTitle}
Section: ${sectionTitle}

Chunk:
${chunkContent}`,
      },
    ])

    const trimmed = sentence.trim()
    return trimmed.length > 0 ? trimmed : ''
  } catch {
    // LLM unavailable or errored — skip context prefix for this chunk.
    // The section title prefix still provides heading-level context.
    return ''
  }
}
