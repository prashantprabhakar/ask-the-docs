/**
 * Query expansion — generate alternative phrasings of the user's question.
 *
 * PROBLEM — Vocabulary mismatch:
 *   The user asks "how do I handle 404s?" but the docs say "not found pages".
 *   The embedding model encodes meaning, but when query vocabulary diverges
 *   enough from doc vocabulary, even semantic search can miss the right chunk.
 *
 * SOLUTION — Multi-query retrieval:
 *   Generate N rephrased versions of the question. Search for each variant
 *   independently, then merge the candidate pools. Any chunk that answers
 *   the question from ANY phrasing angle gets a chance to surface.
 *
 *   question → LLM → [variant1, variant2, variant3]
 *        ↓              ↓           ↓           ↓
 *     search        search      search      search
 *        └──────── merge + deduplicate by content ────────┘
 *                          ↓
 *                       re-rank
 *
 * WHY LLM AND NOT WORDNET / SYNONYM TABLES?
 *   The LLM understands domain context. For "Next.js 404 page" it knows to
 *   generate "not-found route", "custom error page", "notFound() function" —
 *   not just surface-level word synonyms.
 *
 * COST — One extra LLM call per query. Use the same model as the main chat
 *   (or a cheaper/faster model if latency matters). The call is small: a
 *   one-sentence question in, a short JSON array out.
 *
 * PROD NOTE — Alternatives to consider:
 *   - HyDE (Hypothetical Document Embeddings): instead of expanding the
 *     question, ask the LLM to write a hypothetical answer and embed *that*.
 *     Works well for vague or short queries. Simpler to implement (one search,
 *     not N), but loses the breadth that multi-query provides.
 *   - Smaller/faster model for expansion: since the task is mechanical
 *     (rephrase, not reason), a small model (e.g. haiku, gpt-4o-mini) reduces
 *     latency and cost.
 */

import { createLLMClient } from '../llm/factory'

const llm = createLLMClient()

/**
 * How many alternative phrasings to generate.
 * 3 variants + the original = 4 parallel searches.
 * More variants → better recall, more Qdrant queries, more embedding calls.
 */
const EXPANSION_COUNT = 3

/**
 * Generate alternative phrasings of `question` using the LLM.
 *
 * Returns an array that always starts with the original question so the
 * caller can treat all elements uniformly (original is never lost).
 *
 * Fails gracefully: if the LLM response cannot be parsed, returns just
 * [question] so retrieval still works without expansion.
 */
export async function expandQuery(question: string): Promise<string[]> {
  let raw: string
  try {
    raw = await llm.chat([
      {
        role: 'system',
        content: `You are a search query assistant. Given a user question about technical documentation, generate ${EXPANSION_COUNT} alternative phrasings that mean the same thing but use different vocabulary. This helps retrieve relevant docs even when the user's wording differs from the documentation's wording.

Rules:
- Keep each variant concise (one sentence or less)
- Vary vocabulary, not meaning — do not introduce new concepts
- Include technical synonyms and alternative phrasing styles
- Output ONLY a JSON array of strings, no explanation, no markdown

Example input: "how do I handle 404s in Next.js?"
Example output: ["how to create a custom not found page in Next.js", "Next.js notFound() function usage", "configure not-found route in App Router"]`,
      },
      {
        role: 'user',
        content: question,
      },
    ])
  } catch {
    // LLM call failed — degrade gracefully, retrieval still runs on original
    return [question]
  }

  const variants = parseVariants(raw)
  // Always prepend the original so it is always searched
  return [question, ...variants]
}

/**
 * Parse the LLM response into a string array.
 * Handles both clean JSON and JSON wrapped in markdown code fences.
 */
function parseVariants(raw: string): string[] {
  try {
    // Strip markdown code fences if the LLM wrapped the JSON
    const cleaned = raw.trim().replace(/^```(?:json)?\n?/, '').replace(/\n?```$/, '')
    const parsed = JSON.parse(cleaned)

    if (!Array.isArray(parsed)) return []

    return parsed
      .filter((v): v is string => typeof v === 'string' && v.trim().length > 0)
      .slice(0, EXPANSION_COUNT)
  } catch {
    return []
  }
}
