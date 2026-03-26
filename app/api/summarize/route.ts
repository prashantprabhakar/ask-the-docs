import 'dotenv/config'
import { NextRequest } from 'next/server'
import { createLLMClient } from '@/lib/llm/factory'
import type { Message } from '@/lib/llm/types'

const llm = createLLMClient()

/**
 * POST /api/summarize
 *
 * Summarizes old conversation turns that have scrolled outside the recent
 * window. The frontend calls this in the background after each response
 * when the conversation exceeds the summary threshold.
 *
 * Body: { messages: Message[], existingSummary?: string }
 * Response: { summary: string }
 *
 * LEARN — Progressive summarization:
 *   Rather than sending the entire conversation history to the LLM on every
 *   request (expensive, eventually hits context limits), we compress old turns
 *   into a rolling summary paragraph. The summary grows slightly with each
 *   summarization call as new turns are folded in.
 *
 *   Turn window (verbatim):   [T4] [T5] [T6]   ← sent with every request
 *   Summary (compressed):     "User asked about layouts (T1), learned about
 *                              dynamic routes (T2), asked about middleware (T3)"
 *
 * PROD NOTE — For very long conversations the summary itself can grow large.
 *   A production system would cap summary length and use a hierarchical
 *   summarization strategy (summarize the summary when it gets too long).
 */
export async function POST(req: NextRequest) {
  const { messages, existingSummary } = await req.json() as {
    messages: Message[]
    existingSummary?: string
  }

  if (!Array.isArray(messages) || messages.length === 0) {
    return Response.json({ summary: existingSummary ?? '' })
  }

  // Format turns as readable Q&A for the LLM to summarize
  const turns = messages
    .reduce<string[]>((acc, msg, i) => {
      if (msg.role === 'user') acc.push(`Q: ${msg.content}`)
      if (msg.role === 'assistant') acc[acc.length - 1] += `\nA: ${msg.content}`
      return acc
    }, [])
    .join('\n\n')

  const contextBlock = existingSummary
    ? `Prior summary:\n${existingSummary}\n\nNew turns to fold in:\n${turns}`
    : turns

  try {
    const summary = await llm.chat([
      {
        role: 'system',
        content: `You summarize technical Q&A conversations concisely. Write 2–3 sentences in past tense (e.g. "The user asked about..."). Capture the key topics discussed and conclusions reached. If a prior summary is provided, fold the new turns into it — do not repeat information already in the summary. Output only the summary paragraph, no preamble.`,
      },
      {
        role: 'user',
        content: contextBlock,
      },
    ])

    return Response.json({ summary: summary.trim() })
  } catch {
    // If summarization fails, return the existing summary unchanged so the
    // conversation continues working — losing the summary is not fatal.
    return Response.json({ summary: existingSummary ?? '' })
  }
}
