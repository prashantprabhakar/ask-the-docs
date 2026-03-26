'use client'

import { useState, useRef, useEffect, useCallback } from 'react'

interface Source {
  title: string
  source: string
  score: number
  excerpt: string
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
}

/**
 * How many recent turns to send verbatim with each request.
 * Older turns are compressed into the rolling summary instead.
 */
const RECENT_WINDOW = 3 // turns (each turn = 1 user + 1 assistant message)

/**
 * Start summarizing once the conversation exceeds this many turns.
 * Below the threshold we send full history; above it we send
 * summary + recent window.
 */
const SUMMARY_THRESHOLD = 6 // turns

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [expandedSources, setExpandedSources] = useState<Set<number>>(new Set())
  /** Rolling summary of turns older than the recent window. */
  const [summary, setSummary] = useState<string>('')
  /**
   * How many messages (not turns) are already captured in `summary`.
   * Prevents re-summarizing the same turns on every response.
   */
  const [summarizedUpTo, setSummarizedUpTo] = useState<number>(0)
  const bottomRef = useRef<HTMLDivElement>(null)

  const toggleSources = useCallback((index: number) => {
    setExpandedSources((prev) => {
      const next = new Set(prev)
      next.has(index) ? next.delete(index) : next.add(index)
      return next
    })
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function sendMessage(e: React.FormEvent) {
    e.preventDefault()
    if (!input.trim() || loading) return

    const question = input.trim()
    setInput('')
    setLoading(true)

    // Send only the recent window of turns. Older context is in `summary`.
    // Stripping sources (UI-only) down to role + content.
    const history = messages
      .slice(-(RECENT_WINDOW * 2))
      .map(({ role, content }) => ({ role, content }))

    setMessages((prev) => [...prev, { role: 'user', content: question }])
    setMessages((prev) => [...prev, { role: 'assistant', content: '', sources: [] }])

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, history, summary: summary || undefined }),
      })

      const reader = res.body!.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const lines = decoder.decode(value).split('\n')
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const data = line.slice(6)
          if (data === '[DONE]') break

          const parsed = JSON.parse(data)

          if (parsed.type === 'sources') {
            setMessages((prev) => {
              const updated = [...prev]
              updated[updated.length - 1] = { ...updated[updated.length - 1], sources: parsed.sources }
              return updated
            })
          }

          if (parsed.type === 'token') {
            setMessages((prev) => {
              const updated = [...prev]
              updated[updated.length - 1] = {
                ...updated[updated.length - 1],
                content: updated[updated.length - 1].content + parsed.token,
              }
              return updated
            })
          }
        }
      }
    } catch {
      setMessages((prev) => {
        const updated = [...prev]
        updated[updated.length - 1] = {
          ...updated[updated.length - 1],
          content: 'Something went wrong. Is Ollama running?',
        }
        return updated
      })
    } finally {
      setLoading(false)
    }

    // Background summarization — runs after the response is shown, never
    // blocks the UI. Compresses old turns into the rolling summary so the
    // next request doesn't need to send the full history.
    setMessages((current) => {
      const totalTurns = current.length / 2
      const windowEnd = current.length - RECENT_WINDOW * 2

      if (totalTurns > SUMMARY_THRESHOLD && windowEnd > summarizedUpTo) {
        const toSummarize = current
          .slice(summarizedUpTo, windowEnd)
          .map(({ role, content }) => ({ role, content }))

        fetch('/api/summarize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages: toSummarize, existingSummary: summary || undefined }),
        })
          .then((r) => r.json())
          .then(({ summary: newSummary }) => {
            if (newSummary) {
              setSummary(newSummary)
              setSummarizedUpTo(windowEnd)
            }
          })
          .catch(() => {
            // Summarization failed — conversation still works, just without
            // the updated summary. Old summary remains in place.
          })
      }

      return current // no state change here — we just needed access to current
    })
  }

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-gray-100">
      <header className="border-b border-gray-800 px-6 py-4">
        <h1 className="text-xl font-semibold">Ask the Docs</h1>
        <p className="text-sm text-gray-400">Next.js documentation — powered by RAG + Ollama</p>
      </header>

      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6 max-w-3xl mx-auto w-full">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-20 space-y-2">
            <p className="text-lg">Ask anything about Next.js</p>
            <p className="text-sm">&quot;How does the App Router handle layouts?&quot;</p>
            <p className="text-sm">&quot;What is server-side rendering in Next.js?&quot;</p>
            <p className="text-sm">&quot;How do I use useRouter in the app directory?&quot;</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-2xl ${msg.role === 'user' ? 'w-auto' : 'w-full'}`}>
              <div
                className={`rounded-2xl px-4 py-3 whitespace-pre-wrap text-sm leading-relaxed ${
                  msg.role === 'user'
                    ? 'bg-blue-600 text-white inline-block'
                    : 'bg-gray-800 text-gray-100 w-full'
                }`}
              >
                {msg.content || (loading && i === messages.length - 1
                  ? <span className="animate-pulse text-gray-400">Thinking...</span>
                  : null
                )}
              </div>

              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-2">
                  <button
                    onClick={() => toggleSources(i)}
                    className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300 transition-colors py-1"
                  >
                    <svg
                      className={`w-3 h-3 transition-transform duration-200 ${expandedSources.has(i) ? 'rotate-90' : ''}`}
                      fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                    </svg>
                    {msg.sources.length} sources used
                  </button>

                  {expandedSources.has(i) && (
                    <div className="mt-1.5 space-y-2">
                      {msg.sources.map((src, j) => (
                        <div key={j} className="bg-gray-900 rounded-lg px-3 py-2 border border-gray-700">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-xs font-medium text-blue-400">{src.title}</span>
                            <span className="text-xs text-gray-500 bg-gray-800 px-2 py-0.5 rounded-full">
                              {(src.score * 100).toFixed(0)}% match
                            </span>
                          </div>
                          <p className="text-xs text-gray-400 line-clamp-2">{src.excerpt}</p>
                          <p className="text-xs text-gray-600 mt-1 font-mono">{src.source}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-gray-800 px-4 py-4">
        <form onSubmit={sendMessage} className="max-w-3xl mx-auto flex gap-3">
          <input
            className="flex-1 bg-gray-800 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 placeholder-gray-500"
            placeholder="Ask about Next.js..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl px-5 py-3 text-sm font-medium transition-colors"
          >
            {loading ? '...' : 'Send'}
          </button>
        </form>
      </div>
    </div>
  )
}
