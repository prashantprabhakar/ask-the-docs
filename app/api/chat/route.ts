import 'dotenv/config'
import { NextRequest } from 'next/server'
import { LRUCache } from 'lru-cache'
import { ragQueryStream } from '@/lib/rag/retriever'
import { rateLimit as rateLimitConfig } from '@/lib/config'

// ─── Rate limiter ─────────────────────────────────────────────────────────────

/**
 * Sliding-window rate limiter: 100 requests per IP per 60-second window.
 *
 * Each entry stores an array of request timestamps for that IP. On every
 * request we drop timestamps older than the window, then check the count.
 *
 * LRU cache bounds memory: at most 5 000 IPs tracked, each holding at most
 * LIMIT timestamps. Entries expire after 60 s of inactivity automatically.
 *
 * PROD NOTE — In-memory rate limiting only works for single-instance
 *   deployments. With multiple replicas each instance has its own counter,
 *   so the effective limit becomes LIMIT × replicas. For multi-instance
 *   production use Redis + a Lua script (atomic increment + expire) or
 *   an edge middleware (Vercel Rate Limiting, Cloudflare Workers).
 */
const WINDOW_MS = rateLimitConfig.windowMs
const LIMIT = rateLimitConfig.maxRequests

const ipWindows = new LRUCache<string, number[]>({
  max: rateLimitConfig.maxTrackedIPs,
  ttl: WINDOW_MS,
})

function checkRateLimit(ip: string): { allowed: boolean; remaining: number; retryAfterSec: number } {
  const now = Date.now()
  const windowStart = now - WINDOW_MS
  const timestamps = (ipWindows.get(ip) ?? []).filter((t) => t > windowStart)

  if (timestamps.length >= LIMIT) {
    const oldestInWindow = timestamps[0]
    const retryAfterSec = Math.ceil((oldestInWindow + WINDOW_MS - now) / 1000)
    return { allowed: false, remaining: 0, retryAfterSec }
  }

  timestamps.push(now)
  ipWindows.set(ip, timestamps)
  return { allowed: true, remaining: LIMIT - timestamps.length, retryAfterSec: 0 }
}

// ─── Route handler ────────────────────────────────────────────────────────────

export async function POST(req: NextRequest) {
  const ip = req.headers.get('x-forwarded-for')?.split(',')[0].trim() ?? 'unknown'
  const { allowed, remaining, retryAfterSec } = checkRateLimit(ip)

  if (!allowed) {
    return new Response('Too Many Requests', {
      status: 429,
      headers: {
        'Retry-After': String(retryAfterSec),
        'X-RateLimit-Limit': String(LIMIT),
        'X-RateLimit-Remaining': '0',
        'X-RateLimit-Reset': String(Math.ceil((Date.now() + retryAfterSec * 1000) / 1000)),
      },
    })
  }

  const { question, history = [], summary, filter } = await req.json()

  if (!question?.trim()) {
    return new Response('Missing question', { status: 400 })
  }

  const { stream, sources } = await ragQueryStream(question, history, summary, filter)

  // Server-Sent Events — streams tokens to the UI as they arrive
  const encoder = new TextEncoder()

  const readable = new ReadableStream({
    async start(controller) {
      // First send the sources so the UI can show them immediately
      controller.enqueue(
        encoder.encode(`data: ${JSON.stringify({ type: 'sources', sources })}\n\n`)
      )

      // Then stream the answer token by token
      for await (const token of stream) {
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({ type: 'token', token })}\n\n`)
        )
      }

      controller.enqueue(encoder.encode('data: [DONE]\n\n'))
      controller.close()
    },
  })

  return new Response(readable, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'X-RateLimit-Limit': String(LIMIT),
      'X-RateLimit-Remaining': String(remaining),
    },
  })
}
