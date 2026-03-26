import 'dotenv/config'
import { NextRequest } from 'next/server'
import { ragQueryStream } from '@/lib/rag/retriever'

export async function POST(req: NextRequest) {
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
    },
  })
}
