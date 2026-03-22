import { Ollama } from 'ollama'
import type { ILLMClient, IEmbeddingClient, Message } from './types'

// Reads from env — set OLLAMA_BASE_URL and LLM_MODEL / EMBEDDING_MODEL in .env
const baseURL = process.env.OLLAMA_BASE_URL ?? 'http://localhost:11434'
const llmModel = process.env.LLM_MODEL ?? 'llama3.2'
const embeddingModel = process.env.EMBEDDING_MODEL ?? 'nomic-embed-text'

const client = new Ollama({ host: baseURL })

export class OllamaLLMClient implements ILLMClient {
  async chat(messages: Message[]): Promise<string> {
    const response = await client.chat({
      model: llmModel,
      messages,
    })
    return response.message.content
  }

  async *streamChat(messages: Message[]): AsyncIterable<string> {
    const stream = await client.chat({
      model: llmModel,
      messages,
      stream: true,
    })
    for await (const chunk of stream) {
      yield chunk.message.content
    }
  }
}

export class OllamaEmbeddingClient implements IEmbeddingClient {
  async embed(text: string): Promise<number[]> {
    const response = await client.embed({ model: embeddingModel, input: text })
    return response.embeddings[0]
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const response = await client.embed({ model: embeddingModel, input: texts })
    return response.embeddings
  }
}
