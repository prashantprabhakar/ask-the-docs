import { Ollama } from 'ollama'
import type { ILLMClient, IEmbeddingClient, Message } from './types'
import { ollama, llm } from '../config'

const client = new Ollama({ host: ollama.baseURL })
const llmModel = llm.model
const embeddingModel = llm.embeddingModel

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
