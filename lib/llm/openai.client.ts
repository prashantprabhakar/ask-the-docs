import OpenAI from 'openai'
import type { ILLMClient, IEmbeddingClient, Message } from './types'
import { openai as openaiConfig, llm } from '../config'

// GitHub Models is OpenAI-compatible — same SDK, just different baseURL + token.
// Set OPENAI_PROVIDER=github in .env to use GitHub Models instead of OpenAI.
function buildClient() {
  if (openaiConfig.provider === 'github') {
    return new OpenAI({ baseURL: openaiConfig.githubBaseURL, apiKey: openaiConfig.githubToken })
  }
  return new OpenAI({ apiKey: openaiConfig.apiKey })
}

function getModel() {
  if (openaiConfig.provider === 'github') return openaiConfig.githubModel
  return llm.model
}

function getEmbeddingModel() {
  return llm.embeddingModel
}

export class OpenAILLMClient implements ILLMClient {
  private client = buildClient()
  private model = getModel()

  async chat(messages: Message[]): Promise<string> {
    const response = await this.client.chat.completions.create({
      model: this.model,
      messages,
    })
    return response.choices[0].message.content ?? ''
  }

  async *streamChat(messages: Message[]): AsyncIterable<string> {
    const stream = await this.client.chat.completions.create({
      model: this.model,
      messages,
      stream: true,
    })
    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content
      if (delta) yield delta
    }
  }
}

export class OpenAIEmbeddingClient implements IEmbeddingClient {
  private client = buildClient()
  private model = getEmbeddingModel()

  async embed(text: string): Promise<number[]> {
    const response = await this.client.embeddings.create({
      model: this.model,
      input: text,
    })
    return response.data[0].embedding
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const response = await this.client.embeddings.create({
      model: this.model,
      input: texts,
    })
    return response.data.map((d) => d.embedding)
  }
}
