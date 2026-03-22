import type { ILLMClient, IEmbeddingClient } from './types'
import { OllamaLLMClient, OllamaEmbeddingClient } from './ollama.client'
import { OpenAILLMClient, OpenAIEmbeddingClient } from './openai.client'

// Add a new provider: create a client file, add a case here. That's it.

export function createLLMClient(): ILLMClient {
  const provider = process.env.LLM_PROVIDER ?? 'ollama'

  switch (provider) {
    case 'ollama':
      return new OllamaLLMClient()
    case 'openai':
    case 'github': // GitHub Models uses the same OpenAI-compatible client
      return new OpenAILLMClient()
    default:
      throw new Error(
        `Unknown LLM_PROVIDER: "${provider}". Valid options: ollama, openai, github`
      )
  }
}

export function createEmbeddingClient(): IEmbeddingClient {
  const provider = process.env.EMBEDDING_PROVIDER ?? 'ollama'

  switch (provider) {
    case 'ollama':
      return new OllamaEmbeddingClient()
    case 'openai':
    case 'github':
      return new OpenAIEmbeddingClient()
    default:
      throw new Error(
        `Unknown EMBEDDING_PROVIDER: "${provider}". Valid options: ollama, openai, github`
      )
  }
}
