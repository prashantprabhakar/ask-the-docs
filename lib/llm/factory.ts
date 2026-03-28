import type { ILLMClient, IEmbeddingClient } from './types'
import { OllamaLLMClient, OllamaEmbeddingClient } from './ollama.client'
import { OpenAILLMClient, OpenAIEmbeddingClient } from './openai.client'
import { llm } from '../config'

// Add a new provider: create a client file, add a case here. That's it.

export function createLLMClient(): ILLMClient {
  switch (llm.provider) {
    case 'ollama':
      return new OllamaLLMClient()
    case 'openai':
    case 'github': // GitHub Models uses the same OpenAI-compatible client
      return new OpenAILLMClient()
    default:
      throw new Error(
        `Unknown LLM_PROVIDER: "${llm.provider}". Valid options: ollama, openai, github`
      )
  }
}

export function createEmbeddingClient(): IEmbeddingClient {
  switch (llm.embeddingProvider) {
    case 'ollama':
      return new OllamaEmbeddingClient()
    case 'openai':
    case 'github':
      return new OpenAIEmbeddingClient()
    default:
      throw new Error(
        `Unknown EMBEDDING_PROVIDER: "${llm.embeddingProvider}". Valid options: ollama, openai, github`
      )
  }
}
