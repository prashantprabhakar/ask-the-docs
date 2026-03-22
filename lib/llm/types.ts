// The two interfaces the entire app depends on.
// Nothing outside lib/llm/ imports a concrete client — only these types.

export interface Message {
  role: 'system' | 'user' | 'assistant'
  content: string
}

/**
 * Any LLM that can hold a conversation.
 * Ollama, OpenAI, GitHub Models — all implement this.
 */
export interface ILLMClient {
  chat(messages: Message[]): Promise<string>
  streamChat(messages: Message[]): AsyncIterable<string>
}

/**
 * Any model that can turn text into a vector (array of numbers).
 * Ollama, OpenAI — all implement this.
 */
export interface IEmbeddingClient {
  embed(text: string): Promise<number[]>
  embedBatch(texts: string[]): Promise<number[][]>
}
