/**
 * Vector store abstraction layer.
 *
 * All application code imports from here — never from a specific client file.
 * To swap the backing store (Qdrant → Pinecone, or back to flat JSON),
 * change the one line below. Nothing else in the codebase needs to touch.
 *
 * PROD NOTE — In a larger codebase you'd go one step further: define an
 *   IVectorStore interface in this file and have each client implement it.
 *   Then the active client is injected via dependency injection rather than
 *   re-exported here. That makes unit-testing retrievers trivial — you pass
 *   in a mock store instead of needing a real Qdrant instance.
 *
 *   interface IVectorStore {
 *     upsertChunks(chunks: DocChunk[]): Promise<void>
 *     similaritySearch(embedding: number[], topK: number): Promise<SearchResult[]>
 *     deleteChunksBySource(source: string): Promise<void>
 *     getChunkCount(): Promise<number>
 *     clearStore(): Promise<void>
 *     healthCheck(): Promise<boolean>
 *   }
 */

export * from './qdrant.client'
