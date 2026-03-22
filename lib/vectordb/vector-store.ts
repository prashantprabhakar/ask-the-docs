/**
 * File-based vector store — no server, no Docker, no setup.
 *
 * Stores all chunks + their embeddings in a single JSON file.
 * On each query, computes cosine similarity against every stored vector.
 *
 * This is exactly what production vector DBs (Pinecone, Chroma, Weaviate) do
 * internally — just optimized with index structures (HNSW, IVF) for scale.
 * For thousands of docs this pure approach is plenty fast.
 *
 * LEARN: Cosine similarity measures the angle between two vectors.
 *   - Score = 1.0 → identical meaning
 *   - Score = 0.0 → completely unrelated
 *   - The query embedding and chunk embeddings are in the same "semantic space"
 *     because they were created by the same embedding model.
 */

import fs from 'fs'
import path from 'path'

export interface DocChunk {
  id: string
  content: string
  embedding: number[]
  metadata: {
    source: string
    title: string
    chunkIndex: number
  }
}

const STORE_PATH = path.join(process.cwd(), 'data', 'vector-store.json')

// ─── Cosine Similarity ────────────────────────────────────────────────────────
// dot(A, B) / (|A| * |B|)
// Two vectors pointing in the same direction → score near 1
// Two vectors pointing in opposite directions → score near -1

function dotProduct(a: number[], b: number[]): number {
  let sum = 0
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i]
  return sum
}

function magnitude(v: number[]): number {
  return Math.sqrt(v.reduce((sum, x) => sum + x * x, 0))
}

function cosineSimilarity(a: number[], b: number[]): number {
  const magA = magnitude(a)
  const magB = magnitude(b)
  if (magA === 0 || magB === 0) return 0
  return dotProduct(a, b) / (magA * magB)
}

// ─── Persistence ─────────────────────────────────────────────────────────────

function loadStore(): DocChunk[] {
  if (!fs.existsSync(STORE_PATH)) return []
  const raw = fs.readFileSync(STORE_PATH, 'utf-8')
  return JSON.parse(raw) as DocChunk[]
}

function saveStore(chunks: DocChunk[]) {
  fs.mkdirSync(path.dirname(STORE_PATH), { recursive: true })
  fs.writeFileSync(STORE_PATH, JSON.stringify(chunks), 'utf-8')
}

// ─── Public API (same shape as chroma.client.ts was) ─────────────────────────

/**
 * Add or update chunks in the store.
 * Uses chunk.id to deduplicate — safe to re-run ingest.
 */
export function upsertChunks(chunks: DocChunk[]) {
  const existing = loadStore()
  const map = new Map(existing.map((c) => [c.id, c]))
  for (const chunk of chunks) map.set(chunk.id, chunk)
  const updated = Array.from(map.values())
  saveStore(updated)
  console.log(`Stored ${chunks.length} chunks (total: ${updated.length})`)
}

/**
 * Find the top-K chunks most semantically similar to the query.
 * This is the heart of RAG retrieval.
 */
export function similaritySearch(
  queryEmbedding: number[],
  topK = 5
): { content: string; metadata: DocChunk['metadata']; score: number }[] {
  const chunks = loadStore()

  const scored = chunks.map((chunk) => ({
    content: chunk.content,
    metadata: chunk.metadata,
    score: cosineSimilarity(queryEmbedding, chunk.embedding),
  }))

  // Sort descending by similarity score, take top K
  return scored.filter((a) => a.score > .6 ).sort((a, b) => b.score - a.score).slice(0, topK)
}

export function getChunkCount(): number {
  return loadStore().length
}

export function clearStore() {
  if (fs.existsSync(STORE_PATH)) {
    fs.unlinkSync(STORE_PATH)
    console.log('Vector store cleared.')
  }
}
