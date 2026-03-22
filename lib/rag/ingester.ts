import fs from 'fs'
import path from 'path'
import crypto from 'crypto'
import { createEmbeddingClient } from '../llm/factory'
import { chunkDocuments } from './chunker'
import { upsertChunks } from '../vectordb/vector-store'
import type { RawDocument } from './chunker'
import type { DocChunk } from '../vectordb/vector-store'

const embedder = createEmbeddingClient()

/**
 * Recursively find all .md / .mdx / .txt files under a directory.
 */
function walkFiles(dir: string, baseDir: string = dir): string[] {
  const entries = fs.readdirSync(dir, { withFileTypes: true })
  const files: string[] = []
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      files.push(...walkFiles(fullPath, baseDir))
    } else if (entry.name.match(/\.(md|mdx|txt)$/)) {
      files.push(fullPath)
    }
  }
  return files
}

/**
 * Load all .md and .mdx files from a directory (recursively) into RawDocuments.
 */
export function loadMarkdownFiles(dir: string): RawDocument[] {
  const files = walkFiles(dir)
  console.log(`Found ${files.length} files in ${dir}`)

  return files.map((filePath) => {
    const content = fs.readFileSync(filePath, 'utf-8')
    // Use relative path as source so it's readable in citations
    const source = path.relative(dir, filePath)
    // Use first # heading as title, fallback to filename
    const titleMatch = content.match(/^#\s+(.+)/m)
    return {
      content,
      source,
      title: titleMatch?.[1] ?? path.basename(filePath).replace(/\.(md|mdx|txt)$/, ''),
    }
  })
}

/**
 * The full ingestion pipeline:
 * Load docs → chunk → embed → store in ChromaDB
 *
 * This is the "offline" step you run once (or whenever docs change).
 * The chat API only reads from ChromaDB — it never calls this.
 */
export async function ingestDocuments(dir: string) {
  console.log('\n=== Starting ingestion ===\n')

  // Step 1: Load
  const docs = loadMarkdownFiles(dir)
  console.log(`Loaded ${docs.length} documents`)

  // Step 2: Chunk
  const chunks = await chunkDocuments(docs)
  console.log(`Split into ${chunks.length} chunks`)

  // Step 3: Embed + store in batches (avoid hammering the embedding API)
  const BATCH_SIZE = 10
  let stored = 0

  for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
    const batch = chunks.slice(i, i + BATCH_SIZE)

    // Embed all texts in the batch at once (embedBatch is more efficient than one-by-one)
    const embeddings = await embedder.embedBatch(batch.map((c) => c.content))

    const docChunks: DocChunk[] = batch.map((chunk, j) => ({
      // Deterministic ID so re-running ingest updates existing chunks (upsert)
      id: crypto
        .createHash('md5')
        .update(`${chunk.source}-${chunk.chunkIndex}`)
        .digest('hex'),
      content: chunk.content,
      embedding: embeddings[j],
      metadata: {
        source: chunk.source,
        title: chunk.title,
        chunkIndex: chunk.chunkIndex,
      },
    }))

    upsertChunks(docChunks)
    stored += batch.length
    console.log(`Progress: ${stored}/${chunks.length} chunks embedded and stored`)
  }

  console.log('\n=== Ingestion complete ===')
  console.log(`Total: ${chunks.length} chunks from ${docs.length} documents`)
}
