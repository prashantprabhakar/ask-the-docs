import fs from 'fs'
import path from 'path'
import { createEmbeddingClient } from '../llm/factory'
import { chunkDocuments } from './chunker'
import { upsertChunks, chunkId } from '../vectordb'
import type { RawDocument } from './chunker'
import type { DocChunk } from '../vectordb'

/**
 * PROD NOTE — In production, the embedder is not a module-level singleton.
 *   It's injected as a dependency so you can swap it in tests, or run
 *   different embedding models for different doc types (e.g. a code-specific
 *   model for code snippets vs a prose model for narrative docs).
 */
const embedder = createEmbeddingClient()

// ─── File Loading ─────────────────────────────────────────────────────────────

/**
 * Recursively find all .md / .mdx / .txt files under a directory.
 *
 * PROD NOTE — Production loaders handle many more source types:
 *   - PDF (via pdf-parse, LlamaParse, or Docling)
 *   - HTML pages crawled from a website
 *   - Confluence / Notion pages via their APIs
 *   - Word docs, Google Docs exports
 *   Each format needs its own parser to strip nav, ads, and boilerplate before
 *   chunking — raw HTML or PDF text is noisy and degrades retrieval quality.
 */
function walkFiles(dir: string): string[] {
  const entries = fs.readdirSync(dir, { withFileTypes: true })
  const files: string[] = []
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      files.push(...walkFiles(fullPath))
    } else if (entry.name.match(/\.(md|mdx|txt)$/)) {
      files.push(fullPath)
    }
  }
  return files
}

export function loadMarkdownFiles(dir: string): RawDocument[] {
  const files = walkFiles(dir)
  console.log(`Found ${files.length} files in ${dir}`)

  return files.map((filePath) => {
    const content = fs.readFileSync(filePath, 'utf-8')
    const source = path.relative(dir, filePath).replace(/\\/g, '/') // normalize to forward slashes
    const titleMatch = content.match(/^#\s+(.+)/m)
    return {
      content,
      source,
      title: titleMatch?.[1] ?? path.basename(filePath).replace(/\.(md|mdx|txt)$/, ''),
    }
  })
}

// ─── Ingestion Pipeline ───────────────────────────────────────────────────────

/**
 * The full ingestion pipeline: Load → Chunk → Embed → Store
 *
 * This is the "offline" step — run once, or whenever docs change.
 * The chat API only reads from Qdrant; it never calls this.
 *
 * PROD NOTE — In production this becomes a pipeline orchestrated by a task
 *   runner (Temporal, Airflow, or even a simple queue). Each stage runs
 *   independently, has retries, and writes progress to a database so you can
 *   resume a failed ingest without starting from scratch. For large doc sets
 *   (10k+ files), embed batches run in parallel workers.
 */
export async function ingestDocuments(dir: string) {
  console.log('\n=== Starting ingestion ===\n')

  // Step 1: Load
  const docs = loadMarkdownFiles(dir)
  console.log(`Loaded ${docs.length} documents`)

  // Step 2: Chunk
  const chunks = await chunkDocuments(docs)
  console.log(`Split into ${chunks.length} chunks`)

  // Step 3: Embed + store in batches
  /**
   * BATCH_SIZE controls how many chunks we embed per API call.
   *
   * PROD NOTE — Tune this per embedding provider:
   *   Ollama local:          10–20  (CPU/GPU bound, small batches are fine)
   *   OpenAI API:            100–500 (their batch endpoint handles large batches well)
   *   In production you'd also add concurrency (Promise.all over batches)
   *   with a semaphore to respect rate limits. Libraries like p-limit work well.
   */
  const BATCH_SIZE = 10
  let stored = 0

  for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
    const batch = chunks.slice(i, i + BATCH_SIZE)

    const embeddings = await embedder.embedBatch(batch.map((c) => c.content))

    const docChunks: DocChunk[] = batch.map((chunk, j) => ({
      /**
       * Deterministic UUID v5 derived from source + content.
       *
       * Why this is correct:
       *   - Same content in same file → same UUID → upsert overwrites cleanly
       *   - Content changes (even slightly) → new UUID → old chunk becomes
       *     an orphan that gets cleaned up by deleteChunksBySource (Step 3)
       *   - Content stays the same but moves position in the file → same UUID
       *     → not re-embedded (zero wasted API calls) — implemented in Step 3
       *
       * PROD NOTE — Because the ID is content-addressed, identical content in
       *   two different files gets different IDs (source is part of the input).
       *   If you want cross-document deduplication instead, use chunkId('', content).
       */
      id: chunkId(chunk.source, chunk.content),
      content: chunk.content,
      embedding: embeddings[j],
      metadata: {
        source: chunk.source,
        title: chunk.title,
        chunkIndex: chunk.chunkIndex,
        /**
         * PROD NOTE — Add here:
         *   lastModified: fs.statSync(filePath).mtime.toISOString()
         *   docType: inferDocType(chunk.source)  e.g. 'api-reference' | 'guide'
         *   url: buildDocsUrl(chunk.source)
         * These unlock filtered search and freshness-aware ranking later.
         */
      },
    }))

    await upsertChunks(docChunks)
    stored += batch.length
    console.log(`Progress: ${stored}/${chunks.length} chunks embedded and stored`)
  }

  console.log('\n=== Ingestion complete ===')
  console.log(`Total: ${chunks.length} chunks from ${docs.length} documents`)
}
