import fs from 'fs'
import path from 'path'
import { createEmbeddingClient } from '../llm/factory'
import { chunkDocument } from './chunker'
import { upsertChunks, chunkId, deleteChunksBySource } from '../vectordb'
import { encodeSparse } from './sparse-encoder'
import { loadCache, saveCache, hashFileContent } from './ingest-cache'
import type { RawDocument } from './chunker'
import type { DocChunk } from '../vectordb'

/**
 * PROD NOTE — In production, the embedder is not a module-level singleton.
 *   It's injected as a dependency so you can swap it in tests, or run
 *   different embedding models for different doc types (e.g. a code-specific
 *   model for code snippets vs a prose model for narrative docs).
 */
const embedder = createEmbeddingClient()

// ─── Config ───────────────────────────────────────────────────────────────────

/**
 * PROD NOTE — Tune this per embedding provider:
 *   Ollama local:   10–20   (CPU/GPU bound)
 *   OpenAI API:     100–500 (large batches are efficient, watch rate limits)
 *   In production you'd also run batches concurrently with a semaphore
 *   (p-limit) rather than sequentially.
 */
const BATCH_SIZE = 10

// ─── File Walking ─────────────────────────────────────────────────────────────

/**
 * Recursively find all .md / .mdx / .txt files under a directory.
 * Returns full absolute paths.
 *
 * PROD NOTE — Production loaders handle many more source types:
 *   PDF, HTML, Confluence/Notion pages, Word docs, Google Docs exports.
 *   Each needs its own parser to strip nav, footers, and boilerplate —
 *   raw HTML or PDF text is noisy and degrades retrieval quality.
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

function toSource(filePath: string, dir: string): string {
  return path.relative(dir, filePath).replace(/\\/g, '/') // normalize to forward slashes
}

function loadRawDocument(filePath: string, source: string): RawDocument {
  const content = fs.readFileSync(filePath, 'utf-8')
  const titleMatch = content.match(/^#\s+(.+)/m)
  return {
    content,
    source,
    title: titleMatch?.[1] ?? path.basename(filePath).replace(/\.(md|mdx|txt)$/, ''),
  }
}

// ─── Per-file Embed + Store ───────────────────────────────────────────────────

async function embedAndStore(doc: RawDocument): Promise<number> {
  const chunks = await chunkDocument(doc)
  if (chunks.length === 0) return 0

  for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
    const batch = chunks.slice(i, i + BATCH_SIZE)
    const embeddings = await embedder.embedBatch(batch.map((c) => c.content))

    const docChunks: DocChunk[] = batch.map((chunk, j) => ({
      /**
       * Content-addressed UUID v5.
       * Same content = same ID → upsert is idempotent.
       * Changed content = new ID → old chunk was already deleted by deleteChunksBySource.
       */
      id: chunkId(chunk.source, chunk.content),
      content: chunk.content,
      embedding: embeddings[j],
      sparseVector: encodeSparse(chunk.content),
      metadata: {
        source: chunk.source,
        title: chunk.title,
        sectionTitle: chunk.sectionTitle,
        chunkIndex: chunk.chunkIndex,
        /**
         * PROD NOTE — Add here:
         *   lastModified: fs.statSync(filePath).mtime.toISOString()
         *   docType: inferDocType(chunk.source)
         *   url: buildDocsUrl(chunk.source)
         */
      },
    }))

    await upsertChunks(docChunks)
  }

  return chunks.length
}

// ─── Ingestion Pipeline ───────────────────────────────────────────────────────

export interface IngestOptions {
  /**
   * Force re-embed every file even if content hasn't changed.
   * Use when you switch embedding models — the vectors are no longer
   * comparable so everything must be re-embedded from scratch.
   *
   * PROD NOTE — In production, model changes are treated as a migration event:
   *   1. Stand up a second Qdrant collection with the new model's vectors
   *   2. Dual-write queries to both while the new collection fills up
   *   3. Switch traffic to the new collection once ingest is complete
   *   4. Drop the old collection
   *   This gives zero-downtime model upgrades.
   */
  full?: boolean
}

/**
 * Incremental ingestion pipeline: detect changes → delete stale → embed → store.
 *
 * On each run:
 *   - Files with unchanged content are skipped entirely (no API calls)
 *   - Changed or new files: old chunks deleted first, then re-embedded
 *   - Files deleted from disk: their chunks are removed from Qdrant
 *
 * This means you can run this on a cron job (or on every git pull of the docs)
 * and it will stay in sync cheaply.
 *
 * PROD NOTE — This is a sequential, single-process pipeline. For large doc sets
 *   (10k+ files), split the file list across workers and coordinate via a job
 *   queue (BullMQ, Temporal). Each worker claims a file, processes it, and marks
 *   it done. The cache then lives in a shared DB, not a local JSON file.
 */
export async function ingestDocuments(dir: string, options: IngestOptions = {}) {
  console.log(`\n=== Starting ${options.full ? 'full' : 'incremental'} ingestion ===\n`)

  const allFiles = walkFiles(dir)
  const cache = options.full ? {} : loadCache()
  const newCache: Record<string, string> = {}

  // ── Step 1: Detect deleted files ──────────────────────────────────────────
  // Any source in the cache that no longer exists on disk is a deleted file.
  // Remove its chunks from Qdrant.
  const diskSources = new Set(allFiles.map((f) => toSource(f, dir)))
  const deletedSources = Object.keys(cache).filter((s) => !diskSources.has(s))

  if (deletedSources.length > 0) {
    console.log(`Removing ${deletedSources.length} deleted file(s) from vector store...`)
    for (const source of deletedSources) {
      await deleteChunksBySource(source)
      console.log(`  Removed: ${source}`)
    }
  }

  // ── Step 2: Process each file ──────────────────────────────────────────────
  let processed = 0
  let skipped = 0
  let totalChunks = 0

  for (const filePath of allFiles) {
    const source = toSource(filePath, dir)
    const content = fs.readFileSync(filePath, 'utf-8')
    const hash = hashFileContent(content)

    // Always carry the current hash forward into the new cache
    newCache[source] = hash

    if (!options.full && cache[source] === hash) {
      // Content unchanged — no work to do for this file
      skipped++
      continue
    }

    /**
     * File is new or changed.
     *
     * Delete first, then upsert. This is the critical two-step that prevents
     * orphan chunks when a file shrinks (fewer sections after editing) or
     * when section boundaries shift (changing chunk content → new UUID,
     * old UUID becomes an orphan).
     *
     * PROD NOTE — delete + upsert is not atomic. If the process crashes between
     *   the two, the file will have no chunks in Qdrant until the next run
     *   (which will re-ingest it since the cache won't have been saved yet).
     *   For truly atomic updates, use Qdrant's collection aliases: write to a
     *   shadow collection, then swap the alias. Overkill for this project.
     */
    await deleteChunksBySource(source)

    const doc = loadRawDocument(filePath, source)
    const chunks = await embedAndStore(doc)

    totalChunks += chunks
    processed++
    console.log(`  [${processed}] ${source} → ${chunks} chunks`)
  }

  // ── Step 3: Save updated cache ─────────────────────────────────────────────
  saveCache(newCache)

  console.log('\n=== Ingestion complete ===')
  console.log(`Files processed : ${processed}`)
  console.log(`Files skipped   : ${skipped} (unchanged)`)
  console.log(`Files deleted   : ${deletedSources.length}`)
  console.log(`Chunks embedded : ${totalChunks}`)
}
