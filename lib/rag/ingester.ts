import fs from 'fs'
import path from 'path'
import type { DocType } from '../vectordb'
import { createEmbeddingClient } from '../llm/factory'
import { chunkDocument } from './chunker'
import { upsertChunks, chunkId, deleteChunksBySource, scrollAllChunkTexts } from '../vectordb'
import { encodeSparse, tokenize, reloadIdfTable } from './sparse-encoder'
import { buildIdfTable, saveIdfTable } from './idf-table'
import { loadCache, saveCache, hashFileContent } from './ingest-cache'
import { generateContextPrefix } from './context-builder'
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

/**
 * Infer the documentation type from the source path.
 *
 * Next.js docs structure:
 *   - paths containing "api-reference" → API reference pages
 *   - paths containing "error" → error explanation pages
 *   - everything else → conceptual guides and tutorials
 *
 * PROD NOTE — Extend this as your doc set grows. A config map
 *   ({ pattern, docType }[]) scales better than if/else chains.
 */
function inferDocType(source: string): DocType {
  if (source.includes('api-reference')) return 'api-reference'
  if (source.includes('error')) return 'error'
  return 'guide'
}

/**
 * Build the canonical nextjs.org URL for a doc source path.
 *
 * Source paths use numeric prefixes for filesystem ordering:
 *   "02-app/01-building-your-application/01-routing/01-defining-routes.mdx"
 *
 * Canonical URL strips those prefixes:
 *   "https://nextjs.org/docs/app/building-your-application/routing/defining-routes"
 *
 * Steps:
 *   1. Strip file extension
 *   2. Strip trailing "/index" (directory index pages)
 *   3. Remove numeric sort prefixes (/01-foo → /foo, leading 01-foo → foo)
 */
function buildDocsUrl(source: string): string {
  const withoutExt = source.replace(/\.(md|mdx|txt)$/, '')
  const withoutIndex = withoutExt.replace(/\/index$/, '')
  // Replace both leading and mid-path numeric prefixes: "01-foo" → "foo"
  const withoutPrefixes = withoutIndex.replace(/(^|\/)(\d+-)/g, '$1')
  return `https://nextjs.org/docs/${withoutPrefixes}`
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

async function embedAndStore(doc: RawDocument, filePath: string): Promise<number> {
  const chunks = await chunkDocument(doc)
  if (chunks.length === 0) return 0

  /**
   * Contextual retrieval — prepend an LLM-generated context sentence to each
   * chunk before embedding. This makes ambiguous chunks retrievable.
   *
   * Example — without context the embedding of:
   *   "App Router > Image > Lazy Loading\n\nThe default value is `true`."
   * won't match "is lazy loading enabled by default?" well.
   *
   * With context:
   *   "This chunk describes the default lazy loading setting for next/image.\n\n
   *    App Router > Image > Lazy Loading\n\nThe default value is `true`."
   * the embedding is anchored to the right concept.
   *
   * Context is generated sequentially (one LLM call per chunk). Ingest is
   * already a background script so latency here doesn't affect query time.
   * Incremental ingest means this runs only for new or changed chunks.
   *
   * PROD NOTE — To speed up large ingests, generate context in parallel with
   *   a concurrency limit (p-limit). Keep the limit low enough to avoid
   *   overloading the LLM provider's rate limits.
   */
  const contextualContents: string[] = []
  for (const chunk of chunks) {
    const context = await generateContextPrefix(doc.title, chunk.sectionTitle, chunk.content)
    const withContext = context ? `${context}\n\n${chunk.content}` : chunk.content
    contextualContents.push(withContext)
  }

  for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
    const batch = chunks.slice(i, i + BATCH_SIZE)
    const batchContents = contextualContents.slice(i, i + BATCH_SIZE)
    const embeddings = await embedder.embedBatch(batchContents)

    const docChunks: DocChunk[] = batch.map((chunk, j) => ({
      /**
       * Content-addressed UUID v5.
       * Same content = same ID → upsert is idempotent.
       * Changed content = new ID → old chunk was already deleted by deleteChunksBySource.
       *
       * NOTE — We hash the contextual content (with prefix), not the raw chunk.
       * If the context generation output changes (e.g. prompt update), the hash
       * changes and the chunk is re-embedded on the next full ingest.
       */
      id: chunkId(chunk.source, batchContents[j]),
      content: batchContents[j],
      embedding: embeddings[j],
      sparseVector: encodeSparse(batchContents[j]),
      metadata: {
        source: chunk.source,
        title: chunk.title,
        sectionTitle: chunk.sectionTitle,
        chunkIndex: chunk.chunkIndex,
        lastModified: fs.statSync(filePath).mtime.toISOString(),
        docType: inferDocType(chunk.source),
        url: buildDocsUrl(chunk.source),
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
    const chunks = await embedAndStore(doc, filePath)

    totalChunks += chunks
    processed++
    console.log(`  [${processed}] ${source} → ${chunks} chunks`)
  }

  // ── Step 3: Save updated cache ─────────────────────────────────────────────
  saveCache(newCache)

  // ── Step 4: Rebuild IDF table from full corpus ─────────────────────────────
  // Scroll all chunk texts from Qdrant (including chunks from unchanged files)
  // so the IDF table reflects the complete corpus, not just what changed.
  //
  // This runs even when nothing changed (processed === 0) because the table
  // may not exist yet (e.g. first run after pulling the repo).
  //
  // PROD NOTE — For large corpora, maintain term counts incrementally
  //   (update only the chunks that changed) rather than scrolling everything.
  //   See idf-table.ts for details.
  console.log('\nRebuilding IDF table...')
  const allTexts = await scrollAllChunkTexts()
  if (allTexts.length > 0) {
    const termSets = allTexts.map((text) => new Set(tokenize(text)))
    const idfTable = buildIdfTable(termSets)
    saveIdfTable(idfTable)
    reloadIdfTable()
    console.log(`IDF table saved: ${Object.keys(idfTable.termDf).length} unique terms across ${idfTable.totalDocs} chunks`)
  } else {
    console.log('No chunks in store — IDF table not built.')
  }

  console.log('\n=== Ingestion complete ===')
  console.log(`Files processed : ${processed}`)
  console.log(`Files skipped   : ${skipped} (unchanged)`)
  console.log(`Files deleted   : ${deletedSources.length}`)
  console.log(`Chunks embedded : ${totalChunks}`)
}
