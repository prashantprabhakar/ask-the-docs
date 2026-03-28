import fs from 'fs'
import path from 'path'
import pLimit from 'p-limit'

// ─── Ingest logger (tee: stdout + file) ──────────────────────────────────────

/**
 * Writes every ingest progress line to stdout AND a timestamped log file.
 * This means you can watch the run live AND grep the file after to diagnose
 * what was slow.
 *
 * Log path: logs/ingest-{ISO-timestamp}.log  (created automatically)
 */
class IngestLog {
  private stream: fs.WriteStream | null = null
  private logPath = ''

  open(): void {
    const ts = new Date().toISOString().replace(/:/g, '-').replace(/\..+/, '')
    this.logPath = path.join(process.cwd(), 'logs', `ingest-${ts}.log`)
    fs.mkdirSync(path.dirname(this.logPath), { recursive: true })
    this.stream = fs.createWriteStream(this.logPath, { flags: 'a' })
    this.write(`=== Ingest started ${new Date().toISOString()} ===\n`)
    process.stdout.write(`Log file: ${this.logPath}\n`)
  }

  write(msg: string): void {
    process.stdout.write(msg)
    this.stream?.write(msg)
  }

  warn(msg: string): void {
    this.write(`  ⚠  ${msg}\n`)
  }

  close(): void {
    this.write(`=== Ingest ended ${new Date().toISOString()} ===\n`)
    this.stream?.end()
    this.stream = null
  }
}

const ilog = new IngestLog()
import type { DocType } from '../vectordb'
import { createEmbeddingClient } from '../llm/factory'
import { upsertChunks, chunkId, deleteChunksByIds, deleteChunksBySource, scrollAllChunkTexts } from '../vectordb'
import { encodeSparse, tokenize, reloadIdfTable } from './sparse-encoder'
import { buildIdfTable, saveIdfTable } from './idf-table'
import { loadCache, saveCache, hashContent, type FileCache, type SectionCache } from './ingest-cache'
import { generateContextPrefix } from './context-builder'
import { splitIntoSections, chunkSection } from './chunker'
import type { RawDocument } from './chunker'
import type { DocChunk } from '../vectordb'

/**
 * PROD NOTE — In production, the embedder is not a module-level singleton.
 *   It's injected as a dependency so you can swap it in tests, or run
 *   different embedding models for different doc types (e.g. a code-specific
 *   model for code snippets vs a prose model for narrative docs).
 */
const embedder = createEmbeddingClient()

import { ingestion as ingestionConfig } from '../config'

// ─── Config ───────────────────────────────────────────────────────────────────

const BATCH_SIZE = ingestionConfig.batchSize
const CONTEXT_CONCURRENCY = ingestionConfig.contextConcurrency
const FILE_CONCURRENCY = ingestionConfig.fileConcurrency

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

interface EmbedResult {
  chunks: number
  contextMs: number
  embedUpsertMs: number
  logLines: string[]
  newSectionCache: SectionCache
}

function fmtMs(ms: number): string {
  if (ms < 1_000) return `${ms}ms`
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`
  const mins = Math.floor(ms / 60_000)
  const secs = Math.round((ms % 60_000) / 1000)
  return `${mins}m ${secs}s`
}

/**
 * Embed and store a document using section-level diffing.
 *
 * Only sections whose content has changed since the last run are re-embedded.
 * Unchanged sections are skipped entirely — no LLM call, no embedding API call,
 * no Qdrant write. Changed or new sections replace their old chunk IDs precisely.
 *
 * Algorithm:
 *   1. Split doc into sections (cheap — string ops only)
 *   2. Hash each section's raw content and compare against cachedSections
 *   3. Unchanged → carry cached entry forward, skip
 *      Changed/new → mark for embedding; schedule old IDs for deletion
 *      Removed (in cache but gone from file) → schedule old IDs for deletion
 *   4. Delete stale IDs in one batch
 *   5. Embed + upsert only the changed/new sections
 *   6. Record new chunk IDs in the returned section cache
 */
async function embedAndStore(
  doc: RawDocument,
  filePath: string,
  cachedSections: SectionCache,
): Promise<EmbedResult> {
  const logLines: string[] = []
  const newSectionCache: SectionCache = {}

  const sections = splitIntoSections(doc)
  if (sections.length === 0) {
    return { chunks: 0, contextMs: 0, embedUpsertMs: 0, logLines, newSectionCache }
  }

  // ── Diff sections against cache ───────────────────────────────────────────
  const toEmbed: Array<{ section: typeof sections[0]; hash: string }> = []
  const staleIds: string[] = []
  let skippedSections = 0

  for (const section of sections) {
    const hash = hashContent(section.content)
    const cached = cachedSections[section.headingPath]

    if (cached && cached.hash === hash) {
      newSectionCache[section.headingPath] = cached  // carry forward unchanged
      skippedSections++
    } else {
      if (cached) staleIds.push(...cached.chunkIds)  // old IDs for this section
      toEmbed.push({ section, hash })
    }
  }

  // Sections that existed in cache but are gone from the file → delete their IDs
  const currentHeadings = new Set(sections.map((s) => s.headingPath))
  for (const [title, entry] of Object.entries(cachedSections)) {
    if (!currentHeadings.has(title)) staleIds.push(...entry.chunkIds)
  }

  if (staleIds.length > 0) await deleteChunksByIds(staleIds)

  if (skippedSections > 0) {
    logLines.push(`    ${skippedSections} section(s) unchanged, skipped\n`)
  }

  if (toEmbed.length === 0) {
    return { chunks: 0, contextMs: 0, embedUpsertMs: 0, logLines, newSectionCache }
  }

  // ── Flatten sections → chunks ─────────────────────────────────────────────
  const allChunks: Awaited<ReturnType<typeof chunkSection>> = []
  for (const { section } of toEmbed) {
    allChunks.push(...await chunkSection(doc, section))
  }

  if (allChunks.length === 0) {
    return { chunks: 0, contextMs: 0, embedUpsertMs: 0, logLines, newSectionCache }
  }

  // ── Generate contextual prefixes (concurrent) ─────────────────────────────
  //
  // Context is only generated for chunks that are actually being re-embedded,
  // so the per-run cost scales with changed sections, not total file size.
  //
  // PROD NOTE — Tune CONTEXT_CONCURRENCY per provider:
  //   Ollama local: 3 (CPU/GPU bound — higher values thrash the model server)
  //   OpenAI/GitHub API: 20 (network-bound, watch rate limits)
  const contextLimit = pLimit(CONTEXT_CONCURRENCY)
  const t0 = Date.now()
  let completed = 0

  const contextualContents = await Promise.all(
    allChunks.map((chunk, i) =>
      contextLimit(async () => {
        const chunkT = Date.now()
        const context = await generateContextPrefix(doc.title, chunk.sectionTitle, chunk.content)
        const chunkMs = Date.now() - chunkT
        completed++

        const label = chunk.sectionTitle.slice(0, 45).padEnd(45)
        const slowFlag = chunkMs > 30_000 ? '  ← SLOW' : ''
        logLines.push(`    ctx ${String(completed).padStart(2)}/${allChunks.length}  ${label}  ${fmtMs(chunkMs)}${slowFlag}\n`)

        if (chunkMs > 30_000) {
          logLines.push(`  ⚠  chunk ${i + 1} took ${fmtMs(chunkMs)} — section: "${chunk.sectionTitle}"\n`)
        }

        return context ? `${context}\n\n${chunk.content}` : chunk.content
      })
    )
  )
  const contextMs = Date.now() - t0

  // ── Embed + upsert in batches, tracking IDs per section ───────────────────
  //
  // We record which chunk IDs belong to each section so the next run can
  // delete exactly those IDs if the section changes again.
  //
  // Chunk IDs are content-addressed (UUID v5 of source + contextual content):
  //   same content → same ID → upsert is idempotent
  //   changed content → new ID → old ID was already deleted above
  const t1 = Date.now()
  const sectionIdMap = new Map<string, string[]>()

  for (let i = 0; i < allChunks.length; i += BATCH_SIZE) {
    const batch = allChunks.slice(i, i + BATCH_SIZE)
    const batchContents = contextualContents.slice(i, i + BATCH_SIZE)
    const embeddings = await embedder.embedBatch(batchContents)

    const docChunks: DocChunk[] = batch.map((chunk, j) => {
      const id = chunkId(chunk.source, batchContents[j])
      const ids = sectionIdMap.get(chunk.sectionTitle) ?? []
      ids.push(id)
      sectionIdMap.set(chunk.sectionTitle, ids)

      return {
        id,
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
      }
    })

    await upsertChunks(docChunks)
  }
  const embedUpsertMs = Date.now() - t1

  // Record the new section cache entries for everything we just embedded
  for (const { section, hash } of toEmbed) {
    newSectionCache[section.headingPath] = {
      hash,
      chunkIds: sectionIdMap.get(section.headingPath) ?? [],
    }
  }

  return { chunks: allChunks.length, contextMs, embedUpsertMs, logLines, newSectionCache }
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
 * PROD NOTE — For large doc sets (10k+ files), split the file list across
 *   workers and coordinate via a job queue (BullMQ, Temporal). Each worker
 *   claims a file, processes it, and marks it done. The cache then lives in
 *   a shared DB, not a local JSON file.
 */
export async function ingestDocuments(dir: string, options: IngestOptions = {}) {
  const ingestStart = Date.now()
  ilog.open()
  ilog.write(`\n=== Starting ${options.full ? 'full' : 'incremental'} ingestion ===\n\n`)

  const allFiles = walkFiles(dir)
  const cache: FileCache = options.full ? {} : loadCache()
  const newCache: FileCache = {}

  // ── Step 1: Detect deleted files ──────────────────────────────────────────
  // Any source in the cache that no longer exists on disk is a deleted file.
  // Delete only the exact chunk IDs we tracked — no payload-filter scan needed.
  const diskSources = new Set(allFiles.map((f) => toSource(f, dir)))
  const deletedSources = Object.keys(cache).filter((s) => !diskSources.has(s))

  if (deletedSources.length > 0) {
    ilog.write(`Removing ${deletedSources.length} deleted file(s) from vector store...\n`)
    for (const source of deletedSources) {
      const allIds = Object.values(cache[source]).flatMap((e) => e.chunkIds)
      await deleteChunksByIds(allIds)
      ilog.write(`  Removed: ${source}\n`)
    }
  }

  // ── Step 2: Process all files in parallel (FILE_CONCURRENCY at a time) ────
  //
  // Section-level diffing inside embedAndStore means we only embed sections
  // that actually changed. "Processed" = at least one section changed.
  // "Skipped" = every section was identical to the cached version.
  //
  // Log lines are buffered inside embedAndStore and flushed atomically so
  // output from concurrent files doesn't interleave in the log.
  //
  // For --full ingest: deleteChunksBySource first (nuclear wipe), then embed
  // everything fresh. This guarantees no stale vectors survive a model change.
  let processed = 0
  let skipped = 0
  let totalChunks = 0
  const fileLimit = pLimit(FILE_CONCURRENCY)

  await Promise.all(
    allFiles.map((filePath, idx) =>
      fileLimit(async () => {
        const source = toSource(filePath, dir)
        const doc = loadRawDocument(filePath, source)

        if (options.full) {
          // Wipe all existing vectors for this source before re-embedding
          await deleteChunksBySource(source)
        }

        const cachedSections = options.full ? {} : (cache[source] ?? {})
        const { chunks, contextMs, embedUpsertMs, logLines, newSectionCache } =
          await embedAndStore(doc, filePath, cachedSections)

        // Mutations safe — JS event loop serialises between awaits
        newCache[source] = newSectionCache

        if (chunks > 0) {
          processed++
          totalChunks += chunks
          ilog.write(`\n  [${idx + 1}/${allFiles.length}] ${source}\n`)
          ilog.write(logLines.join(''))
          ilog.write(`       context: ${fmtMs(contextMs)} | embed+upsert: ${fmtMs(embedUpsertMs)} | ${chunks} chunks\n`)
        } else {
          skipped++
        }
      })
    )
  )

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
  ilog.write('\nRebuilding IDF table...\n')
  const allTexts = await scrollAllChunkTexts()
  if (allTexts.length > 0) {
    const termSets = allTexts.map((text) => new Set(tokenize(text)))
    const idfTable = buildIdfTable(termSets)
    saveIdfTable(idfTable)
    reloadIdfTable()
    ilog.write(`IDF table saved: ${Object.keys(idfTable.termDf).length} unique terms across ${idfTable.totalDocs} chunks\n`)
  } else {
    ilog.write('No chunks in store — IDF table not built.\n')
  }

  ilog.write('\n=== Ingestion complete ===\n')
  ilog.write(`Files processed : ${processed}\n`)
  ilog.write(`Files skipped   : ${skipped} (unchanged)\n`)
  ilog.write(`Files deleted   : ${deletedSources.length}\n`)
  ilog.write(`Chunks embedded : ${totalChunks}\n`)
  ilog.write(`Total time      : ${fmtMs(Date.now() - ingestStart)}\n`)
  ilog.close()
}
