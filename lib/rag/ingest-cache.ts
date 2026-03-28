/**
 * Section-level cache for incremental ingestion.
 *
 * Stores { source → { sectionTitle → { hash, chunkIds } } } so the ingester
 * can skip sections that haven't changed since the last run, and delete only
 * the exact chunk IDs belonging to sections that have changed or been removed.
 *
 * Upgrade from file-level cache (v2): previously stored { source → MD5(fileContent) }.
 * Old entries are automatically discarded on first run after upgrade — the file
 * will be fully re-ingested once, then section-level tracking takes over.
 *
 * PROD NOTE — This is a local JSON file, fine for a single-machine ingest script.
 *   In a distributed pipeline (multiple workers), store this in a database
 *   (Postgres, Redis) so all workers share the same view of what's been processed.
 */

import fs from 'fs'
import path from 'path'
import crypto from 'crypto'

const CACHE_PATH = path.join(process.cwd(), 'data', 'ingest-cache.json')

export interface SectionEntry {
  /** MD5 of the section's raw content — used to detect changes. */
  hash: string
  /** IDs of every chunk upserted for this section. Tracked so we can delete precisely. */
  chunkIds: string[]
}

/** sectionTitle (headingPath) → cache entry */
export type SectionCache = Record<string, SectionEntry>

/** source (relative file path) → per-section cache */
export type FileCache = Record<string, SectionCache>

export function loadCache(): FileCache {
  if (!fs.existsSync(CACHE_PATH)) return {}
  try {
    const raw = JSON.parse(fs.readFileSync(CACHE_PATH, 'utf-8'))
    const migrated: FileCache = {}
    for (const [source, value] of Object.entries(raw)) {
      if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
        // Check it's the new format: values must have { hash, chunkIds }
        const firstEntry = Object.values(value as object)[0]
        if (firstEntry && typeof firstEntry === 'object' && 'hash' in firstEntry) {
          migrated[source] = value as SectionCache
          continue
        }
      }
      // String value (old file-level format) or unrecognised shape → discard.
      // The file will be fully re-ingested on this run, then section-level takes over.
    }
    return migrated
  } catch {
    console.warn('Warning: ingest cache corrupted, starting fresh')
    return {}
  }
}

export function saveCache(cache: FileCache) {
  fs.mkdirSync(path.dirname(CACHE_PATH), { recursive: true })
  fs.writeFileSync(CACHE_PATH, JSON.stringify(cache, null, 2), 'utf-8')
}

export function clearCache() {
  if (fs.existsSync(CACHE_PATH)) {
    fs.unlinkSync(CACHE_PATH)
    console.log('Ingest cache cleared.')
  }
}

/**
 * MD5 of content — used only for change detection, not security.
 *
 * PROD NOTE — MD5 is fine here. You're not protecting against adversarial
 *   input, just detecting accidental changes. SHA-256 works too but is slower
 *   for no benefit in this context.
 */
export function hashContent(content: string): string {
  return crypto.createHash('md5').update(content).digest('hex')
}
