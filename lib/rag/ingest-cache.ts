/**
 * File-hash cache for incremental ingestion.
 *
 * Stores a map of { source → MD5(fileContent) } so the ingester can skip
 * files that haven't changed since the last run.
 *
 * PROD NOTE — This is a local JSON file, which is fine for a single-machine
 *   ingest script. In a distributed pipeline (multiple workers ingesting in
 *   parallel), you'd store this in a database (Postgres, Redis) so all workers
 *   share the same view of what's been processed. A simple `ingest_state` table
 *   with (source, content_hash, ingested_at) is the typical approach.
 */

import fs from 'fs'
import path from 'path'
import crypto from 'crypto'

const CACHE_PATH = path.join(process.cwd(), 'data', 'ingest-cache.json')

// Map of source (relative path) → MD5 hash of file content
type CacheMap = Record<string, string>

export function loadCache(): CacheMap {
  if (!fs.existsSync(CACHE_PATH)) return {}
  try {
    return JSON.parse(fs.readFileSync(CACHE_PATH, 'utf-8')) as CacheMap
  } catch {
    // Corrupted cache — treat as empty and re-ingest everything
    console.warn('Warning: ingest cache corrupted, starting fresh')
    return {}
  }
}

export function saveCache(cache: CacheMap) {
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
 * MD5 of file content — used only for change detection, not security.
 *
 * PROD NOTE — MD5 is fine here. You're not protecting against adversarial
 *   input, just detecting accidental changes. SHA-256 works too but is slower
 *   for no benefit in this context.
 */
export function hashFileContent(content: string): string {
  return crypto.createHash('md5').update(content).digest('hex')
}
