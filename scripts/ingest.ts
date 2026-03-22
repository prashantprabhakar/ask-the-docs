import 'dotenv/config'
import path from 'path'
import { ingestDocuments } from '../lib/rag/ingester'
import { getChunkCount, clearStore } from '../lib/vectordb'
import { clearCache } from '../lib/rag/ingest-cache'

const docsDir = path.join(process.cwd(), 'data', 'docs')

/**
 * --full   Re-embed every file even if unchanged. Use when switching
 *          embedding models (old vectors are incompatible with new ones).
 *          Also wipes the Qdrant collection and cache so you start clean.
 *
 * default  Incremental — only processes new or changed files.
 */
const isFull = process.argv.includes('--full')

async function main() {
  if (isFull) {
    console.log('Full re-ingest requested — clearing store and cache...')
    await clearStore()
    clearCache()
  }

  await ingestDocuments(docsDir, { full: isFull })

  const count = await getChunkCount()
  console.log(`\nVector store now contains ${count} total chunks.`)
}

main().catch((err) => {
  console.error('Ingestion failed:', err)
  process.exit(1)
})
