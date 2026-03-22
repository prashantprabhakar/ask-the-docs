import 'dotenv/config'
import path from 'path'
import { ingestDocuments } from '../lib/rag/ingester'
import { getChunkCount, clearStore } from '../lib/vectordb'

const docsDir = path.join(process.cwd(), 'data', 'docs')
const shouldClear = process.argv.includes('--clear')

async function main() {
  if (shouldClear) {
    console.log('Clearing existing store...')
    await clearStore()
  }

  await ingestDocuments(docsDir)

  const count = await getChunkCount()
  console.log(`\nVector DB now contains ${count} total chunks.`)
}

main().catch((err) => {
  console.error('Ingestion failed:', err)
  process.exit(1)
})
