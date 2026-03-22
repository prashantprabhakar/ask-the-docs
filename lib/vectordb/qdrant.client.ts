/**
 * Qdrant vector store client.
 *
 * LEARN — What Qdrant gives us over the flat JSON store:
 *   - Vectors live on disk, indexed with HNSW — search is O(log n) not O(n)
 *   - Payload fields can be indexed for fast filtering (used in deleteBySource)
 *   - Native hybrid search (dense + sparse) — wired up in Step 4
 *   - Concurrent reads/writes without locking a single file
 *
 * PROD NOTE — In a real service you would also:
 *   - Use a connection pool (the REST client here opens a new HTTP connection
 *     per request; production uses the gRPC client for persistent connections)
 *   - Add a circuit breaker so one slow Qdrant node doesn't hang your API
 *   - Emit metrics (latency, error rate) per operation to your observability stack
 *   - Configure TLS + API key auth (Qdrant supports both)
 *   - Run Qdrant in a cluster (3+ nodes) for high availability
 */

import { QdrantClient } from '@qdrant/js-client-rest'
import { v5 as uuidv5 } from 'uuid'

// ─── Config ──────────────────────────────────────────────────────────────────

const COLLECTION = 'ask-the-docs'

/**
 * Must match the output dimension of your embedding model.
 *   nomic-embed-text  (Ollama default) → 768
 *   mxbai-embed-large (Ollama)         → 1024
 *   text-embedding-3-small (OpenAI)    → 1536
 *   text-embedding-3-large (OpenAI)    → 3072
 *
 * PROD NOTE — Larger dimensions = more accurate but slower search and more RAM.
 *   Benchmark on your domain before picking. OpenAI's text-embedding-3-small at
 *   1536 dims outperforms ada-002 at 1536 dims on most retrieval benchmarks.
 *   For a cost/quality tradeoff, text-embedding-3-small with matryoshka
 *   truncation to 512 dims is a common production choice.
 */
const VECTOR_SIZE = parseInt(process.env.EMBEDDING_DIM ?? '768')

/**
 * Fixed namespace UUID for this application.
 * Used by UUID v5 to generate deterministic chunk IDs.
 *
 * PROD NOTE — Generate this once with `uuidv4()` and commit it. Never change
 *   it — doing so invalidates every ID in the store and forces a full re-ingest.
 *   One namespace per logical dataset is a clean convention.
 */
const CHUNK_NAMESPACE = 'b3d2e1f0-4a5b-6c7d-8e9f-0a1b2c3d4e5f'

const client = new QdrantClient({
  url: process.env.QDRANT_URL ?? 'http://localhost:6333',
  /**
   * PROD NOTE — Also set:
   *   apiKey: process.env.QDRANT_API_KEY   (required for Qdrant Cloud)
   *   https: true                           (for Qdrant Cloud / TLS)
   *   timeout: 30_000                       (ms; default is sometimes too short for large upserts)
   */
})

// ─── Types ────────────────────────────────────────────────────────────────────

export interface DocChunk {
  /**
   * Deterministic UUID v5 derived from source path + content.
   * Same content = same ID → safe to re-run ingest (upsert, not duplicate).
   * Changed content = new ID → old chunk becomes an orphan (cleaned up by
   * deleteChunksBySource before upsert).
   *
   * PROD NOTE — Content-addressed IDs mean you get deduplication for free
   *   even across documents (two docs with identical sections share one chunk).
   *   That's usually desirable but watch for it if sections carry different
   *   metadata you need to preserve.
   */
  id: string
  content: string
  embedding: number[]
  metadata: {
    source: string   // relative file path — used for filtering + citations
    title: string    // document title
    chunkIndex: number
    /**
     * PROD NOTE — In production you'd add more metadata:
     *   section: string       (heading path, e.g. "Installation > macOS")
     *   lastModified: string  (ISO date — lets you surface freshness to the user)
     *   docType: string       ("guide" | "api-reference" | "changelog")
     *   url: string           (link to live docs for citation)
     * Richer metadata enables filtered search: "only search API reference docs"
     * without re-embedding anything.
     */
  }
}

// ─── ID Generation ────────────────────────────────────────────────────────────

/**
 * Generate a deterministic UUID v5 for a chunk.
 *
 * UUID v5 = SHA-1 hash of (namespace + name), formatted as a UUID.
 * The same inputs always produce the same UUID — no randomness.
 *
 * We include source in the name so identical content in two different files
 * gets different IDs (preserving their separate metadata).
 *
 * PROD NOTE — If you ever need cross-document deduplication (detect that two
 *   docs have the same section), drop the source from the name. But you'd
 *   then need a separate mapping to track which sources reference each chunk.
 */
export function chunkId(source: string, content: string): string {
  return uuidv5(`${source}::${content}`, CHUNK_NAMESPACE)
}

// ─── Collection Bootstrap ─────────────────────────────────────────────────────

let collectionReady = false

/**
 * Idempotent collection setup — runs once per process.
 *
 * Creates the collection if it doesn't exist, then ensures the payload index
 * on metadata.source exists (needed for deleteChunksBySource to be fast).
 *
 * PROD NOTE — In production, collection setup lives in a one-time migration
 *   script, not in application startup. Running DDL at request time is risky:
 *   - Concurrent startup race conditions
 *   - Unexpected latency on first request
 *   - You want schema changes to be explicit, reviewed, and versioned
 */
async function ensureCollection() {
  if (collectionReady) return

  const collections = await client.getCollections()
  const exists = collections.collections.some((c) => c.name === COLLECTION)

  if (!exists) {
    await client.createCollection(COLLECTION, {
      vectors: {
        size: VECTOR_SIZE,
        distance: 'Cosine',
        /**
         * PROD NOTE — HNSW parameters to tune for production:
         *
         *   on_disk: true
         *     Store vectors on disk instead of RAM. Slower but much cheaper.
         *     Use for large collections (>1M vectors) or memory-constrained hosts.
         *
         *   hnsw_config.m (default 16)
         *     Edges per node in the graph. Higher = better recall, more RAM/disk.
         *     16–32 is typical for production. Don't go below 8.
         *
         *   hnsw_config.ef_construct (default 100)
         *     Nodes visited during index build. Higher = better index quality,
         *     slower ingest. 100–200 is fine for most cases.
         *
         *   quantization_config
         *     Compress vectors to int8 or binary — up to 32× smaller.
         *     Minimal recall loss with rescoring. Almost always worth it in prod.
         *
         * Example prod config:
         *   on_disk: true,
         *   hnsw_config: { m: 16, ef_construct: 128 },
         *   quantization_config: { scalar: { type: 'int8', quantile: 0.99, always_ram: true } }
         */
      },
    })
    console.log(`Created Qdrant collection "${COLLECTION}" (dim=${VECTOR_SIZE}, distance=Cosine)`)

    /**
     * Create a keyword payload index on metadata.source.
     *
     * Without this, deleteChunksBySource scans every point to find matches.
     * With this index, Qdrant looks up the source in a hash map — O(1).
     *
     * PROD NOTE — Index every field you filter on. Common ones:
     *   metadata.source   → for per-file delete/update
     *   metadata.docType  → for filtered search ("search only API reference")
     *   metadata.lastModified → for freshness filtering
     */
    await client.createPayloadIndex(COLLECTION, {
      field_name: 'metadata.source',
      field_schema: 'keyword',
    })
    console.log('Created payload index on metadata.source')
  }

  collectionReady = true
}

// ─── Retry Helper ─────────────────────────────────────────────────────────────

/**
 * Retry a Qdrant operation with exponential backoff.
 *
 * Qdrant is generally reliable, but transient failures happen:
 *   - Network blips between your app and the Qdrant container
 *   - Qdrant briefly unavailable during a rolling restart
 *   - Rate limiting on Qdrant Cloud
 *
 * PROD NOTE — This is a minimal implementation. Production retry logic should:
 *   - Distinguish retryable errors (503, timeout) from non-retryable (400 bad request)
 *   - Add jitter to the backoff to avoid thundering herd
 *   - Report retry attempts to your metrics/alerting system
 *   - Respect a global deadline, not just per-attempt timeouts
 *
 * Libraries like `async-retry` or `p-retry` handle this more robustly.
 */
async function withRetry<T>(
  fn: () => Promise<T>,
  retries = 3,
  delayMs = 500
): Promise<T> {
  let lastError: unknown
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      return await fn()
    } catch (err) {
      lastError = err
      if (attempt < retries) {
        const wait = delayMs * 2 ** (attempt - 1) // 500ms, 1000ms, 2000ms
        console.warn(`Qdrant attempt ${attempt} failed, retrying in ${wait}ms...`)
        await new Promise((r) => setTimeout(r, wait))
      }
    }
  }
  throw lastError
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Add or update chunks in Qdrant.
 *
 * `wait: true` blocks until Qdrant confirms the write is persisted and indexed.
 * Without it, a subsequent similaritySearch might not see the new chunks yet.
 *
 * PROD NOTE — For high-throughput ingest pipelines, set `wait: false` and
 *   poll collection status instead. `wait: true` is correct for scripts like
 *   ours that ingest sequentially.
 */
export async function upsertChunks(chunks: DocChunk[]): Promise<void> {
  await ensureCollection()

  await withRetry(() =>
    client.upsert(COLLECTION, {
      wait: true,
      points: chunks.map((chunk) => ({
        id: chunk.id, // already a proper UUID
        vector: chunk.embedding,
        payload: {
          content: chunk.content,
          metadata: chunk.metadata,
        },
      })),
    })
  )

  console.log(`Upserted ${chunks.length} chunks`)
}

/**
 * Find the top-K chunks most semantically similar to the query embedding.
 *
 * PROD NOTE — This is dense-only search. Step 4 replaces this with a hybrid
 *   query that combines dense + sparse (BM25) scores using Reciprocal Rank
 *   Fusion. Hybrid search significantly improves recall for exact-term queries
 *   (API names, config keys, error codes).
 */
export async function similaritySearch(
  queryEmbedding: number[],
  topK = 5
): Promise<{ content: string; metadata: DocChunk['metadata']; score: number }[]> {
  await ensureCollection()

  const results = await withRetry(() =>
    client.query(COLLECTION, {
      query: queryEmbedding,
      limit: topK,
      with_payload: true,
      /**
       * PROD NOTE — `params.ef` controls search quality vs speed:
       *   Higher ef = more nodes explored = better recall, slower.
       *   Default is 128. For high-recall production use, set to 256+.
       *   params: { hnsw_ef: 256 }
       *
       * You can also filter here before searching:
       *   filter: { must: [{ key: 'metadata.docType', match: { value: 'api-reference' } }] }
       * This narrows the search space without a separate filter pass.
       */
    })
  )

  return results.points.map((point) => ({
    content: point.payload!.content as string,
    metadata: point.payload!.metadata as DocChunk['metadata'],
    score: point.score,
  }))
}

/**
 * Delete all chunks belonging to a specific source file.
 *
 * Called before re-ingesting a changed file. The payload index on
 * metadata.source makes this a fast indexed lookup, not a full scan.
 *
 * PROD NOTE — In a pipeline, you'd batch these deletions across all changed
 *   files before starting any upserts — one roundtrip per file is fine at our
 *   scale but adds up when processing thousands of files concurrently.
 */
export async function deleteChunksBySource(source: string): Promise<void> {
  await ensureCollection()

  await withRetry(() =>
    client.delete(COLLECTION, {
      wait: true,
      filter: {
        must: [
          {
            key: 'metadata.source',
            match: { value: source },
          },
        ],
      },
    })
  )

  console.log(`Deleted existing chunks for: ${source}`)
}

/**
 * PROD NOTE — Health check belongs in your readiness probe (e.g. /api/health).
 *   Kubernetes calls it before routing traffic to a pod. If Qdrant is down,
 *   the pod reports not-ready and gets no traffic instead of returning 500s.
 */
export async function healthCheck(): Promise<boolean> {
  try {
    await client.getCollections()
    return true
  } catch {
    return false
  }
}

export async function getChunkCount(): Promise<number> {
  try {
    const info = await client.getCollection(COLLECTION)
    return info.points_count ?? 0
  } catch {
    return 0
  }
}

/**
 * Drop and recreate the collection. Used by `npm run ingest --clear`.
 *
 * PROD NOTE — Never expose this as an API endpoint. In production, clearing
 *   a collection is a data migration — it goes through a review process,
 *   runs against a staging environment first, and is never triggered at runtime.
 */
export async function clearStore(): Promise<void> {
  try {
    await client.deleteCollection(COLLECTION)
    collectionReady = false
    console.log(`Qdrant collection "${COLLECTION}" dropped.`)
  } catch {
    // Collection didn't exist — nothing to clear
  }
}
