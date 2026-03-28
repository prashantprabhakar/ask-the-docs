/**
 * Qdrant vector store client.
 *
 * LEARN — What Qdrant gives us over the flat JSON store:
 *   - Vectors live on disk, indexed with HNSW — search is O(log n) not O(n)
 *   - Payload fields can be indexed for fast filtering (used in deleteBySource)
 *   - Native hybrid search (dense + sparse) with Reciprocal Rank Fusion
 *   - Concurrent reads/writes without locking a single file
 *
 * PROD NOTE — In a real service you would also:
 *   - Use the gRPC client for persistent connections (REST opens a new HTTP
 *     connection per request — fine for scripts, not for high-QPS APIs)
 *   - Add a circuit breaker so one slow Qdrant node doesn't cascade
 *   - Emit metrics (latency, error rate) per operation to your observability stack
 *   - Configure TLS + API key auth (Qdrant supports both)
 *   - Run Qdrant in a cluster (3+ nodes) for high availability
 */

import { QdrantClient } from '@qdrant/js-client-rest'
import { v5 as uuidv5 } from 'uuid'
import pRetry, { AbortError } from 'p-retry'
import { logger } from '../logger'
import type { SparseVector } from '../rag/sparse-encoder'

// ─── Config ──────────────────────────────────────────────────────────────────

const COLLECTION = 'ask-the-docs'

/**
 * Named vector keys — Qdrant stores multiple vectors per point under names.
 * Query and upsert reference these names to address the right vector space.
 */
const DENSE_VECTOR = 'dense'
const SPARSE_VECTOR = 'sparse'

/**
 * Must match the output dimension of your embedding model.
 *   nomic-embed-text  (Ollama default) → 768
 *   mxbai-embed-large (Ollama)         → 1024
 *   text-embedding-3-small (OpenAI)    → 1536
 *   text-embedding-3-large (OpenAI)    → 3072
 *
 * PROD NOTE — Larger dimensions = more accurate but slower search and more RAM.
 *   Benchmark on your domain before picking. OpenAI's text-embedding-3-small at
 *   1536 dims outperforms ada-002 on most retrieval benchmarks. For cost/quality
 *   tradeoff, matryoshka truncation to 512 dims is a common production choice.
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
   *   timeout: 30_000                       (ms; default can be too short for large upserts)
   */
})

// ─── Types ────────────────────────────────────────────────────────────────────

export type DocType = 'guide' | 'api-reference' | 'error'

export interface DocChunk {
  /**
   * Deterministic UUID v5 derived from source path + content.
   * Same content = same ID → safe to re-run ingest (upsert, not duplicate).
   * Changed content = new ID → old chunk is orphaned and cleaned up by
   * deleteChunksBySource before the next upsert.
   */
  id: string
  content: string
  embedding: number[]         // dense vector from the embedding model
  sparseVector: SparseVector  // sparse vector for BM25-style keyword matching
  metadata: {
    source: string        // relative file path — used for filtering + citations
    title: string         // document title (from first # heading)
    sectionTitle: string  // full heading path, e.g. "App Router > Layouts > Nested Layouts"
    chunkIndex: number    // position within the section (0 if section fits in one chunk)
    lastModified: string  // ISO 8601 — file mtime at ingest time, surfaces doc freshness
    docType: DocType      // inferred from path — enables filtered search by doc type
    url: string           // canonical nextjs.org URL for this section (for citations)
  }
}

/**
 * Caller-facing filter — subset of what the user can restrict search to.
 * Converted to a Qdrant payload filter inside similaritySearch.
 *
 * PROD NOTE — Extend this as you add more indexed payload fields.
 *   Common additions: `sinceDate` (filter by lastModified), `source` (specific file).
 */
export interface ChunkFilter {
  docType?: DocType
}

// ─── ID Generation ────────────────────────────────────────────────────────────

/**
 * Generate a deterministic UUID v5 for a chunk.
 *
 * UUID v5 = SHA-1 hash of (namespace + name), formatted as a UUID.
 * Same inputs → same UUID. No randomness.
 *
 * Source is included so identical content in two different files gets
 * different IDs (preserving their separate metadata).
 */
export function chunkId(source: string, content: string): string {
  return uuidv5(`${source}::${content}`, CHUNK_NAMESPACE)
}

// ─── Collection Bootstrap ─────────────────────────────────────────────────────

let collectionReady = false

/**
 * Idempotent collection setup — runs once per process.
 *
 * Creates the collection with both dense and sparse vector configs,
 * then ensures the payload index on metadata.source exists.
 *
 * NOTE — If you already have a collection from a previous version (dense-only),
 *   you must run `npm run ingest:full` to drop and recreate it with sparse
 *   vector support. Qdrant does not allow adding new vector types to an
 *   existing collection.
 *
 * PROD NOTE — In production, collection setup is a versioned migration script,
 *   not part of application startup. Schema changes go through review, run
 *   against staging first, and use collection aliases for zero-downtime swaps.
 */
async function ensureCollection() {
  if (collectionReady) return

  const collections = await client.getCollections()
  const exists = collections.collections.some((c) => c.name === COLLECTION)

  if (!exists) {
    await client.createCollection(COLLECTION, {
      /**
       * Named vectors — each point stores multiple vectors under different names.
       * This lets Qdrant search each space independently then fuse the results.
       */
      vectors: {
        [DENSE_VECTOR]: {
          size: VECTOR_SIZE,
          distance: 'Cosine',
          /**
           * PROD NOTE — HNSW tuning:
           *   on_disk: true                → store on disk (cheaper for large collections)
           *   hnsw_config.m: 16–32         → edges per node, higher = better recall
           *   hnsw_config.ef_construct: 128 → index build quality
           *   quantization_config          → int8 compression, up to 32× smaller
           */
        },
      },
      sparse_vectors: {
        /**
         * Sparse vector config for the keyword index.
         *
         * LEARN — Sparse vectors work differently from dense:
         *   - Most values are zero (only terms present in the text have a value)
         *   - Qdrant uses an inverted index (like Elasticsearch) not HNSW
         *   - Dot product is used, not cosine — sparse vectors are not normalized
         *
         * PROD NOTE — For production sparse vectors, use SPLADE via Qdrant's
         *   FastEmbed instead of our custom TF encoder. SPLADE learns which terms
         *   are important across the corpus, producing much better sparse vectors.
         *   https://qdrant.tech/documentation/fastembed/fastembed-sparse/
         */
        [SPARSE_VECTOR]: {
          index: {
            on_disk: false, // keep sparse index in RAM for fast lookups
          },
        },
      },
    })

    process.stdout.write(`Created Qdrant collection "${COLLECTION}" (dense dim=${VECTOR_SIZE} + sparse)\n`)

    await client.createPayloadIndex(COLLECTION, {
      field_name: 'metadata.source',
      field_schema: 'keyword',
    })

    /**
     * Index metadata.docType so filtered searches (e.g. "only guide chunks")
     * hit an inverted index instead of scanning every point's payload.
     *
     * PROD NOTE — Add an index for every field you filter on. Without an index
     *   Qdrant falls back to a full payload scan — O(N) per query.
     */
    await client.createPayloadIndex(COLLECTION, {
      field_name: 'metadata.docType',
      field_schema: 'keyword',
    })

    process.stdout.write('Created payload indexes on metadata.source, metadata.docType\n')
  }

  collectionReady = true
}

// ─── Retry Helper ─────────────────────────────────────────────────────────────

/**
 * Returns true for HTTP 4xx errors — these are client errors (bad request,
 * not found, unauthorized) that will not succeed on retry. We abort immediately
 * so we don't waste retries on errors the server will always reject.
 *
 * The Qdrant REST client surfaces errors as plain Error objects whose message
 * starts with the HTTP status code (e.g. "400 Bad Request").
 */
function isClientError(err: unknown): boolean {
  return err instanceof Error && /^4\d\d\b/.test(err.message)
}

/**
 * Retry a Qdrant operation with exponential backoff + jitter.
 *
 * - Retryable: 5xx, network errors, timeouts
 * - Non-retryable (AbortError): 4xx — server will reject these regardless
 * - Jitter (randomize: true): spreads retries so a thundering herd of requests
 *   hitting a recovering Qdrant node don't all retry at the same instant
 * - onFailedAttempt: logs each retry with attempt number and remaining retries
 *   so slow degradation is visible in structured logs before the final failure
 */
async function withRetry<T>(fn: () => Promise<T>): Promise<T> {
  return pRetry(
    async () => {
      try {
        return await fn()
      } catch (err) {
        if (isClientError(err)) throw new AbortError(err as Error)
        throw err
      }
    },
    {
      retries: 3,
      minTimeout: 500,
      randomize: true,
      onFailedAttempt: (err) => {
        logger.warn({
          event: 'qdrant_retry',
          attempt: err.attemptNumber,
          retriesLeft: err.retriesLeft,
          error: String(err),
        })
      },
    }
  )
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Add or update chunks in Qdrant with both dense and sparse vectors.
 *
 * `wait: true` blocks until Qdrant confirms the write is indexed.
 * Without it, a subsequent search might not see the new chunks yet.
 *
 * PROD NOTE — For high-throughput pipelines, set `wait: false` and poll
 *   collection status instead. `wait: true` is correct for sequential scripts.
 */
export async function upsertChunks(chunks: DocChunk[]): Promise<void> {
  await ensureCollection()

  await withRetry(() =>
    client.upsert(COLLECTION, {
      wait: true,
      points: chunks.map((chunk) => ({
        id: chunk.id,
        /**
         * Named vectors — each goes into its own indexed space.
         * Qdrant stores and indexes them independently.
         * `vector` (singular) accepts a map when multiple vector types are configured.
         */
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        vector: {
          [DENSE_VECTOR]: chunk.embedding,
          [SPARSE_VECTOR]: chunk.sparseVector,
        } as any, // Qdrant REST client types don't fully model the named-vector map with mixed types
        payload: {
          content: chunk.content,
          metadata: chunk.metadata,
        },
      })),
    })
  )

  // Batch upsert confirmed — individual batch logs suppressed to keep ingest output clean
}

/**
 * Hybrid search — combines dense (semantic) and sparse (keyword) retrieval
 * with Reciprocal Rank Fusion (RRF).
 *
 * HOW RRF WORKS:
 *   1. Dense search returns top-20 candidates ranked by cosine similarity
 *   2. Sparse search returns top-20 candidates ranked by keyword match score
 *   3. RRF merges both ranked lists:
 *        score(doc) = Σ 1 / (rank_in_list + 60)
 *      A document that ranks #1 in both lists scores highest.
 *      A document that only appears in one list still gets a partial score.
 *   4. Final top-K returned from the fused ranking
 *
 * WHY PREFETCH WITH LARGER LIMITS?
 *   We prefetch 20 from each search so RRF has enough candidates to fuse.
 *   If we only fetched 5 from each, a relevant document might fall outside
 *   the top-5 of one search and never make it into the fused result.
 *   The final `limit: topK` cuts down to what the caller actually needs.
 *
 * PROD NOTE — The prefetch limit (20) is a tunable. Higher = better recall,
 *   more work. In production, benchmark this against your query set. A
 *   re-ranker (cross-encoder) after retrieval can further improve precision.
 */
export async function similaritySearch(
  queryEmbedding: number[],
  querySparse: SparseVector,
  topK = 5,
  filter?: ChunkFilter
): Promise<{ content: string; metadata: DocChunk['metadata']; score: number }[]> {
  await ensureCollection()

  const PREFETCH_LIMIT = 20

  /**
   * Convert the caller-facing ChunkFilter into Qdrant's payload filter format.
   *
   * LEARN — Qdrant payload filters:
   *   Applied BEFORE vector scoring — Qdrant only scores points that pass the
   *   filter. This makes filtered search cheaper, not more expensive: fewer
   *   points to score, and the indexed keyword lookup is O(1).
   *
   *   `must` = AND (all conditions must match)
   *   `should` = OR (any condition must match)
   *   `must_not` = NOT
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const qdrantFilter: any = filter?.docType
    ? { must: [{ key: 'metadata.docType', match: { value: filter.docType } }] }
    : undefined

  const results = await withRetry(() =>
    client.query(COLLECTION, {
      /**
       * Prefetch candidates from both vector spaces independently.
       * These are not the final results — they feed into the fusion step.
       */
      prefetch: [
        {
          query: queryEmbedding,
          using: DENSE_VECTOR,
          limit: PREFETCH_LIMIT,
          ...(qdrantFilter && { filter: qdrantFilter }),
        },
        {
          query: querySparse,
          using: SPARSE_VECTOR,
          limit: PREFETCH_LIMIT,
          ...(qdrantFilter && { filter: qdrantFilter }),
        },
      ],
      /**
       * Fuse the two candidate lists with Reciprocal Rank Fusion.
       * RRF is parameter-free and robust — no weights to tune.
       *
       * PROD NOTE — Qdrant also supports `dbsf` (Distribution-Based Score Fusion)
       *   which normalizes scores before merging. RRF is generally preferred
       *   because it's rank-based and less sensitive to score distribution differences
       *   between dense and sparse searches.
       */
      query: { fusion: 'rrf' },
      limit: topK,
      with_payload: true,
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
 * The payload index on metadata.source makes this a fast indexed lookup.
 */
export async function deleteChunksBySource(source: string): Promise<void> {
  await ensureCollection()

  await withRetry(() =>
    client.delete(COLLECTION, {
      wait: true,
      filter: {
        must: [{ key: 'metadata.source', match: { value: source } }],
      },
    })
  )

  // Deletion confirmed — logged by the caller in the per-file progress line
}

/**
 * PROD NOTE — Wire this into your /api/health readiness probe.
 *   Kubernetes calls it before routing traffic to a pod.
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
 * Scroll through all chunks in the collection and return their text content.
 *
 * Used by the ingester to build the IDF table after each ingest run.
 * Fetches in pages of 100 to avoid loading the entire corpus into memory
 * at once — Qdrant's scroll API returns a `next_page_offset` cursor that
 * is null when the last page has been reached.
 *
 * PROD NOTE — For very large collections (millions of points), this scroll
 *   is still O(N). At that scale, maintain term counts incrementally in a
 *   separate store (Redis / Postgres) and update them during ingest instead
 *   of scanning the full corpus each time.
 */
export async function scrollAllChunkTexts(): Promise<string[]> {
  await ensureCollection()

  const texts: string[] = []
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let offset: any = undefined

  do {
    const result = await withRetry(() =>
      client.scroll(COLLECTION, {
        limit: 100,
        ...(offset !== undefined && { offset }),
        with_payload: ['content'],
        with_vector: false,
      })
    )

    for (const point of result.points) {
      const content = point.payload?.content
      if (typeof content === 'string') texts.push(content)
    }

    offset = result.next_page_offset ?? null
  } while (offset !== null)

  return texts
}

/**
 * Drop the collection entirely. Used by `npm run ingest:full`.
 *
 * PROD NOTE — Never expose this as an API endpoint. Treat collection drops
 *   as migrations: reviewed, staged, and never triggered at runtime.
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
