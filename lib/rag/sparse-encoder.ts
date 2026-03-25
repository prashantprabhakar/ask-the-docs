/**
 * Sparse vector encoder for hybrid search.
 *
 * Produces a sparse vector { indices, values } from text, where:
 *   - indices = integer IDs representing unique terms
 *   - values  = BM25-weighted scores (TF × IDF) when an IDF table is
 *               available, or plain TF weights as a fallback
 *
 * This is stored alongside the dense embedding in Qdrant and used in hybrid
 * search to boost results that contain the exact terms from the query.
 *
 * WHY HYBRID SEARCH?
 *   Dense vectors capture *meaning* — good for conceptual queries.
 *   Sparse vectors capture *exact terms* — good for API names, error codes,
 *   config keys. "What does useRouter do?" benefits from both: the dense
 *   vector finds semantically related chunks, the sparse vector ensures
 *   chunks that literally mention "useRouter" score higher.
 *   Qdrant fuses both with Reciprocal Rank Fusion (RRF).
 *
 * BM25 VS TF:
 *   TF alone gives "component" and "useRouter" the same weight if they
 *   appear the same number of times in a chunk. BM25 adds IDF — "component"
 *   appears in hundreds of chunks so its IDF is near zero. "useRouter"
 *   appears in only a handful, so it gets a high IDF multiplier. The sparse
 *   search now correctly prioritises chunks that are specifically about
 *   what the user typed.
 *
 *   IDF is applied at query time (asymmetric BM25). Document sparse vectors
 *   in Qdrant use plain TF. See idf-table.ts for the reasoning.
 *
 * PROD NOTE — Production options, roughly in order of quality:
 *
 *   1. SPLADE (best) — a neural model that produces learned sparse vectors.
 *      Much better recall than BM25. Available via Qdrant's FastEmbed (Python)
 *      or as a HuggingFace model. This is what Qdrant recommends for production.
 *      https://qdrant.tech/articles/sparse-vectors/
 *
 *   2. BM25 with corpus IDF (this implementation) — classic, interpretable.
 *      Requires a two-pass ingest: collect term counts → compute IDF → encode.
 *
 *   3. TF-only — simplest, no corpus stats, good enough to demonstrate hybrid
 *      search. Used as a fallback when the IDF table has not been built yet.
 */

import { loadIdfTable, computeIdf, buildIdfTable } from './idf-table'
import type { IdfTable } from './idf-table'

// ─── Stopwords ────────────────────────────────────────────────────────────────

/**
 * Common English stopwords to skip.
 * Skipping them reduces noise and keeps sparse vectors compact.
 *
 * PROD NOTE — For technical docs, keep domain-specific short words that
 *   would normally be stopwords but carry meaning: "on", "off", "get", "set".
 *   Tune this list for your domain.
 */
const STOPWORDS = new Set([
  'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
  'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
  'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
  'those', 'it', 'its', 'as', 'if', 'then', 'than', 'so', 'up', 'out',
  'about', 'into', 'through', 'during', 'before', 'after', 'above',
  'below', 'between', 'each', 'all', 'both', 'more', 'also', 'when',
  'where', 'which', 'who', 'how', 'what', 'not', 'no', 'nor', 'very',
])

// ─── Tokenizer ────────────────────────────────────────────────────────────────

/**
 * Tokenize text into clean lowercase terms.
 *
 * Exported so the ingester can reuse the same tokenization when building
 * the IDF table — consistency is critical. If the IDF table is built with
 * different tokenization than the encoder uses, term lookups will miss.
 *
 * PROD NOTE — Production tokenizers also handle:
 *   - Stemming / lemmatization ("running" → "run") to match more variants
 *   - Subword tokenization for code identifiers ("useRouter" → ["use", "router"])
 *   - Language detection for multilingual corpora
 */
export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/```[\s\S]*?```/g, ' ') // strip code blocks — often too noisy
    .replace(/`[^`]+`/g, ' ')        // strip inline code
    .replace(/[^a-z0-9\s]/g, ' ')   // strip punctuation
    .split(/\s+/)
    .filter((t) => t.length > 1 && !STOPWORDS.has(t))
}

// ─── Term → Index Mapping ─────────────────────────────────────────────────────

/**
 * Map a term to a stable integer index using DJB2 hash.
 *
 * Qdrant sparse vector indices are arbitrary non-negative integers — they just
 * need to be consistent: the same term must always map to the same index so
 * that query and document sparse vectors share the same "vocabulary space".
 *
 * VOCAB_SIZE controls the index space. Larger = fewer hash collisions but
 * slightly more memory. 500,000 is plenty for English technical docs.
 *
 * PROD NOTE — Hash collisions (two different terms mapping to the same index)
 *   are rare at this vocab size and don't cause incorrect results — just
 *   slightly inflated scores for those terms. With SPLADE or a fixed
 *   vocabulary, collisions are zero.
 */
const VOCAB_SIZE = 500_000

function termIndex(term: string): number {
  let hash = 5381
  for (let i = 0; i < term.length; i++) {
    hash = ((hash << 5) + hash) + term.charCodeAt(i)
    hash = hash & hash // keep as 32-bit integer
  }
  return Math.abs(hash) % VOCAB_SIZE
}

// ─── IDF Table (lazy singleton) ───────────────────────────────────────────────

/**
 * Lazy-loaded IDF table.
 *
 * `undefined` = not yet attempted to load.
 * `null`      = attempted but table does not exist yet (first ingest run).
 * `IdfTable`  = loaded and ready.
 *
 * The ingester calls reloadIdfTable() after building a fresh table so the
 * next query in the same process sees the updated weights immediately.
 */
let _idfTable: IdfTable | null | undefined = undefined

function getIdfTable(): IdfTable | null {
  if (_idfTable === undefined) {
    _idfTable = loadIdfTable()
    if (_idfTable) {
      console.log(`[sparse-encoder] IDF table loaded (${_idfTable.totalDocs} docs, ${Object.keys(_idfTable.termDf).length} terms)`)
    }
  }
  return _idfTable
}

/**
 * Force a reload of the IDF table from disk.
 * Called by the ingester after it rebuilds the table so the new weights
 * take effect immediately without restarting the process.
 */
export function reloadIdfTable(): void {
  _idfTable = loadIdfTable()
}

// ─── Public API ───────────────────────────────────────────────────────────────

export interface SparseVector {
  indices: number[]
  values: number[]
}

/**
 * Encode text as a BM25-weighted sparse vector (or TF-only if no IDF table).
 *
 * Each unique term in the text becomes one entry in the sparse vector:
 *   index = termIndex(term)        — stable integer ID for this term
 *   value = TF(term) × IDF(term)  — BM25 weight (or TF if no IDF table)
 *
 * Terms with a BM25 weight of zero (present in every chunk → IDF = 0)
 * are excluded from the vector — they carry no discriminating signal.
 *
 * WHEN IDF IS NOT AVAILABLE (first ingest run):
 *   Falls back to plain TF: value = count / total_tokens.
 *   The IDF table is built at the end of the ingest run, so subsequent
 *   runs and all query-time calls will use full BM25 weights.
 */
export function encodeSparse(text: string): SparseVector {
  const tokens = tokenize(text)
  if (tokens.length === 0) return { indices: [], values: [] }

  // Count raw term frequencies
  const tfCounts = new Map<string, number>()
  for (const token of tokens) {
    tfCounts.set(token, (tfCounts.get(token) ?? 0) + 1)
  }

  const total = tokens.length
  const idf = getIdfTable()

  const indices: number[] = []
  const values: number[] = []

  for (const [term, count] of tfCounts) {
    const tf = count / total

    let weight: number
    if (idf) {
      // BM25: TF × IDF. Terms with IDF=0 (in every chunk) are skipped.
      const idfScore = computeIdf(term, idf)
      if (idfScore === 0) continue
      weight = tf * idfScore
    } else {
      // Fallback: plain TF (first run, no IDF table yet)
      weight = tf
    }

    indices.push(termIndex(term))
    values.push(weight)
  }

  return { indices, values }
}

// ─── IDF Table Builder (used by ingester) ─────────────────────────────────────

export { buildIdfTable, saveIdfTable } from './idf-table'
