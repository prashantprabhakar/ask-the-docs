/**
 * Sparse vector encoder for hybrid search.
 *
 * Produces a sparse vector { indices, values } from text, where:
 *   - indices = integer IDs representing unique terms
 *   - values  = normalized term frequency (TF) weights
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
 * PROD NOTE — This is simplified TF-based sparse encoding, not true BM25.
 *   True BM25 requires IDF (Inverse Document Frequency) scores computed from
 *   the full corpus, which conflicts with incremental ingestion unless you
 *   maintain a running IDF table.
 *
 *   Production options, roughly in order of quality:
 *
 *   1. SPLADE (best) — a neural model that produces learned sparse vectors.
 *      Much better recall than BM25. Available via Qdrant's FastEmbed (Python)
 *      or as a HuggingFace model. This is what Qdrant recommends for production.
 *      https://qdrant.tech/articles/sparse-vectors/
 *
 *   2. BM25 with corpus IDF — classic, interpretable, no ML needed.
 *      Requires a two-pass ingest (collect term counts → compute IDF → encode).
 *      Libraries: `wink-bm25-text-search`, `okapibm25`.
 *
 *   3. TF-only (this implementation) — simplest, no corpus stats needed,
 *      good enough to demonstrate hybrid search and improve keyword matching.
 */

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
 * Splits on whitespace and punctuation, strips Markdown syntax characters,
 * removes short tokens and stopwords.
 *
 * PROD NOTE — Production tokenizers also handle:
 *   - Stemming / lemmatization ("running" → "run") to match more variants
 *   - Subword tokenization for code identifiers ("useRouter" → ["use", "router"])
 *   - Language detection for multilingual corpora
 */
function tokenize(text: string): string[] {
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

// ─── Public API ───────────────────────────────────────────────────────────────

export interface SparseVector {
  indices: number[]
  values: number[]
}

/**
 * Encode text as a TF-weighted sparse vector.
 *
 * Each unique term in the text becomes one entry in the sparse vector:
 *   index = termIndex(term)   — stable integer ID for this term
 *   value = count / total     — normalized term frequency (0 to 1)
 *
 * Higher TF → term appears more in this chunk → higher weight in sparse search.
 */
export function encodeSparse(text: string): SparseVector {
  const tokens = tokenize(text)
  if (tokens.length === 0) return { indices: [], values: [] }

  // Count raw term frequencies
  const tf = new Map<string, number>()
  for (const token of tokens) {
    tf.set(token, (tf.get(token) ?? 0) + 1)
  }

  const total = tokens.length
  const indices: number[] = []
  const values: number[] = []

  for (const [term, count] of tf) {
    indices.push(termIndex(term))
    values.push(count / total) // normalized TF
  }

  return { indices, values }
}
