/**
 * IDF table for BM25-weighted sparse encoding.
 *
 * Stores { term → document frequency } across the full chunk corpus,
 * plus the total chunk count N. Used by the sparse encoder to weight
 * query terms by how rare they are across the corpus.
 *
 * Built at the end of every ingest run by scrolling all chunks from Qdrant.
 * Loaded once at query time and cached for the process lifetime.
 *
 * WHY ONLY QUERY-SIDE IDF?
 *   Document sparse vectors in Qdrant are already stored as TF weights.
 *   Re-upserting every point after each IDF rebuild would be expensive
 *   and adds complexity. Applying IDF only at query time (asymmetric BM25)
 *   is still a significant improvement: common query terms like "component"
 *   get down-weighted so rare terms like "useRouter" drive the score.
 *
 * PROD NOTE — In production you'd go further:
 *   - Re-encode document sparse vectors with BM25 after full ingests
 *   - Use SPLADE (neural sparse model) instead of hand-crafted BM25 —
 *     SPLADE learns which terms matter for retrieval, not just frequency
 */

import fs from 'fs'
import path from 'path'

const IDF_TABLE_PATH = path.join(process.cwd(), 'data', 'idf-table.json')

export interface IdfTable {
  /** How many chunks each term appears in (document frequency) */
  termDf: Record<string, number>
  /** Total number of chunks in the corpus at build time */
  totalDocs: number
}

// ─── Build ─────────────────────────────────────────────────────────────────

/**
 * Build an IDF table from an array of per-chunk term sets.
 *
 * Each Set contains the unique terms from one chunk — we use a Set (not
 * a list of all tokens) because IDF counts *chunks containing the term*,
 * not total occurrences across the corpus.
 *
 * Example:
 *   chunk A: { "userouter", "router", "returns" }
 *   chunk B: { "router", "component", "render" }
 *   chunk C: { "component", "props", "render" }
 *
 *   termDf: { userouter: 1, router: 2, returns: 1, component: 2, render: 2, props: 1 }
 *   totalDocs: 3
 */
export function buildIdfTable(termSets: Set<string>[]): IdfTable {
  const termDf: Record<string, number> = {}

  for (const terms of termSets) {
    for (const term of terms) {
      termDf[term] = (termDf[term] ?? 0) + 1
    }
  }

  return { termDf, totalDocs: termSets.length }
}

// ─── IDF Formula ───────────────────────────────────────────────────────────

/**
 * BM25 IDF formula (Robertson-Spärck Jones variant with smoothing):
 *
 *   IDF(term) = log((N - df + 0.5) / (df + 0.5) + 1)
 *
 * The +1 inside the log ensures the result is always ≥ 0 even when the
 * term appears in every chunk (df = N → numerator and denominator equal
 * → log(1) = 0).
 *
 * Concrete examples with N = 1000 chunks:
 *   "useRouter" in 3 chunks  → log((997.5) / (3.5) + 1)  ≈ 5.7  ← very rare, high signal
 *   "component" in 600 chunks → log((400.5) / (600.5) + 1) ≈ 0.5  ← common, low signal
 *   "the"       in 1000 chunks → log((0.5) / (1000.5) + 1) ≈ 0.0  ← everywhere, no signal
 *
 * Returns 0 for terms not in the table — unknown terms get no IDF boost,
 * so their raw TF score is used instead.
 */
export function computeIdf(term: string, table: IdfTable): number {
  const df = table.termDf[term]
  if (df === undefined) return 0
  const N = table.totalDocs
  return Math.log((N - df + 0.5) / (df + 0.5) + 1)
}

// ─── Persistence ───────────────────────────────────────────────────────────

export function loadIdfTable(): IdfTable | null {
  if (!fs.existsSync(IDF_TABLE_PATH)) return null
  try {
    return JSON.parse(fs.readFileSync(IDF_TABLE_PATH, 'utf-8')) as IdfTable
  } catch {
    console.warn('Warning: IDF table corrupted or unreadable, falling back to TF-only encoding')
    return null
  }
}

export function saveIdfTable(table: IdfTable): void {
  fs.mkdirSync(path.dirname(IDF_TABLE_PATH), { recursive: true })
  fs.writeFileSync(IDF_TABLE_PATH, JSON.stringify(table), 'utf-8')
}
