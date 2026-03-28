/**
 * Cross-encoder re-ranker for retrieved chunks.
 *
 * LEARN — Two-tower vs. cross-encoder:
 *   Bi-encoders (our dense embeddings) encode query and document independently,
 *   then compare via dot product. Fast at search time, but loses fine-grained
 *   interaction between query and document tokens.
 *
 *   A cross-encoder reads the query and document together as one sequence:
 *     [CLS] query [SEP] document [SEP]
 *   Every token in the query attends to every token in the document (full
 *   attention), so it captures nuanced relevance signals bi-encoders miss.
 *   The trade-off: you can't pre-compute document vectors, so you must score
 *   each (query, doc) pair at query time — O(N) forward passes, not O(1) lookup.
 *
 * STRATEGY — Two-stage retrieval:
 *   1. Retrieval (recall): Fast hybrid search returns top-20 candidates.
 *      Optimized for recall — we don't want to miss relevant chunks.
 *   2. Re-ranking (precision): Cross-encoder scores each of the 20 candidates
 *      against the query. Optimized for precision — top-5 after re-ranking
 *      are much more likely to be truly relevant.
 *
 * MODEL — ms-marco-MiniLM-L-6-v2:
 *   Trained on MS MARCO passage re-ranking task. Fine-tuned on (query, passage,
 *   label) triples where label = 1 if the passage answers the query.
 *   Outputs a single logit — higher = more relevant. No softmax needed.
 *
 * PROD NOTE — Alternatives to consider:
 *   - cross-encoder/ms-marco-electra-base: better quality, ~3× slower
 *   - Cohere Rerank API: hosted, no local GPU, pay-per-use
 *   - mixedbread-ai/mxbai-rerank-base-v1: strong open model, HuggingFace
 *   At scale, run the cross-encoder in a separate Python sidecar (FastAPI +
 *   sentence-transformers) and call it over HTTP to avoid blocking Node.js.
 */

import { reranker as rerankerConfig } from '../config'

// eslint-disable-next-line @typescript-eslint/no-require-imports
const { pipeline } = require('@xenova/transformers')

const MODEL = rerankerConfig.model

/**
 * Eager warm-up — start loading the model at module import time.
 *
 * We store a Promise, not the resolved value. This means:
 *   - Load begins the moment this module is first imported (server startup).
 *   - Every caller awaits the same Promise — no duplicate loads, no races.
 *   - By the time the first query arrives the model is likely already ready.
 *     If not, the query simply waits for the in-flight load to finish.
 *
 * PROD NOTE — In production, import this module during server startup (e.g.
 *   in instrumentation.ts or a Next.js middleware) so the warm-up begins before
 *   any request arrives, not on the first import triggered by a request.
 */
const rerankerPromise: Promise<Awaited<ReturnType<typeof pipeline>>> =
  pipeline('text-classification', MODEL)

function getReranker() {
  return rerankerPromise
}

export interface RerankCandidate {
  content: string
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  metadata: any
  score: number  // original retrieval score (RRF)
}

export interface RerankResult extends RerankCandidate {
  rerankScore: number  // cross-encoder score — use this for final ordering
}

/**
 * Re-rank candidates using a cross-encoder.
 *
 * Scores each (query, chunk) pair and returns candidates sorted by
 * cross-encoder score descending, truncated to topK.
 *
 * The original retrieval score is preserved on each result so callers
 * can log or display it alongside the re-rank score.
 */
export async function rerank(
  query: string,
  candidates: RerankCandidate[],
  topK: number
): Promise<RerankResult[]> {
  if (candidates.length === 0) return []

  const reranker = await getReranker()

  /**
   * The cross-encoder expects an array of [query, document] pairs.
   * It outputs a label ('true'/'false') and a score per pair.
   * We want the raw logit for the 'true' class (i.e. "this doc is relevant").
   *
   * LEARN — text-classification with ms-marco models returns:
   *   { label: 'true' | 'false', score: number }
   * where score is the probability of the predicted label (after softmax).
   * We use the score for 'true' as the relevance score.
   * If the model predicts 'false' with high confidence, the 'true' score is low.
   */
  const pairs = candidates.map((c) => [query, c.content])

  /**
   * Default top_k=1 returns the single highest-scoring label per pair.
   * ms-marco is a binary classifier: label is 'true' (relevant) or 'false'.
   * If the model predicts 'true', score IS the relevance probability.
   * If it predicts 'false', relevance is the complement: 1 - score.
   * Either way we get a value in [0, 1] that is comparable across pairs.
   */
  const outputs: { label: string; score: number }[] = await reranker(pairs)

  const scored: RerankResult[] = candidates.map((candidate, i) => {
    const { label, score } = outputs[i]
    const rerankScore = label === 'true' ? score : 1 - score
    return { ...candidate, rerankScore }
  })

  return scored.sort((a, b) => b.rerankScore - a.rerankScore).slice(0, topK)
}
