/**
 * Eval pipeline — measures retrieval and answer quality against a golden dataset.
 *
 * Three metrics per question:
 *
 *   Context recall    (mechanical) — did the expected source(s) appear in the
 *                     retrieved chunks? Catches retrieval regressions: if you
 *                     change the chunker or embedder and the right doc stops
 *                     being found, this drops immediately.
 *
 *   Faithfulness      (LLM judge)  — does the answer stay within the retrieved
 *                     context? Catches hallucinations: the LLM should not add
 *                     facts that aren't in the chunks.
 *
 *   Answer relevance  (LLM judge)  — does the answer address what was asked?
 *                     Catches evasions: the LLM should not deflect or go
 *                     off-topic even when context is sparse.
 *
 * Usage:
 *   npm run eval             — run all cases, print report
 *   npm run eval -- --verbose — also show per-query pipeline logs
 *
 * Exit codes:
 *   0 — overall score >= 70%
 *   1 — overall score < 70% (useful for CI gating)
 *
 * PROD NOTE — This is a binary (pass/fail) judge. Production eval pipelines
 *   use continuous scores (0.0–1.0) and LLM-as-judge frameworks like RAGAS or
 *   DeepEval that run multiple judge calls per metric and average them to reduce
 *   variance. For a first eval, binary is easier to reason about and act on.
 */

import 'dotenv/config'
import fs from 'fs'
import path from 'path'
import { ragQuery } from '../lib/rag/retriever'
import { createLLMClient } from '../lib/llm/factory'

// ─── Logger suppression ───────────────────────────────────────────────────────

/**
 * The retriever uses a structured logger that writes JSON to console.log.
 * During eval we route those lines to stderr so the eval report on stdout
 * is clean. Pass --verbose to see full pipeline logs on stderr.
 */
const verbose = process.argv.includes('--verbose')
if (!verbose) {
  // Intercept structured log lines (start with '{') → stderr. Let anything
  // else (e.g. unexpected console.log from dependencies) pass through.
  const _log = console.log.bind(console)
  console.log = (...args: unknown[]) => {
    if (typeof args[0] === 'string' && args[0].startsWith('{')) {
      process.stderr.write(args[0] + '\n')
    } else {
      _log(...args)
    }
  }
}

// ─── Types ────────────────────────────────────────────────────────────────────

interface EvalCase {
  question: string
  expected_sources: string[]
  notes?: string
}

interface JudgeResult {
  score: 0 | 1
  reason: string
}

interface CaseResult {
  question: string
  contextRecall: { score: 0 | 1; found: string[]; missed: string[] }
  faithfulness: JudgeResult
  answerRelevance: JudgeResult
}

// ─── LLM Judge ────────────────────────────────────────────────────────────────

const llm = createLLMClient()

/**
 * Call the LLM with a scoring prompt and extract a { score, reason } JSON object.
 *
 * Fails closed (score=0) so a broken judge call does not silently inflate scores.
 * The "Judge failed" reason makes it visible in the report.
 *
 * PROD NOTE — LLM judge variance is real: the same prompt on the same input can
 *   return different scores across runs. Production evals run each judge 3–5
 *   times and take the majority vote. For a first pipeline, single-shot is fine.
 */
async function judge(prompt: string): Promise<JudgeResult> {
  try {
    const response = await llm.chat([
      {
        role: 'system',
        content:
          'You are an evaluation judge. Respond ONLY with a valid JSON object matching the requested schema. No markdown code fences, no explanation outside the JSON.',
      },
      { role: 'user', content: prompt },
    ])
    // Extract the first {...} block — guards against models that add prose anyway
    const match = response.match(/\{[\s\S]*?\}/)
    if (!match) return { score: 0, reason: 'Judge returned no JSON' }
    const parsed = JSON.parse(match[0])
    return {
      score: parsed.score === 1 ? 1 : 0,
      reason: String(parsed.reason ?? ''),
    }
  } catch {
    return { score: 0, reason: 'Judge call failed' }
  }
}

// ─── Metrics ──────────────────────────────────────────────────────────────────

/**
 * Context recall — purely mechanical, no LLM needed.
 * Score 1 if at least one expected source appears in the retrieved set.
 *
 * Using "at least one" rather than "all" because a question may have multiple
 * relevant sources and retrieving any of them is a retrieval success.
 */
function scoreContextRecall(
  expectedSources: string[],
  retrievedSources: string[]
): CaseResult['contextRecall'] {
  const found = expectedSources.filter((s) => retrievedSources.includes(s))
  const missed = expectedSources.filter((s) => !retrievedSources.includes(s))
  return { score: found.length > 0 ? 1 : 0, found, missed }
}

/**
 * Faithfulness — LLM judge.
 * Does the answer contain only information supported by the retrieved chunks?
 *
 * PROD NOTE — We pass 200-char excerpts (the max returned by ragQuery).
 *   For higher-fidelity faithfulness checks, the eval runner would call the
 *   retrieval pipeline directly to get full chunk content before judging.
 */
async function scoreFaithfulness(answer: string, excerpts: string[]): Promise<JudgeResult> {
  if (excerpts.length === 0) {
    return { score: 1, reason: 'No context retrieved — no factual claims to verify' }
  }
  const context = excerpts.join('\n\n---\n\n')
  return judge(`Context (documentation excerpts retrieved for this query):
${context}

Answer generated by the AI:
${answer}

Does the answer contain ONLY information supported by the context above?
- Score 1: answer is grounded in the context (may be incomplete but not contradictory or invented)
- Score 0: answer states facts not present in or directly contradicted by the context

{"score": 0 or 1, "reason": "one concise sentence explaining the score"}`)
}

/**
 * Answer relevance — LLM judge.
 * Does the answer address what was asked?
 */
async function scoreAnswerRelevance(question: string, answer: string): Promise<JudgeResult> {
  return judge(`Question asked by the user:
${question}

Answer generated by the AI:
${answer}

Does this answer directly address the question?
- Score 1: answer addresses what was asked (even a "the docs don't cover this" response counts if accurate)
- Score 0: answer is off-topic, ignores the question, or only discusses unrelated information

{"score": 0 or 1, "reason": "one concise sentence explaining the score"}`)
}

// ─── Per-case runner ──────────────────────────────────────────────────────────

async function runCase(c: EvalCase, idx: number, total: number): Promise<CaseResult> {
  process.stderr.write(`\n[${idx + 1}/${total}] ${c.question}\n`)

  const { answer, sources } = await ragQuery(c.question)
  const retrievedSources = sources.map((s) => s.source)
  const excerpts = sources.map((s) => s.excerpt)

  process.stderr.write('  Running judges...\n')

  // Run the mechanical recall check and both LLM judges in parallel
  const [contextRecall, faithfulness, answerRelevance] = await Promise.all([
    Promise.resolve(scoreContextRecall(c.expected_sources, retrievedSources)),
    scoreFaithfulness(answer, excerpts),
    scoreAnswerRelevance(c.question, answer),
  ])

  return { question: c.question, contextRecall, faithfulness, answerRelevance }
}

// ─── Report ───────────────────────────────────────────────────────────────────

function tick(score: 0 | 1): string {
  return score === 1 ? '✓' : '✗'
}

function pct(n: number, total: number): string {
  return ((n / total) * 100).toFixed(1)
}

function printReport(results: CaseResult[]): void {
  const n = results.length
  const out = process.stdout

  out.write('\n' + '═'.repeat(72) + '\n')
  out.write('  Eval Report\n')
  out.write('═'.repeat(72) + '\n')

  for (let i = 0; i < results.length; i++) {
    const r = results[i]
    const recall = r.contextRecall
    const recallDetail =
      recall.score === 1
        ? `found: ${recall.found.join(', ')}`
        : `missed: ${recall.missed.join(', ')}`

    out.write(`\n #${i + 1}  ${r.question}\n`)
    out.write(`       Recall   : ${tick(recall.score)}  ${recallDetail}\n`)
    out.write(`       Faithful : ${tick(r.faithfulness.score)}  ${r.faithfulness.reason}\n`)
    out.write(`       Relevant : ${tick(r.answerRelevance.score)}  ${r.answerRelevance.reason}\n`)
  }

  const recallScore = results.filter((r) => r.contextRecall.score === 1).length
  const faithfulScore = results.filter((r) => r.faithfulness.score === 1).length
  const relevantScore = results.filter((r) => r.answerRelevance.score === 1).length
  const total = recallScore + faithfulScore + relevantScore
  const maxTotal = n * 3

  out.write('\n' + '─'.repeat(72) + '\n')
  out.write('  Summary\n')
  out.write('─'.repeat(72) + '\n')
  out.write(`  Context recall   : ${recallScore} / ${n}  (${pct(recallScore, n)}%)\n`)
  out.write(`  Faithfulness     : ${faithfulScore} / ${n}  (${pct(faithfulScore, n)}%)\n`)
  out.write(`  Answer relevance : ${relevantScore} / ${n}  (${pct(relevantScore, n)}%)\n`)
  out.write('─'.repeat(72) + '\n')
  out.write(`  Overall          : ${total} / ${maxTotal}  (${pct(total, maxTotal)}%)\n`)
  out.write('═'.repeat(72) + '\n\n')
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  const evalPath = path.join(process.cwd(), 'data', 'eval.json')
  const cases: EvalCase[] = JSON.parse(fs.readFileSync(evalPath, 'utf-8'))

  process.stderr.write(`\nRunning ${cases.length} eval cases...\n`)
  if (!verbose) process.stderr.write('Pass --verbose to see pipeline logs.\n')

  const results: CaseResult[] = []
  for (let i = 0; i < cases.length; i++) {
    results.push(await runCase(cases[i], i, cases.length))
  }

  printReport(results)

  const overallPct =
    results.reduce(
      (sum, r) => sum + r.contextRecall.score + r.faithfulness.score + r.answerRelevance.score,
      0
    ) /
    (results.length * 3)

  // Exit 1 if below 70% — allows CI to gate on eval quality
  process.exit(overallPct < 0.7 ? 1 : 0)
}

main().catch((err) => {
  console.error('Eval failed:', err)
  process.exit(1)
})
