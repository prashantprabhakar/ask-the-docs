/**
 * Thin structured logger.
 *
 * Writes newline-delimited JSON to stdout/stderr. Every log line is a single
 * JSON object — machine-readable by default, pipe through `jq` for humans:
 *
 *   npm run dev | jq .
 *
 * LEARN — Why structured logs?
 *   `console.log("retrieval took 142ms")` is readable but not queryable.
 *   `log.info({ event: 'retrieval', durationMs: 142 })` can be filtered,
 *   aggregated, and alerted on by any log platform (Datadog, Loki, CloudWatch).
 *   The shape of the data is the contract — the human-readable message is
 *   optional decoration on top.
 *
 * PROD NOTE — In production, replace the console.* calls with a proper
 *   logging library (pino is the Node.js standard: low overhead, async writes,
 *   pluggable transports). Pino's API is nearly identical to this one, so the
 *   migration is a one-line import change.
 *
 *   import pino from 'pino'
 *   export const logger = pino()
 */

type LogData = Record<string, unknown>

function write(level: 'info' | 'warn' | 'error', data: LogData): void {
  const line = JSON.stringify({ level, ts: new Date().toISOString(), ...data })
  if (level === 'error') {
    console.error(line)
  } else {
    console.log(line)
  }
}

export const logger = {
  info:  (data: LogData) => write('info',  data),
  warn:  (data: LogData) => write('warn',  data),
  error: (data: LogData) => write('error', data),

  /**
   * Start a high-resolution timer. Call the returned function to get the
   * elapsed time in milliseconds at any later point.
   *
   * Usage:
   *   const elapsed = logger.timer()
   *   await doWork()
   *   logger.info({ event: 'done', durationMs: elapsed() })
   */
  timer(): () => number {
    const start = Date.now()
    return () => Date.now() - start
  },
}
