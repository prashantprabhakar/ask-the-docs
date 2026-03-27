import 'dotenv/config'
import { healthCheck, getChunkCount } from '@/lib/vectordb'

/**
 * GET /api/health
 *
 * Readiness probe — returns 200 when the service can handle traffic,
 * 503 when a dependency (Qdrant) is unreachable.
 *
 * PROD NOTE — Kubernetes / load balancers call this before routing traffic
 *   to a pod. A 503 here causes the pod to be removed from the rotation
 *   until it recovers. Wire it into your deployment's readinessProbe:
 *
 *   readinessProbe:
 *     httpGet:
 *       path: /api/health
 *       port: 3000
 *     initialDelaySeconds: 5
 *     periodSeconds: 10
 */
export async function GET() {
  const qdrantOk = await healthCheck()
  const chunkCount = qdrantOk ? await getChunkCount() : 0

  const body = {
    status: qdrantOk ? 'ok' : 'degraded',
    qdrant: qdrantOk,
    chunkCount,
    uptimeSeconds: Math.floor(process.uptime()),
  }

  return Response.json(body, { status: qdrantOk ? 200 : 503 })
}
