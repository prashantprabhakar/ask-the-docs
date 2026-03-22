/**
 * Fetches the Next.js documentation from the official GitHub repository
 * and writes it to data/docs/.
 *
 * HOW IT WORKS:
 *   GitHub exposes a tarball download for any repo + branch:
 *     https://codeload.github.com/{owner}/{repo}/tar.gz/refs/heads/{branch}
 *
 *   This is a single HTTP request that returns the entire repo as a .tar.gz.
 *   We stream it directly into a tar extractor without saving the whole archive
 *   to disk. The extractor filters to only the docs/ folder and strips the
 *   leading directory (next.js-canary/docs/foo.mdx → data/docs/foo.mdx).
 *
 * WHY THIS APPROACH (not the GitHub REST API)?
 *   The GitHub API has rate limits: 60 requests/hour unauthenticated.
 *   The Next.js docs directory has 200+ files — that would exhaust the limit
 *   immediately. The tarball endpoint has no per-file limit because it is a
 *   single download, not per-file API calls.
 *
 * WHY STREAM AND NOT DOWNLOAD-THEN-EXTRACT?
 *   The full Next.js repo tarball is ~100MB. Writing it to disk first wastes
 *   space. Streaming pipes the download directly into the extractor — only the
 *   docs/ files we actually want ever touch disk.
 */

import https from 'node:https'
import fsp from 'node:fs/promises'
import path from 'node:path'
import { IncomingMessage } from 'node:http'
import * as tar from 'tar'

// ─── Config ───────────────────────────────────────────────────────────────────

const REPO_OWNER = 'vercel'
const REPO_NAME = 'next.js'
const BRANCH = 'canary'
const DOCS_SUBDIR = 'docs'

/**
 * In the tarball, every entry is prefixed with "{repo}-{branch}/".
 * e.g. "next.js-canary/docs/foo.mdx"
 *
 * The tar filter receives the ORIGINAL path (before strip is applied), so we
 * must match against the full prefixed path, not the stripped one.
 */
const TARBALL_DOCS_PREFIX = `${REPO_NAME}-${BRANCH}/${DOCS_SUBDIR}/`

const OUT_DIR = path.resolve('data', 'docs')

// ─── HTTP Helper ──────────────────────────────────────────────────────────────

/**
 * Download a URL, following up to 5 redirects.
 *
 * GitHub's codeload endpoint (the tarball download) redirects once to a CDN
 * before returning the file. We need to follow that redirect manually because
 * Node's built-in https.get does not follow redirects automatically.
 */
async function download(url: string, redirectsLeft = 5): Promise<IncomingMessage> {
  return new Promise((resolve, reject) => {
    https
      .get(url, { headers: { 'User-Agent': 'ask-the-docs/fetch-docs' } }, (res) => {
        const { statusCode, headers } = res

        if ((statusCode === 301 || statusCode === 302 || statusCode === 307) && headers.location) {
          if (redirectsLeft === 0) {
            reject(new Error('Too many redirects'))
            return
          }
          // Consume the redirect response body so the socket is freed
          res.resume()
          resolve(download(headers.location, redirectsLeft - 1))
          return
        }

        if (statusCode !== 200) {
          reject(new Error(`HTTP ${statusCode} from ${url}`))
          return
        }

        resolve(res)
      })
      .on('error', reject)
  })
}

// ─── File Counter ─────────────────────────────────────────────────────────────

async function countFiles(dir: string): Promise<number> {
  let count = 0
  for (const entry of await fsp.readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    count += entry.isDirectory() ? await countFiles(full) : 1
  }
  return count
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  const tarballUrl = `https://codeload.github.com/${REPO_OWNER}/${REPO_NAME}/tar.gz/refs/heads/${BRANCH}`

  console.log(`\nFetching Next.js docs from ${REPO_OWNER}/${REPO_NAME}@${BRANCH}`)
  console.log('Clearing existing data/docs...')
  await fsp.rm(OUT_DIR, { recursive: true, force: true })
  await fsp.mkdir(OUT_DIR, { recursive: true })

  console.log('Downloading tarball (streaming)...')
  const response = await download(tarballUrl)

  /**
   * Pipe the download stream into the tar extractor.
   *
   * strip: 1  — removes the top-level directory from every entry path.
   *             In the tarball, paths look like: next.js-canary/docs/foo.mdx
   *             After strip: 1 they become: docs/foo.mdx
   *
   * cwd: data  — the base directory for extraction. Combined with strip: 1,
   *             docs/foo.mdx extracts to data/docs/foo.mdx — exactly OUT_DIR.
   *
   * filter     — called per entry (after stripping). We only extract entries
   *             whose path starts with "docs/" — everything else (src/, test/,
   *             packages/, etc.) is skipped without touching disk.
   */
  await new Promise<void>((resolve, reject) => {
    const extractor = tar.extract({
      strip: 1,
      cwd: path.resolve('data'),
      filter: (entryPath) => entryPath.startsWith(TARBALL_DOCS_PREFIX),
    })

    response.pipe(extractor)
    extractor.on('finish', resolve)
    extractor.on('error', reject)
    response.on('error', reject)
  })

  const fileCount = await countFiles(OUT_DIR)
  console.log(`\nDone. ${fileCount} files written to data/docs/`)
  console.log('Run `npm run ingest` to embed and store them.\n')
}

main().catch((err) => {
  console.error('\nFetch failed:', err.message)
  process.exit(1)
})
