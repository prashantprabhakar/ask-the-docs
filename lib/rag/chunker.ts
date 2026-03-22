import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'

// ─── Types ────────────────────────────────────────────────────────────────────

export interface RawDocument {
  content: string
  source: string
  title: string
}

export interface Chunk {
  content: string  // text that gets embedded — includes section prefix (see below)
  source: string
  title: string
  sectionTitle: string
  chunkIndex: number  // position within the section, not the whole document
}

// ─── Config ───────────────────────────────────────────────────────────────────

/**
 * Maximum characters per chunk after structural splitting.
 *
 * PROD NOTE — The right value depends on your embedding model's token limit
 *   and your retrieval quality goals:
 *   - Smaller chunks (500–800 chars) → more precise retrieval, less context per chunk
 *   - Larger chunks (1500–2000 chars) → more context but noisier matches
 *   - Rule of thumb: keep chunks under ~400 tokens (≈1600 chars at 4 chars/token)
 *   - Benchmark on your actual queries before locking this in
 */
const MAX_CHUNK_SIZE = 1500
const CHUNK_OVERLAP = 150

/**
 * Only used to sub-split sections that exceed MAX_CHUNK_SIZE.
 * Separators ordered from coarsest to finest — splitter falls through them
 * until it can fit within MAX_CHUNK_SIZE.
 *
 * PROD NOTE — Overlap is intentionally smaller here (150 vs 200) because
 *   sections are already semantically bounded. Overlap only guards against
 *   answers that straddle a sub-split boundary within a long section.
 */
const subSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: MAX_CHUNK_SIZE,
  chunkOverlap: CHUNK_OVERLAP,
  separators: ['\n\n', '\n', '. ', ' '],
})

// ─── Structural Header Splitting ──────────────────────────────────────────────

interface Section {
  headingPath: string  // e.g. "App Router > Layouts > Nested Layouts"
  content: string
}

/**
 * Split a markdown document into sections at heading boundaries (h1–h3).
 *
 * Why this beats character splitting:
 *   - Sections are logically self-contained — a chunk about "Dynamic Routes"
 *     will never mix with content from "Static Exports"
 *   - Adding content to one section doesn't affect any other section's chunks
 *   - The heading path gives the embedding model context about what the chunk
 *     is about even when the content itself is ambiguous
 *
 * PROD NOTE — In production you'd also handle:
 *   - Front-matter stripping (--- yaml blocks at the top of MDX files)
 *   - Code block preservation (avoid splitting mid-code-block)
 *   - Table preservation (a half-table is useless as a chunk)
 *   Tools like Unstructured.io or Docling handle all of these.
 */
function splitIntoSections(doc: RawDocument): Section[] {
  const lines = doc.content.split('\n')

  // heading stack tracks the current h1/h2/h3 breadcrumb
  // index 0 = h1, index 1 = h2, index 2 = h3
  const headingStack: string[] = []
  const sections: Section[] = []

  let currentLines: string[] = []

  const flushSection = () => {
    const text = currentLines.join('\n').trim()
    if (text.length === 0) return
    const headingPath = headingStack.filter(Boolean).join(' > ') || doc.title
    sections.push({ headingPath, content: text })
    currentLines = []
  }

  for (const line of lines) {
    const h1 = line.match(/^#\s+(.+)/)
    const h2 = line.match(/^##\s+(.+)/)
    const h3 = line.match(/^###\s+(.+)/)

    if (h1) {
      flushSection()
      headingStack[0] = h1[1].trim()
      headingStack[1] = ''
      headingStack[2] = ''
    } else if (h2) {
      flushSection()
      headingStack[1] = h2[1].trim()
      headingStack[2] = ''
    } else if (h3) {
      flushSection()
      headingStack[2] = h3[1].trim()
    } else {
      currentLines.push(line)
    }
  }

  flushSection()
  return sections
}

// ─── Contextual Prefix ────────────────────────────────────────────────────────

/**
 * Prepend the section heading path to each chunk before embedding.
 *
 * A chunk like "The default value is `true`." is ambiguous in isolation.
 * With a prefix it becomes:
 *   "App Router > Image Optimization > Lazy Loading\n\nThe default value is `true`."
 *
 * The embedding model now understands the topic and produces a much more
 * accurate vector. This technique is sometimes called "contextual chunking".
 *
 * PROD NOTE — Anthropic's "contextual retrieval" paper (2024) takes this
 *   further: use an LLM to generate a custom context sentence per chunk
 *   ("This chunk describes the lazy loading default for next/image in the
 *   App Router"). That costs extra LLM calls at ingest time but significantly
 *   improves retrieval recall — reported ~49% reduction in retrieval failures.
 *   Worth it for production if your query volume justifies the ingest cost.
 */
function withSectionPrefix(sectionTitle: string, content: string): string {
  return `${sectionTitle}\n\n${content}`
}

// ─── Public API ───────────────────────────────────────────────────────────────

export async function chunkDocument(doc: RawDocument): Promise<Chunk[]> {
  const sections = splitIntoSections(doc)
  const chunks: Chunk[] = []

  for (const section of sections) {
    if (section.content.length <= MAX_CHUNK_SIZE) {
      // Section fits in one chunk — no sub-splitting needed.
      // chunkIndex = 0 signals "this is the whole section, not a fragment".
      chunks.push({
        content: withSectionPrefix(section.headingPath, section.content),
        source: doc.source,
        title: doc.title,
        sectionTitle: section.headingPath,
        chunkIndex: 0,
      })
    } else {
      /**
       * Section is too large — sub-split it with the character splitter.
       *
       * We sub-split the raw content (without prefix), then add the prefix
       * to each sub-chunk. This avoids the prefix inflating the size check.
       *
       * PROD NOTE — Sub-split fragments lose full-section context. If precision
       *   matters more than coverage, consider increasing MAX_CHUNK_SIZE to
       *   avoid sub-splitting, or use a sliding window approach instead.
       */
      const subTexts = await subSplitter.splitText(section.content)
      subTexts.forEach((text, i) => {
        chunks.push({
          content: withSectionPrefix(section.headingPath, text),
          source: doc.source,
          title: doc.title,
          sectionTitle: section.headingPath,
          chunkIndex: i,
        })
      })
    }
  }

  return chunks
}

export async function chunkDocuments(docs: RawDocument[]): Promise<Chunk[]> {
  const allChunks: Chunk[] = []
  for (const doc of docs) {
    const chunks = await chunkDocument(doc)
    allChunks.push(...chunks)
  }
  return allChunks
}
