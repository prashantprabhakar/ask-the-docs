import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'

export interface RawDocument {
  content: string
  source: string // filename or URL
  title: string
}

export interface Chunk {
  content: string
  source: string
  title: string
  chunkIndex: number
}

// RecursiveCharacterTextSplitter tries to split on paragraphs → sentences → words
// so chunks don't cut off mid-sentence.
//
// chunkSize    = max characters per chunk (~300-500 tokens at 4 chars/token)
// chunkOverlap = how many characters the next chunk shares with the previous one
//                (overlap ensures context isn't lost at boundaries)
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1500,
  chunkOverlap: 200,
  separators: ['\n## ', '\n### ', '\n\n', '\n', ' '],
})

/**
 * Split a document into overlapping chunks ready for embedding.
 *
 * Why overlap? If an answer spans a chunk boundary, having 200 chars of
 * shared context means at least one chunk will contain the full answer.
 */
export async function chunkDocument(doc: RawDocument): Promise<Chunk[]> {
  const texts = await splitter.splitText(doc.content)

  return texts.map((text, i) => ({
    content: text,
    source: doc.source,
    title: doc.title,
    chunkIndex: i,
  }))
}

export async function chunkDocuments(docs: RawDocument[]): Promise<Chunk[]> {
  const allChunks: Chunk[] = []
  for (const doc of docs) {
    const chunks = await chunkDocument(doc)
    allChunks.push(...chunks)
  }
  return allChunks
}
