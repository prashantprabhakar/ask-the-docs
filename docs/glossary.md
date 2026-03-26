# Glossary

Quick-reference definitions for terms used across this codebase. For deeper explanations with worked examples, see [concepts-and-decisions.md](./concepts-and-decisions.md).

---

**BM25** (Best Match 25)
A ranking function that scores how relevant a document is to a query. Combines Term Frequency (how often the term appears in this document) with Inverse Document Frequency (how rare the term is across all documents), plus length normalisation. The "25" refers to the 25th iteration of the formula in the Okapi research project. BM25 is the backbone of classical search engines like Elasticsearch. In this project it is used to weight sparse vectors at query time.

---

**Chunk**
A fragment of a source document that is embedded and stored as a single vector. Chunking is necessary because embedding models have token limits and because a whole document is too coarse for precise retrieval — you want to return the specific section that answers the question, not the entire file. In this project, chunks are split at Markdown heading boundaries first, then sub-split by character count if a section is too long.

---

**Chunking (structural vs character)**
*Character chunking* splits every N characters regardless of content structure — fast but produces fragments that cross sentence or section boundaries. *Structural chunking* splits at meaningful boundaries (headings, paragraphs) so each chunk is a logically self-contained unit. This project uses structural chunking on Markdown headings.

---

**Contextual Retrieval**
A technique (from Anthropic, 2024) where an LLM generates a short context sentence for each chunk at ingest time, describing what the chunk covers in relation to the full document. The sentence is prepended to the chunk before embedding. This improves retrieval for chunks that are ambiguous in isolation (e.g. "The default value is `true`." — true for what?). Reported to reduce retrieval failures by ~49%.

---

**Cross-encoder**
A model that takes a (query, document) pair as input and outputs a single relevance score. Unlike embeddings (which encode query and document independently), a cross-encoder reads both together so it can model fine-grained interactions between them. Much more accurate for ranking than embedding similarity, but much slower — cannot be pre-computed, must run at query time. Used as a re-ranker after retrieval (V3 plan).

---

**Dense Vector / Embedding**
A fixed-length list of floating-point numbers (e.g. 768 floats) that represents the *meaning* of a piece of text. Produced by an embedding model. Texts with similar meaning have vectors that are close together in that high-dimensional space, even if they share no words. Contrast with sparse vector.

---

**Document Frequency (df)**
The number of documents (chunks) in the corpus that contain a given term. Used in the IDF formula. A term with high df is common and less informative; a term with low df is rare and more informative.

---

**@xenova/transformers**
A JavaScript/TypeScript port of Hugging Face's `transformers` Python library. Runs transformer models directly in Node.js using the ONNX Runtime — no Python process, no API key, no GPU required. Models are downloaded from Hugging Face on first use and cached on disk. In this project it powers the cross-encoder re-ranker. The `Xenova/` namespace on Hugging Face hosts pre-converted ONNX versions of popular models.

---

**HyDE** (Hypothetical Document Embeddings)
A query expansion technique where instead of embedding the raw user question, an LLM first writes a hypothetical answer to the question. That hypothetical answer is then embedded and used for retrieval. The hypothesis lives in the same semantic space as real documents, so it often retrieves better than the short, keyword-heavy original question.

---

**Hybrid Search**
Running dense (semantic) search and sparse (keyword) search simultaneously and merging the results. Dense search finds chunks that *mean* what you asked; sparse search finds chunks that *literally contain* the words you typed. Combined, they cover cases where either alone would fail. In this project, Qdrant runs both searches and fuses them with RRF.

---

**IDF** (Inverse Document Frequency)
A measure of how rare a term is across the full corpus. Terms that appear in every document (like "the") have IDF near zero — they are useless for distinguishing documents. Terms that appear in only a handful of documents have high IDF — finding them is meaningful. Formula: `log((N - df + 0.5) / (df + 0.5) + 1)` where N is total documents and df is documents containing the term.

---

**Incremental Ingestion**
Re-processing only files that have changed since the last ingest run, rather than re-embedding the entire corpus every time. Detected via a file-content hash cache. In this project the cache lives in `data/ingest-cache.json`.

---

**Inverted Index**
A data structure that maps each term to the list of documents containing it. The backbone of keyword search engines. Qdrant uses an inverted index internally for sparse vectors (same structure as Elasticsearch/Lucene), while it uses HNSW for dense vectors.

---

**HNSW** (Hierarchical Navigable Small World)
A graph-based index structure for approximate nearest-neighbour search. Organises vectors in a layered graph so that search skips most of the corpus and only inspects promising candidates. Makes dense vector search O(log n) rather than O(n). Used by Qdrant for dense vector indexing.

---

**ONNX** (Open Neural Network Exchange)
A standard open file format for ML models — like a PDF, but for neural networks. A model trained in PyTorch or TensorFlow can be exported to `.onnx`, and any ONNX Runtime (available for Python, Node.js, C++, browsers, mobile) can load and execute it. This portability is why `@xenova/transformers` can run Hugging Face models in Node.js: the models are pre-exported to ONNX format and shipped as static files. No framework-specific code is needed at inference time.

---

**Query Expansion**
Generating multiple alternative phrasings of the user's question (using an LLM) and running retrieval for each. The results are merged and de-duplicated before re-ranking. Improves recall when the user's vocabulary differs from the documentation's vocabulary (e.g. "404 page" vs "not found page").

---

**RAG** (Retrieval-Augmented Generation)
An architecture where an LLM's answer is grounded in retrieved documents rather than its training data alone. The pipeline: embed the question → search a vector database for relevant chunks → inject those chunks into the prompt → generate an answer. The LLM does not need to be retrained when the knowledge base changes — only the vector database needs to be updated.

---

**Re-ranking**
A second-pass scoring step after retrieval. The initial retrieval (hybrid search) fetches a pool of candidates (e.g. top 20). A re-ranker — typically a cross-encoder — scores each (query, candidate) pair more accurately and reorders the list. The final top-K from the re-ranked list goes into the prompt. Improves precision without sacrificing recall.

---

**RRF** (Reciprocal Rank Fusion)
An algorithm for merging two ranked lists into one. Rather than combining raw scores (which live on different scales), RRF uses rank positions: `score = 1 / (rank + 60)`. A document that appears near the top of both lists beats a document that tops only one. The constant 60 dampens the cliff between rank #1 and rank #2. Used in this project to merge dense and sparse search results.

---

**Sparse Vector**
A vector where most values are zero — only positions corresponding to terms that actually appear in the text are non-zero. Represents a document as a weighted bag-of-words. Contrast with dense vector. Used for keyword matching: the dot product between two sparse vectors is non-zero only where they share terms.

---

**SPLADE** (Sparse Lexical and Expansion)
A neural model that produces learned sparse vectors. Unlike BM25 (which counts term frequencies), SPLADE learns which terms are important for retrieval by training on query-document relevance pairs. It also performs query/document expansion — assigning non-zero weights to related terms not literally present in the text. Significantly better than BM25 for retrieval quality. Recommended for production; available via Qdrant's FastEmbed (Python).

---

**TF** (Term Frequency)
The frequency of a term within a single document, typically normalised: `count / total_tokens`. A higher TF means the term appears more in this document. Without IDF, TF alone treats "the" and "useRouter" as equally important if they appear the same number of times.

---

**UUID v5**
A deterministic UUID derived from a namespace + input string via SHA-1 hashing. Same inputs always produce the same UUID. Used in this project for chunk IDs: `uuidv5(source + content)` — so re-ingesting an unchanged chunk produces the same ID, making Qdrant upserts idempotent.

---

**Vector Database**
A database optimised for storing and searching high-dimensional vectors. Supports approximate nearest-neighbour queries (find the K vectors most similar to this query vector) efficiently using index structures like HNSW. In this project: Qdrant. Alternatives: Pinecone, Weaviate, Chroma, pgvector (Postgres extension).
