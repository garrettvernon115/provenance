-- Phase 2: semantic-retrieval embeddings for chunks.
-- Dimension 384 is locked to the chosen embedding model, BAAI/bge-small-en-v1.5
-- (see backend/embedding.py). Changing models means re-embedding the corpus.
-- Backfilled out-of-band by backend/embed_chunks.py, so the column is nullable.

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding vector(384);

-- HNSW cosine index. At the current corpus size queries are effectively exact;
-- the index keeps the path realistic as the corpus grows.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
