-- Phase 3: LLM-generated questions per chunk (training-data positives).
-- Each row is one synthetic question that its chunk answers; the chunk is the
-- positive passage, and BM25/FTS mines hard negatives at export time
-- (reranker/build_training_data.py). Kept in the same migration chain as the
-- rest of the schema.

CREATE TABLE IF NOT EXISTS generated_questions (
    id         BIGSERIAL PRIMARY KEY,
    chunk_id   BIGINT      NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    question   TEXT        NOT NULL,
    model      TEXT        NOT NULL,   -- provenance of the generator (e.g. claude-haiku-4-5)
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS generated_questions_chunk_idx
    ON generated_questions (chunk_id);
