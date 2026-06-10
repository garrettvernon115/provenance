-- Phase 1 schema: parsed filings and their retrieval chunks.
-- The embedding vector column is deliberately deferred to Phase 2, when the
-- embedding model (and therefore the dimension) is chosen alongside baseline
-- retrieval.

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    accession   TEXT        NOT NULL UNIQUE,
    cik         BIGINT      NOT NULL,
    company     TEXT        NOT NULL,
    form        TEXT        NOT NULL,
    filed       DATE        NOT NULL,
    source_file TEXT        NOT NULL,  -- path relative to the data dir
    source_url  TEXT,
    sha256      TEXT,                  -- hash of the fetched source document
    parser      TEXT        NOT NULL,
    full_text   TEXT        NOT NULL,  -- canonical extracted text; chunk offsets index into this
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id             BIGSERIAL PRIMARY KEY,
    document_id    BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index    INT    NOT NULL,
    section        TEXT,              -- e.g. "Item 1A" for 10-Ks, "Form 4", "page 3"
    text           TEXT   NOT NULL,
    char_start     INT    NOT NULL,   -- invariant: text = documents.full_text[char_start:char_end]
    char_end       INT    NOT NULL,
    token_estimate INT,
    ts tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_ts_idx ON chunks USING GIN (ts);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS documents_form_idx ON documents (form);
