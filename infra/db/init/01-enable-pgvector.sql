-- Runs once, on first initialization of the pgdata volume.
-- Schema (chunks table, indexes) arrives in Phase 1.
CREATE EXTENSION IF NOT EXISTS vector;
