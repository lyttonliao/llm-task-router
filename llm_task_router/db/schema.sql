-- Tier-2 continuous-learning classifier's vector store.
--
-- Applied manually, once, against a real local Postgres instance:
--   psql -f llm_task_router/db/schema.sql
-- Not run automatically by application code - there's no migration
-- framework at this stage (see the plan this schema came from). Re-running
-- this file against a database that already has the extension/table is a
-- no-op for CREATE EXTENSION IF NOT EXISTS, but CREATE TABLE/INDEX will
-- error if routing_examples already exists - this is a one-time setup
-- script, not idempotent DDL.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE routing_examples (
    id BIGSERIAL PRIMARY KEY,
    description TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    task_type TEXT,
    is_high_stakes BOOLEAN,
    source TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX routing_examples_embedding_hnsw
    ON routing_examples USING hnsw (embedding vector_cosine_ops);
