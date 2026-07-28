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

-- Drift-auditing decision log: one row per live route() call, written by
-- decision_log.py (see that module and scripts/audit_tier2.py). Queried by
-- scan/time-range for audits, not nearest-neighbor lookup - no HNSW index.
CREATE TABLE routing_decisions (
    id BIGSERIAL PRIMARY KEY,
    description TEXT NOT NULL,
    embedding VECTOR(384),                -- null when the heuristic resolved everything for free
    resolved_task_type TEXT NOT NULL,
    task_type_source TEXT NOT NULL,       -- provided | inferred | tier2_nn | tier2_llm_fallback | fallback
    domain TEXT,
    domain_source TEXT NOT NULL,          -- provided | inferred | fallback
    resolved_is_high_stakes BOOLEAN,      -- null when the corroboration branch was never entered
    high_stakes_source TEXT,              -- keyword | tier2_nn | tier2_llm_fallback | unavailable | null
    no_signal_llm_used BOOLEAN NOT NULL DEFAULT false,
    bias TEXT NOT NULL,
    tier TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX routing_decisions_created_at ON routing_decisions (created_at);
