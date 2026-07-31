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
    -- Shadow evaluation (see router.py's _shadow_tier1_only_decision() and
    -- scripts/report_shadow_divergence.py): what tier 1 alone, with zero
    -- tier2_classifier calls, would have decided for this same request.
    -- Never acted on - bias/tier/provider/model above are still what
    -- actually served the request. Nullable for migration compatibility
    -- with the ALTER TABLE below; every row logged after this column
    -- existed always populates it (see decision_log.log_decision()).
    shadow_bias TEXT,
    shadow_tier TEXT,
    shadow_reason TEXT,
    -- Real cost/latency from the provider call this decision actually led
    -- to (see decision_log.log_result(), called from router.route_and_run()
    -- once the call completes). Null for any row where route() alone was
    -- called (a dry-run, or the provider call itself failed before this
    -- ever got attached) - never fabricated. Nullable for migration
    -- compatibility with the ALTER TABLE below, same as shadow_* above.
    cost_usd DOUBLE PRECISION,
    duration_ms INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX routing_decisions_created_at ON routing_decisions (created_at);

-- Migration for an existing live routing_decisions table created before
-- shadow evaluation was added (2026-07-28) - run this instead of the
-- CREATE TABLE above against a database that already has the table.
-- ALTER TABLE routing_decisions ADD COLUMN shadow_bias TEXT;
-- ALTER TABLE routing_decisions ADD COLUMN shadow_tier TEXT;
-- ALTER TABLE routing_decisions ADD COLUMN shadow_reason TEXT;

-- Migration for an existing live routing_decisions table created before
-- real cost/duration write-back was added (2026-07-31) - run this instead
-- of the CREATE TABLE above against a database that already has the table.
-- ALTER TABLE routing_decisions ADD COLUMN cost_usd DOUBLE PRECISION;
-- ALTER TABLE routing_decisions ADD COLUMN duration_ms INTEGER;
