"""Drift-auditing decision log for the router (see `db/schema.sql`'s
`routing_decisions` table and `scripts/audit_tier2.py`, which reads it back).

This is the second (and, for now, final) module allowed to contain SQL -
`vector_store.py` is the first, and remains the only place SQL touches
`routing_examples`. The two are kept separate on purpose: `vector_store.py`
is a read/write nearest-neighbor store consulted mid-request by
`tier2_classifier.py`; this module is a write-only audit trail appended to
once per `route()` call, a different concern even though both hit the same
Postgres instance.

Connection: same shape as `vector_store.py` - a fresh `psycopg.connect()` per
call, configured via `DATABASE_URL`, no pooling (low-QPS personal CLI tool,
not a service). Deliberately not importing `vector_store`'s connection
helpers - the two SQL modules stay independent rather than coupled through a
shared private helper.

`log_decision()` is expected to be called from `router.route()` wrapped in
`try/except Exception: pass` - a logging failure must never break routing,
same "tier 2 unavailable is not a hard failure" discipline
`tier2_classifier.py` already follows for its own Postgres/LLM calls. That
discipline lives in the caller, not here, so this module's own errors
(missing DATABASE_URL, a real connection failure) still raise normally and
are visible to anything that calls it directly (e.g. a future backfill
script) without a swallow already baked in.
"""

import os

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    return url


def _connect() -> psycopg.Connection:
    conn = psycopg.connect(_database_url())
    register_vector(conn)
    return conn


def log_decision(
    description: str,
    embedding: list[float] | None,
    *,
    resolved_task_type: str,
    task_type_source: str,
    domain: str | None,
    domain_source: str,
    resolved_is_high_stakes: bool | None,
    high_stakes_source: str | None,
    no_signal_llm_used: bool,
    bias: str,
    tier: str,
    provider: str,
    model: str,
    reason: str,
) -> None:
    """Inserts one row into routing_decisions. `embedding` is None whenever
    the heuristic grid resolved the request without ever calling
    `embeddings.embed()` - a pure-heuristic decision has no embedding to
    log, and forcing one into existence just for this row would mean paying
    for an embedding call this request never otherwise needed."""
    query = """
        INSERT INTO routing_decisions (
            description, embedding, resolved_task_type, task_type_source,
            domain, domain_source, resolved_is_high_stakes, high_stakes_source,
            no_signal_llm_used, bias, tier, provider, model, reason
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    vector = Vector(embedding) if embedding is not None else None
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            query,
            (
                description,
                vector,
                resolved_task_type,
                task_type_source,
                domain,
                domain_source,
                resolved_is_high_stakes,
                high_stakes_source,
                no_signal_llm_used,
                bias,
                tier,
                provider,
                model,
                reason,
            ),
        )
