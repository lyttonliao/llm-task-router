"""Postgres/pgvector client for the tier-2 continuous-learning classifier
(see /Users/lliao/.claude/plans/i-believe-for-now-smooth-cook.md for the full
design). This is the ONLY module in the repo allowed to contain SQL - keep it
that way; don't add a second SQL string-building helper elsewhere.

Connection: a fresh `psycopg.connect()` per call, configured via the
`DATABASE_URL` environment variable (standard convention, no hardcoded
credentials). This is a low-QPS personal CLI tool, not a service - no
connection pooling, that would be premature at this scale. `psycopg.Connection`
used as a context manager commits on clean exit / rolls back on exception and
always closes (see `psycopg.Connection.__exit__`), so callers don't need to
manage transactions explicitly.

Schema: see `llm_task_router/db/schema.sql`. `label_column` is restricted to
the two nullable label columns on `routing_examples` (`task_type`,
`is_high_stakes`) - validated against a fixed allow-list before being
interpolated into the query string, so this is not a SQL-injection surface
despite the f-string (never plumb an arbitrary caller-supplied column name
into this validation).
"""

import os
from typing import NamedTuple

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

VALID_LABEL_COLUMNS = ("task_type", "is_high_stakes")
VALID_SOURCES = ("seed", "llm_fallback")


class NeighborMatch(NamedTuple):
    label: str | bool  # the value of whichever label_column was queried
    similarity: float  # cosine similarity, higher = closer


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    return url


def _connect() -> psycopg.Connection:
    conn = psycopg.connect(_database_url())
    register_vector(conn)
    return conn


def nearest_neighbors(embedding: list[float], label_column: str, k: int = 5) -> list[NeighborMatch]:
    """Cosine-nearest neighbors among rows where `label_column` is non-null,
    using pgvector's `<=>` cosine-distance operator (`1 - distance` is
    reported as `similarity`). Closest first."""
    if label_column not in VALID_LABEL_COLUMNS:
        raise ValueError(f"label_column must be one of {VALID_LABEL_COLUMNS}, got {label_column!r}")

    query = f"""
        SELECT {label_column}, 1 - (embedding <=> %s) AS similarity
        FROM routing_examples
        WHERE {label_column} IS NOT NULL
        ORDER BY embedding <=> %s
        LIMIT %s
    """
    vector = Vector(embedding)
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(query, (vector, vector, k))
        rows = cur.fetchall()
    return [NeighborMatch(label=row[0], similarity=row[1]) for row in rows]


def existing_descriptions() -> set[str]:
    """Returns every `description` currently in `routing_examples`, fetched in
    one round-trip. Used by seed_vector_store.py to dedupe a ~98-row seed
    batch in Python rather than issuing one existence query per row."""
    query = "SELECT description FROM routing_examples"
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return {row[0] for row in rows}


def insert_example(
    description: str,
    embedding: list[float],
    *,
    task_type: str | None = None,
    is_high_stakes: bool | None = None,
    source: str,
) -> None:
    """Inserts one row into routing_examples. `source` must be 'seed' (cold
    start from llm-eval-harness's golden cases) or 'llm_fallback' (written
    back by tier 2 after a cheap-LLM resolution)."""
    if source not in VALID_SOURCES:
        raise ValueError(f"source must be one of {VALID_SOURCES}, got {source!r}")

    query = """
        INSERT INTO routing_examples (description, embedding, task_type, is_high_stakes, source)
        VALUES (%s, %s, %s, %s, %s)
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(query, (description, Vector(embedding), task_type, is_high_stakes, source))
