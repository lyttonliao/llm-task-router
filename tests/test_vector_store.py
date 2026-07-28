"""vector_store.py must never open a real Postgres connection. Every test
here patches `llm_task_router.vector_store.psycopg.connect` (and
`register_vector`, which does a real round-trip to fetch pgvector's type OID
- also patched to a no-op) with fake connection/cursor doubles, mirroring
how test_claude_cli.py patches subprocess.Popen with _FakeProcess rather than
touching a real process."""

import os
from unittest.mock import patch

import pytest
from pgvector import Vector

from llm_task_router.vector_store import NeighborMatch, insert_example, nearest_neighbors

_DATABASE_URL = "postgresql://test:test@localhost/test"


class _FakeCursor:
    """Stand-in for a psycopg Cursor: records every execute() call and
    returns a canned fetchall() result, supporting the `with conn.cursor()
    as cur:` context-manager usage vector_store.py depends on."""

    def __init__(self, fetchall_return=()):
        self.executed = []
        self._fetchall_return = fetchall_return

    def execute(self, query, params=None):
        self.executed.append((query, params))
        return self

    def fetchall(self):
        return self._fetchall_return

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeConnection:
    """Stand-in for psycopg.connect()'s return value - a connection used as
    a context manager (commits/closes on exit, see Connection.__exit__).
    Never opens a real socket."""

    def __init__(self, cursor):
        self._cursor = cursor
        self.closed = False

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _patched_env():
    return patch.dict(os.environ, {"DATABASE_URL": _DATABASE_URL})


# --- nearest_neighbors -------------------------------------------------


def test_nearest_neighbors_rejects_unknown_label_column_without_connecting():
    with (
        _patched_env(),
        patch("llm_task_router.vector_store.psycopg.connect") as mock_connect,
        patch("llm_task_router.vector_store.register_vector") as mock_register,
    ):
        with pytest.raises(ValueError, match="label_column"):
            nearest_neighbors([0.1, 0.2], "not_a_real_column")

    mock_connect.assert_not_called()
    mock_register.assert_not_called()


def test_nearest_neighbors_queries_expected_column_and_returns_closest_first():
    fake_cursor = _FakeCursor(fetchall_return=[("code_gen", 0.93), ("refactor", 0.81)])
    fake_conn = _FakeConnection(fake_cursor)
    with (
        _patched_env(),
        patch("llm_task_router.vector_store.psycopg.connect", return_value=fake_conn) as mock_connect,
        patch("llm_task_router.vector_store.register_vector") as mock_register,
    ):
        result = nearest_neighbors([0.1, 0.2, 0.3], "task_type", k=2)

    mock_connect.assert_called_once_with(_DATABASE_URL)
    mock_register.assert_called_once_with(fake_conn)
    assert result == [NeighborMatch(label="code_gen", similarity=0.93), NeighborMatch(label="refactor", similarity=0.81)]

    assert len(fake_cursor.executed) == 1
    query, params = fake_cursor.executed[0]
    assert "task_type IS NOT NULL" in query
    assert "SELECT task_type" in query
    assert "<=>" in query
    vector_param, vector_param_repeated, limit_param = params
    assert vector_param == Vector([0.1, 0.2, 0.3])
    assert vector_param_repeated == Vector([0.1, 0.2, 0.3])
    assert limit_param == 2


def test_nearest_neighbors_validates_is_high_stakes_column_too():
    fake_cursor = _FakeCursor(fetchall_return=[(True, 0.75)])
    fake_conn = _FakeConnection(fake_cursor)
    with (
        _patched_env(),
        patch("llm_task_router.vector_store.psycopg.connect", return_value=fake_conn),
        patch("llm_task_router.vector_store.register_vector"),
    ):
        result = nearest_neighbors([0.4, 0.5], "is_high_stakes")

    assert result == [NeighborMatch(label=True, similarity=0.75)]
    query, _params = fake_cursor.executed[0]
    assert "is_high_stakes IS NOT NULL" in query


def test_nearest_neighbors_defaults_k_to_5():
    fake_cursor = _FakeCursor(fetchall_return=[])
    fake_conn = _FakeConnection(fake_cursor)
    with (
        _patched_env(),
        patch("llm_task_router.vector_store.psycopg.connect", return_value=fake_conn),
        patch("llm_task_router.vector_store.register_vector"),
    ):
        nearest_neighbors([0.1], "task_type")

    _query, params = fake_cursor.executed[0]
    assert params[2] == 5


# --- insert_example ------------------------------------------------------


def test_insert_example_rejects_unknown_source_without_connecting():
    with (
        _patched_env(),
        patch("llm_task_router.vector_store.psycopg.connect") as mock_connect,
        patch("llm_task_router.vector_store.register_vector") as mock_register,
    ):
        with pytest.raises(ValueError, match="source"):
            insert_example("some description", [0.1, 0.2], source="not_a_real_source")

    mock_connect.assert_not_called()
    mock_register.assert_not_called()


def test_insert_example_inserts_expected_row_for_seed_source():
    fake_cursor = _FakeCursor()
    fake_conn = _FakeConnection(fake_cursor)
    with (
        _patched_env(),
        patch("llm_task_router.vector_store.psycopg.connect", return_value=fake_conn) as mock_connect,
        patch("llm_task_router.vector_store.register_vector") as mock_register,
    ):
        insert_example(
            "fix the null pointer bug in checkout",
            [0.1, 0.2, 0.3],
            task_type="triage",
            is_high_stakes=None,
            source="seed",
        )

    mock_connect.assert_called_once_with(_DATABASE_URL)
    mock_register.assert_called_once_with(fake_conn)
    assert len(fake_cursor.executed) == 1
    query, params = fake_cursor.executed[0]
    assert "INSERT INTO routing_examples" in query
    description, embedding_param, task_type, is_high_stakes, source = params
    assert description == "fix the null pointer bug in checkout"
    assert embedding_param == Vector([0.1, 0.2, 0.3])
    assert task_type == "triage"
    assert is_high_stakes is None
    assert source == "seed"


def test_insert_example_llm_fallback_source_carries_both_labels():
    fake_cursor = _FakeCursor()
    fake_conn = _FakeConnection(fake_cursor)
    with (
        _patched_env(),
        patch("llm_task_router.vector_store.psycopg.connect", return_value=fake_conn),
        patch("llm_task_router.vector_store.register_vector"),
    ):
        insert_example(
            "migrate the payments DB with zero downtime",
            [0.4, 0.5],
            task_type="multi_step",
            is_high_stakes=True,
            source="llm_fallback",
        )

    _query, params = fake_cursor.executed[0]
    _description, _embedding_param, task_type, is_high_stakes, source = params
    assert task_type == "multi_step"
    assert is_high_stakes is True
    assert source == "llm_fallback"


def test_database_url_missing_raises_runtime_error():
    env_without_url = {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}
    with (
        patch.dict(os.environ, env_without_url, clear=True),
        patch("llm_task_router.vector_store.psycopg.connect") as mock_connect,
    ):
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            nearest_neighbors([0.1], "task_type")

    mock_connect.assert_not_called()
