"""decision_log.py must never open a real Postgres connection. Same
fake-cursor/fake-connection doubles as test_vector_store.py, patching
`llm_task_router.decision_log.psycopg.connect` and `.register_vector`
directly rather than touching a real process."""

import os
from unittest.mock import patch

import pytest
from pgvector import Vector

from llm_task_router.decision_log import LoggedDecision, fetch_decisions, log_decision

_DATABASE_URL = "postgresql://test:test@localhost/test"


class _FakeCursor:
    def __init__(self, fetchall_result=None):
        self.executed = []
        self._fetchall_result = fetchall_result or []

    def execute(self, query, params=None):
        self.executed.append((query, params))
        return self

    def fetchall(self):
        return self._fetchall_result

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _patched_env():
    return patch.dict(os.environ, {"DATABASE_URL": _DATABASE_URL})


def test_log_decision_inserts_expected_row_with_embedding():
    fake_cursor = _FakeCursor()
    fake_conn = _FakeConnection(fake_cursor)
    with (
        _patched_env(),
        patch(
            "llm_task_router.decision_log.psycopg.connect", return_value=fake_conn
        ) as mock_connect,
        patch("llm_task_router.decision_log.register_vector") as mock_register,
    ):
        log_decision(
            "write a function that validates emails",
            [0.1, 0.2, 0.3],
            resolved_task_type="code_gen",
            task_type_source="inferred",
            domain=None,
            domain_source="fallback",
            resolved_is_high_stakes=None,
            high_stakes_source=None,
            no_signal_llm_used=False,
            bias="L",
            tier="cheap",
            provider="claude",
            model="haiku",
            reason="heuristic grid: code_gen x None -> L (type inferred, domain fallback)",
            shadow_bias="L",
            shadow_tier="cheap",
            shadow_reason="shadow (no tier 2): heuristic grid code_gen x None -> L",
        )

    mock_connect.assert_called_once_with(_DATABASE_URL)
    mock_register.assert_called_once_with(fake_conn)
    assert len(fake_cursor.executed) == 1
    query, params = fake_cursor.executed[0]
    assert "INSERT INTO routing_decisions" in query
    (
        description,
        embedding_param,
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
        shadow_bias,
        shadow_tier,
        shadow_reason,
    ) = params
    assert description == "write a function that validates emails"
    assert embedding_param == Vector([0.1, 0.2, 0.3])
    assert resolved_task_type == "code_gen"
    assert task_type_source == "inferred"
    assert domain is None
    assert domain_source == "fallback"
    assert resolved_is_high_stakes is None
    assert high_stakes_source is None
    assert no_signal_llm_used is False
    assert bias == "L"
    assert tier == "cheap"
    assert provider == "claude"
    assert model == "haiku"
    assert "heuristic grid" in reason
    assert shadow_bias == "L"
    assert shadow_tier == "cheap"
    assert "shadow (no tier 2)" in shadow_reason


def test_log_decision_null_embedding_for_pure_heuristic_path():
    """A pure-heuristic decision never calls embeddings.embed() - the logged
    row should carry a null embedding, not a fabricated one."""
    fake_cursor = _FakeCursor()
    fake_conn = _FakeConnection(fake_cursor)
    with (
        _patched_env(),
        patch("llm_task_router.decision_log.psycopg.connect", return_value=fake_conn),
        patch("llm_task_router.decision_log.register_vector"),
    ):
        log_decision(
            "refactor this function to remove duplication",
            None,
            resolved_task_type="refactor",
            task_type_source="inferred",
            domain="backend",
            domain_source="inferred",
            resolved_is_high_stakes=None,
            high_stakes_source=None,
            no_signal_llm_used=False,
            bias="L",
            tier="cheap",
            provider="claude",
            model="haiku",
            reason="heuristic grid: refactor x backend -> L (type inferred, domain inferred)",
            shadow_bias="L",
            shadow_tier="cheap",
            shadow_reason="shadow (no tier 2): heuristic grid refactor x backend -> L",
        )

    _query, params = fake_cursor.executed[0]
    assert params[1] is None


def test_log_decision_records_tier2_and_high_stakes_sources():
    fake_cursor = _FakeCursor()
    fake_conn = _FakeConnection(fake_cursor)
    with (
        _patched_env(),
        patch("llm_task_router.decision_log.psycopg.connect", return_value=fake_conn),
        patch("llm_task_router.decision_log.register_vector"),
    ):
        log_decision(
            "migrate the k8s cluster to a new region",
            [0.4, 0.5],
            resolved_task_type="multi_step",
            task_type_source="tier2_nn",
            domain="infra",
            domain_source="inferred",
            resolved_is_high_stakes=True,
            high_stakes_source="tier2_llm_fallback",
            no_signal_llm_used=False,
            bias="H",
            tier="flagship",
            provider="claude",
            model="opus",
            reason="tier 2's continuous-learning classifier confirmed genuine high stakes",
            shadow_bias="M",
            shadow_tier="mid",
            shadow_reason="shadow (no tier 2): heuristic grid said H, no keyword corroboration - capped at mid",
        )

    _query, params = fake_cursor.executed[0]
    resolved_is_high_stakes = params[6]
    high_stakes_source = params[7]
    task_type_source = params[3]
    shadow_tier = params[15]
    assert task_type_source == "tier2_nn"
    assert resolved_is_high_stakes is True
    assert high_stakes_source == "tier2_llm_fallback"
    assert shadow_tier == "mid"


def test_database_url_missing_raises_runtime_error():
    env_without_url = {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}
    with (
        patch.dict(os.environ, env_without_url, clear=True),
        patch("llm_task_router.decision_log.psycopg.connect") as mock_connect,
    ):
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            log_decision(
                "some description",
                None,
                resolved_task_type="code_gen",
                task_type_source="inferred",
                domain=None,
                domain_source="fallback",
                resolved_is_high_stakes=None,
                high_stakes_source=None,
                no_signal_llm_used=False,
                bias="L",
                tier="cheap",
                provider="claude",
                model="haiku",
                reason="test",
                shadow_bias="L",
                shadow_tier="cheap",
                shadow_reason="test",
            )

    mock_connect.assert_not_called()


# --- fetch_decisions --------------------------------------------------------


def test_fetch_decisions_returns_logged_decision_rows_in_order():
    row = (
        1,
        "write a function that validates emails",
        "code_gen",
        "inferred",
        None,
        "fallback",
        None,
        None,
        "L",
        "cheap",
        "claude",
        "haiku",
        "heuristic grid: code_gen x None -> L",
        "L",
        "cheap",
        "shadow (no tier 2): heuristic grid code_gen x None -> L",
    )
    fake_cursor = _FakeCursor(fetchall_result=[row])
    fake_conn = _FakeConnection(fake_cursor)
    with (
        _patched_env(),
        patch(
            "llm_task_router.decision_log.psycopg.connect", return_value=fake_conn
        ) as mock_connect,
        patch("llm_task_router.decision_log.register_vector"),
    ):
        decisions = fetch_decisions()

    mock_connect.assert_called_once_with(_DATABASE_URL)
    query, _params = fake_cursor.executed[0]
    assert "SELECT" in query
    assert "FROM routing_decisions" in query
    assert "ORDER BY created_at ASC" in query
    assert decisions == [
        LoggedDecision(
            id=1,
            description="write a function that validates emails",
            resolved_task_type="code_gen",
            task_type_source="inferred",
            domain=None,
            domain_source="fallback",
            resolved_is_high_stakes=None,
            high_stakes_source=None,
            bias="L",
            tier="cheap",
            provider="claude",
            model="haiku",
            reason="heuristic grid: code_gen x None -> L",
            shadow_bias="L",
            shadow_tier="cheap",
            shadow_reason="shadow (no tier 2): heuristic grid code_gen x None -> L",
        )
    ]


def test_fetch_decisions_database_url_missing_raises_runtime_error():
    env_without_url = {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}
    with (
        patch.dict(os.environ, env_without_url, clear=True),
        patch("llm_task_router.decision_log.psycopg.connect") as mock_connect,
    ):
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            fetch_decisions()

    mock_connect.assert_not_called()
