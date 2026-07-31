from unittest.mock import Mock, patch

import pytest


@pytest.fixture(autouse=True)
def _noop_decision_log():
    """route() logs every decision via decision_log.log_decision() (see
    router.py's docstring). Autouse-patch it to a no-op Mock for the whole
    suite so the ~140 existing tests that call route()/route_and_run() don't
    each need to mock it individually - tests that care about what gets
    logged grab this fixture explicitly and inspect the mock.

    return_value=None (rather than the default Mock-returns-a-Mock behavior)
    matters here: route() threads log_decision()'s return straight into
    RouteDecision.decision_log_id, and route_and_run() only calls
    log_result() (below) when that id is not None - a real Mock object is
    truthy, which would make every test's route_and_run() attempt a second,
    unmocked-by-default write."""
    with patch(
        "llm_task_router.router.decision_log.log_decision", Mock(return_value=None)
    ) as mock_log:
        yield mock_log


@pytest.fixture(autouse=True)
def _noop_decision_log_result():
    """route_and_run() writes the real ProviderResult's cost/duration back
    onto the row log_decision() created, via decision_log.log_result() -
    see router.py. Autouse-patched the same way as _noop_decision_log above,
    for the same reason (most tests don't care, and a real Postgres call
    must never happen in the test suite)."""
    with patch("llm_task_router.router.decision_log.log_result", Mock()) as mock_log_result:
        yield mock_log_result
