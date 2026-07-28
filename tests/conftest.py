from unittest.mock import Mock, patch

import pytest


@pytest.fixture(autouse=True)
def _noop_decision_log():
    """route() logs every decision via decision_log.log_decision() (see
    router.py's docstring). Autouse-patch it to a no-op Mock for the whole
    suite so the ~140 existing tests that call route()/route_and_run() don't
    each need to mock it individually - tests that care about what gets
    logged grab this fixture explicitly and inspect the mock."""
    with patch("llm_task_router.router.decision_log.log_decision", Mock()) as mock_log:
        yield mock_log
