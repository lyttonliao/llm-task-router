from unittest.mock import patch

from llm_task_router.schema import ProviderResult, TaskRequest
from llm_task_router.router import route, route_and_run


def test_route_resolves_tier_and_provider_with_no_io():
    request = TaskRequest(description="fix null pointer in login form", task_type="triage", domain="frontend")
    decision = route(request)

    assert decision.tier == "cheap"
    assert decision.provider == "claude"
    assert decision.model == "haiku"
    assert "triage x frontend" in decision.reason


def test_route_escalates_architecture_to_flagship():
    request = TaskRequest(description="design the new auth system", task_type="architecture", domain="backend")
    decision = route(request)

    assert decision.tier == "flagship"
    assert decision.model == "opus"


def test_route_infers_metadata_when_caller_omits_it():
    decision = route(TaskRequest(description="Summarize the React release notes"))

    assert decision.tier == "cheap"
    assert decision.model == "haiku"
    assert "summarization x frontend" in decision.reason
    assert "type inferred" in decision.reason


def test_route_and_run_invokes_resolved_provider():
    request = TaskRequest(description="summarize this report", task_type="summarization", domain="other")
    fake_result = ProviderResult(text="summary here", cost_usd=0.001, duration_ms=200)

    with patch("llm_task_router.router.claude_cli.invoke", return_value=fake_result) as mock_invoke:
        decision, result = route_and_run(request)

    mock_invoke.assert_called_once_with("summarize this report", "haiku")
    assert decision.tier == "cheap"
    assert result is fake_result
