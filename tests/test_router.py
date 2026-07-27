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

    mock_invoke.assert_called_once_with("summarize this report", "haiku", session_id=None, on_event=None)
    assert decision.tier == "cheap"
    assert result is fake_result


def test_route_and_run_threads_session_id_through_to_invoke():
    request = TaskRequest(
        description="summarize this report", task_type="summarization", domain="other", session_id="fixed-uuid"
    )
    fake_result = ProviderResult(text="summary here", cost_usd=0.001, duration_ms=200)

    with patch("llm_task_router.router.claude_cli.invoke", return_value=fake_result) as mock_invoke:
        route_and_run(request)

    mock_invoke.assert_called_once_with("summarize this report", "haiku", session_id="fixed-uuid", on_event=None)


def test_route_and_run_forwards_on_event_to_invoke():
    request = TaskRequest(description="summarize this report", task_type="summarization", domain="other")
    fake_result = ProviderResult(text="summary here", cost_usd=0.001, duration_ms=200)
    on_event = lambda event: None  # noqa: E731

    with patch("llm_task_router.router.claude_cli.invoke", return_value=fake_result) as mock_invoke:
        route_and_run(request, on_event=on_event)

    assert mock_invoke.call_args.kwargs["on_event"] is on_event


def test_route_and_run_calls_on_decision_before_invoke_with_resolved_decision():
    """on_decision must fire before the (potentially long) provider call, not
    after - that's the whole point of adding it, so repl.py can print a
    routing header immediately instead of only once the full response
    lands."""
    request = TaskRequest(description="summarize this report", task_type="summarization", domain="other")
    fake_result = ProviderResult(text="summary here", cost_usd=0.001, duration_ms=200)
    call_order = []

    def fake_invoke(*args, **kwargs):
        call_order.append("invoke")
        return fake_result

    def on_decision(decision):
        call_order.append("decision")
        assert decision.tier == "cheap"

    with patch("llm_task_router.router.claude_cli.invoke", side_effect=fake_invoke):
        route_and_run(request, on_decision=on_decision)

    assert call_order == ["decision", "invoke"]
