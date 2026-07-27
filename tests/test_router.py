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


def test_route_caps_inferred_architecture_keywords_at_mid_without_high_stakes_signal():
    """Superseded 2026-07-27: an *inferred* architecture-shape match ("design",
    "scalable", "fault-tolerant") used to escalate to flagship on shape alone.
    That's too broad - the same "design" keyword fires just as readily on a
    trivial question. Without a genuine high-stakes signal (production/
    security/compliance/irreversibility/scale vocabulary) in the description,
    this is now capped at mid, not flagship."""
    request = TaskRequest(description="design a scalable, fault-tolerant system for this workload")
    decision = route(request)

    assert decision.tier == "mid"
    assert decision.model == "sonnet"
    assert "no high-stakes signal" in decision.reason


def test_route_escalates_inferred_architecture_to_flagship_with_high_stakes_signal():
    """A real high-stakes signal (compliance + payment processing) alongside
    an inferred architecture-shape match is what should actually earn
    flagship - not the shape keyword alone."""
    request = TaskRequest(
        description="design the multi-region disaster recovery strategy for our payment processing system, "
        "given a strict compliance requirement"
    )
    decision = route(request)

    assert decision.tier == "flagship"
    assert decision.model == "opus"


def test_route_does_not_second_guess_an_explicit_caller_provided_type():
    """A caller-provided task_type="architecture" is a deliberate override,
    not a heuristic guess - it's trusted as-is and not gated behind a
    high-stakes keyword check the way an inferred match is."""
    request = TaskRequest(description="design a small internal tool", task_type="architecture", domain="backend")
    decision = route(request)

    assert decision.tier == "flagship"
    assert decision.model == "opus"


def test_route_hedges_to_mid_not_flagship_when_no_keyword_signal_on_either_axis():
    """Regression guard for a real incident: a trivial toy prompt with zero
    keyword overlap on either axis ("reply with exactly the word: pong")
    used to silently inherit architecture's uniform-H row via
    classify_description()'s label-only fallback, routing a $0.0002 task to
    opus at real cost (~$0.18 observed). Total absence of signal must hedge
    at mid, not flagship - flagship is earned by an actual signal, not
    awarded to "we don't know"."""
    request = TaskRequest(description="reply with exactly the word: pong")
    decision = route(request)

    assert decision.tier == "mid"
    assert decision.model == "sonnet"
    assert "no keyword signal" in decision.reason


def test_route_uses_grid_normally_when_only_domain_axis_is_unresolved():
    """One resolved axis (a real task_type keyword match) is enough real
    signal to consult the grid normally, even if domain falls back to
    "other" - this is not the no-signal case."""
    request = TaskRequest(description="refactor this messy function")
    decision = route(request)

    assert decision.tier == "cheap"  # refactor x other = L, per the calibration-derived uniform-L row
    assert "heuristic grid" in decision.reason


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
