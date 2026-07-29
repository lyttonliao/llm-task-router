from unittest.mock import patch

from llm_task_router import tier2_classifier
from llm_task_router.schema import ProviderResult, TaskClassification, TaskRequest
from llm_task_router.router import _shadow_tier1_only_decision, route, route_and_run
from llm_task_router.tier2_classifier import HighStakesResolution


def test_route_resolves_tier_and_provider_with_no_io():
    request = TaskRequest(
        description="fix null pointer in login form",
        task_type="triage",
        domain="frontend",
    )
    decision = route(request)

    assert decision.tier == "cheap"
    assert decision.provider == "claude"
    assert decision.model == "haiku"
    assert "triage x frontend" in decision.reason


def test_route_escalates_architecture_to_flagship():
    request = TaskRequest(
        description="design the new auth system",
        task_type="architecture",
        domain="backend",
    )
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
    """Superseded 2026-07-27 (twice): an *inferred* architecture-shape match
    ("design", "scalable", "fault-tolerant") used to escalate to flagship on
    shape alone - too broad, since the same "design" keyword fires just as
    readily on a trivial question. First fix: cap at mid without a genuine
    high-stakes keyword signal. Second fix (same day): the keyword-negative
    case now also falls through to tier 2's resolve_high_stakes() instead of
    capping unconditionally - this test pins tier 2 as unavailable (mocked
    None) to isolate the *keyword-negative-and-tier-2-unavailable* case,
    which must still cap at mid exactly like the old unconditional behavior.
    See test_route_escalates_high_stakes_via_tier2_without_a_keyword_match
    below for the case where tier 2 actually corroborates it."""
    request = TaskRequest(
        description="design a scalable, fault-tolerant system for this workload"
    )
    with (
        patch("llm_task_router.router.embeddings.embed", return_value=[0.1, 0.2]),
        patch(
            "llm_task_router.router.tier2_classifier.resolve_high_stakes",
            return_value=None,
        ) as mock_resolve,
    ):
        decision = route(request)

    mock_resolve.assert_called_once_with(
        "design a scalable, fault-tolerant system for this workload",
        [0.1, 0.2],
        task_type="architecture",
    )
    assert decision.tier == "mid"
    assert decision.model == "sonnet"
    assert "no high-stakes signal" in decision.reason
    assert "tier 2's continuous-learning classifier was unavailable" in decision.reason


def test_route_escalates_high_stakes_via_tier2_without_a_keyword_match():
    """The actually-ambiguous case tier 2's corroboration extension targets:
    no IMPACT_KEYWORDS match, but tier 2's continuous-learning classifier
    (NN lookup or its own cheap-LLM fallback) confirms genuine high stakes
    anyway - a real task that just doesn't happen to use recognized
    high-stakes vocabulary should still be able to reach flagship."""
    request = TaskRequest(
        description="design a scalable, fault-tolerant system for this workload"
    )
    with (
        patch("llm_task_router.router.embeddings.embed", return_value=[0.1, 0.2]),
        patch(
            "llm_task_router.router.tier2_classifier.resolve_high_stakes",
            return_value=HighStakesResolution(
                is_high_stakes=True, source="llm_fallback"
            ),
        ) as mock_resolve,
    ):
        decision = route(request)

    mock_resolve.assert_called_once_with(
        "design a scalable, fault-tolerant system for this workload",
        [0.1, 0.2],
        task_type="architecture",
    )
    assert decision.tier == "flagship"
    assert decision.model == "opus"
    assert (
        "tier 2's continuous-learning classifier confirmed genuine high stakes"
        in decision.reason
    )


def test_route_caps_at_mid_when_tier2_corroboration_confirms_no_high_stakes():
    request = TaskRequest(
        description="design a scalable, fault-tolerant system for this workload"
    )
    with (
        patch("llm_task_router.router.embeddings.embed", return_value=[0.1, 0.2]),
        patch(
            "llm_task_router.router.tier2_classifier.resolve_high_stakes",
            return_value=HighStakesResolution(
                is_high_stakes=False, source="llm_fallback"
            ),
        ),
    ):
        decision = route(request)

    assert decision.tier == "mid"
    assert "confirmed no genuine high stakes" in decision.reason


def test_route_needs_corroboration_keyword_positive_skips_tier2_entirely():
    """Free keyword evidence should never be re-spent on a tier-2 call - only
    the keyword-negative case falls through."""
    request = TaskRequest(
        description="design the multi-region disaster recovery strategy for our payment processing system, "
        "given a strict compliance requirement"
    )
    with patch(
        "llm_task_router.router.tier2_classifier.resolve_high_stakes"
    ) as mock_resolve:
        decision = route(request)

    mock_resolve.assert_not_called()
    assert decision.tier == "flagship"


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
    request = TaskRequest(
        description="design a small internal tool",
        task_type="architecture",
        domain="backend",
    )
    decision = route(request)

    assert decision.tier == "flagship"
    assert decision.model == "opus"


def test_route_hedges_to_mid_when_no_keyword_signal_and_tier3_is_unavailable():
    """Regression guard for a real incident: a trivial toy prompt with zero
    keyword overlap on either axis ("reply with exactly the word: pong")
    used to silently inherit architecture's uniform-H row via
    classify_description()'s label-only fallback, routing a $0.0002 task to
    opus at real cost (~$0.18 observed). This no longer reflects the steady
    state (tier-3's cheap-LLM fallback now classifies this correctly as
    cheap - see the test_route_no_signal_llm_fallback_* tests below) - this
    test mocks tier-3 as unavailable specifically to pin the *fallback*
    behavior: total absence of signal, with no tier-3 answer to lean on
    either, must still hedge at mid, not flagship - flagship is earned by an
    actual signal, not awarded to "we don't know"."""
    request = TaskRequest(description="reply with exactly the word: pong")
    with (
        patch("llm_task_router.router.embeddings.embed", return_value=[0.0]),
        patch(
            "llm_task_router.router.tier2_classifier.resolve_task_type",
            return_value=None,
        ),
        patch(
            "llm_task_router.router.claude_cli.invoke",
            return_value=ProviderResult(
                text="",
                cost_usd=0.0,
                duration_ms=0,
                error="mocked - not exercising tier-3 here",
            ),
        ),
    ):
        decision = route(request)

    assert decision.tier == "mid"
    assert decision.model == "sonnet"
    assert "unavailable" in decision.reason


def test_route_no_signal_llm_fallback_returns_cheap():
    request = TaskRequest(description="repeat this word: hello")
    with (
        patch("llm_task_router.router.embeddings.embed", return_value=[0.0]),
        patch(
            "llm_task_router.router.tier2_classifier.resolve_task_type",
            return_value=None,
        ),
        patch(
            "llm_task_router.router.claude_cli.invoke",
            return_value=ProviderResult(text="CHEAP", cost_usd=0.003, duration_ms=400),
        ) as mock_invoke,
    ):
        decision = route(request)

    assert decision.tier == "cheap"
    assert decision.model == "haiku"
    assert "cheap" in decision.reason
    args, kwargs = mock_invoke.call_args
    assert args[1] == "haiku"
    assert kwargs["disable_tools"] is True
    assert "session_id" not in kwargs
    assert "on_event" not in kwargs


def test_route_no_signal_llm_fallback_returns_mid():
    request = TaskRequest(description="repeat this word: hello")
    with (
        patch("llm_task_router.router.embeddings.embed", return_value=[0.0]),
        patch(
            "llm_task_router.router.tier2_classifier.resolve_task_type",
            return_value=None,
        ),
        patch(
            "llm_task_router.router.claude_cli.invoke",
            return_value=ProviderResult(text="MID", cost_usd=0.003, duration_ms=400),
        ),
    ):
        decision = route(request)

    assert decision.tier == "mid"
    assert decision.model == "sonnet"
    assert "mid" in decision.reason


def test_route_no_signal_llm_fallback_returns_flagship():
    request = TaskRequest(description="repeat this word: hello")
    with (
        patch("llm_task_router.router.embeddings.embed", return_value=[0.0]),
        patch(
            "llm_task_router.router.tier2_classifier.resolve_task_type",
            return_value=None,
        ),
        patch(
            "llm_task_router.router.claude_cli.invoke",
            return_value=ProviderResult(
                text="FLAGSHIP", cost_usd=0.003, duration_ms=400
            ),
        ),
    ):
        decision = route(request)

    assert decision.tier == "flagship"
    assert decision.model == "opus"
    assert "flagship" in decision.reason


def test_route_no_signal_llm_fallback_unparseable_text_falls_back_to_mid():
    request = TaskRequest(description="repeat this word: hello")
    with (
        patch("llm_task_router.router.embeddings.embed", return_value=[0.0]),
        patch(
            "llm_task_router.router.tier2_classifier.resolve_task_type",
            return_value=None,
        ),
        patch(
            "llm_task_router.router.claude_cli.invoke",
            return_value=ProviderResult(
                text="uh, I'm not sure?", cost_usd=0.003, duration_ms=400
            ),
        ),
    ):
        decision = route(request)

    assert decision.tier == "mid"
    assert decision.model == "sonnet"
    assert "unavailable or returned an unparseable response" in decision.reason


def test_route_no_signal_llm_fallback_provider_error_falls_back_to_mid():
    request = TaskRequest(description="repeat this word: hello")
    with (
        patch("llm_task_router.router.embeddings.embed", return_value=[0.0]),
        patch(
            "llm_task_router.router.tier2_classifier.resolve_task_type",
            return_value=None,
        ),
        patch(
            "llm_task_router.router.claude_cli.invoke",
            return_value=ProviderResult(
                text="",
                cost_usd=0.0,
                duration_ms=0,
                error="auth check failed: not logged in",
            ),
        ),
    ):
        decision = route(request)

    assert decision.tier == "mid"
    assert decision.model == "sonnet"


def test_route_no_signal_llm_fallback_exception_falls_back_to_mid():
    """Load-bearing, not defensive-programming theater: invoke() really can
    raise (check_auth()'s subprocess.run has no FileNotFoundError guard), so
    _classify_via_llm()'s try/except must actually catch it."""
    request = TaskRequest(description="repeat this word: hello")
    with (
        patch("llm_task_router.router.embeddings.embed", return_value=[0.0]),
        patch(
            "llm_task_router.router.tier2_classifier.resolve_task_type",
            return_value=None,
        ),
        patch(
            "llm_task_router.router.claude_cli.invoke",
            side_effect=RuntimeError("boom"),
        ),
    ):
        decision = route(request)

    assert decision.tier == "mid"
    assert decision.model == "sonnet"


def test_route_resolves_task_type_via_tier2_when_heuristic_finds_no_signal():
    """A description with zero TYPE_KEYWORDS overlap used to fall all the way
    to the no-signal branch's difficulty-judgment call. Tier 2 now gets a
    chance to resolve task_type first - a confident tier-2 answer reaches the
    grid directly, without ever making the tier-3 CHEAP/MID/FLAGSHIP call."""
    request = TaskRequest(description="put together a plan for the quarterly numbers")
    with (
        patch(
            "llm_task_router.router.embeddings.embed", return_value=[0.1, 0.2]
        ) as mock_embed,
        patch(
            "llm_task_router.router.tier2_classifier.resolve_task_type",
            return_value=tier2_classifier.Tier2Resolution(
                task_type="code_gen", source="llm_fallback"
            ),
        ) as mock_resolve,
        patch("llm_task_router.router.claude_cli.invoke") as mock_invoke,
    ):
        decision = route(request)

    mock_embed.assert_called_once_with("put together a plan for the quarterly numbers")
    mock_resolve.assert_called_once_with(
        "put together a plan for the quarterly numbers", [0.1, 0.2]
    )
    mock_invoke.assert_not_called()  # resolved via tier 2 + grid, no tier-3 LLM call needed
    assert decision.tier == "cheap"  # code_gen x other = L
    assert "code_gen x other" in decision.reason
    assert "type tier2" in decision.reason


def test_route_falls_through_to_tier3_when_tier2_also_has_nothing():
    """Tier 2 declining (returns None) must still fall through to the
    existing no-signal tier-3 behavior, not crash or silently misroute."""
    request = TaskRequest(description="put together a plan for the quarterly numbers")
    with (
        patch("llm_task_router.router.embeddings.embed", return_value=[0.1, 0.2]),
        patch(
            "llm_task_router.router.tier2_classifier.resolve_task_type",
            return_value=None,
        ) as mock_resolve,
        patch(
            "llm_task_router.router.claude_cli.invoke",
            return_value=ProviderResult(text="MID", cost_usd=0.003, duration_ms=400),
        ) as mock_invoke,
    ):
        decision = route(request)

    mock_resolve.assert_called_once()
    mock_invoke.assert_called_once()
    assert decision.tier == "mid"


def test_route_uses_grid_normally_when_only_domain_axis_is_unresolved():
    """One resolved axis (a real task_type keyword match) is enough real
    signal to consult the grid normally, even if domain falls back to
    "other" - this is not the no-signal case."""
    request = TaskRequest(description="refactor this messy function")
    decision = route(request)

    assert (
        decision.tier == "cheap"
    )  # refactor x other = L, per the calibration-derived uniform-L row
    assert "heuristic grid" in decision.reason


# --- _shadow_tier1_only_decision (unit-level) -------------------------------
# These test the pure counterfactual function directly, isolated from route()
# and tier 2 entirely - the same coverage the "decision_log wiring" section
# below exercises end to end via route(), just without needing to mock
# tier2_classifier to force each branch.


def test_shadow_hedges_to_mid_when_both_axes_unresolved():
    classification = TaskClassification(
        task_type="architecture",
        domain="other",
        task_type_source="fallback",
        domain_source="fallback",
    )
    bias, tier, reason = _shadow_tier1_only_decision(
        classification, "reply with exactly the word: pong"
    )

    assert (bias, tier) == ("M", "mid")
    assert "no keyword signal on either axis" in reason


def test_shadow_uses_grid_directly_when_not_high_stakes_cell():
    classification = TaskClassification(
        task_type="refactor",
        domain="backend",
        task_type_source="inferred",
        domain_source="inferred",
    )
    bias, tier, reason = _shadow_tier1_only_decision(
        classification, "refactor this messy function"
    )

    assert (bias, tier) == ("L", "cheap")
    assert "heuristic grid" in reason


def test_shadow_caps_inferred_high_stakes_cell_at_mid_without_keyword_signal():
    classification = TaskClassification(
        task_type="architecture",
        domain="backend",
        task_type_source="inferred",
        domain_source="inferred",
    )
    bias, tier, reason = _shadow_tier1_only_decision(
        classification, "design a scalable system for this workload"
    )

    assert (bias, tier) == ("M", "mid")
    assert "no keyword corroboration" in reason


def test_shadow_matches_real_grid_when_keyword_corroborates_high_stakes():
    classification = TaskClassification(
        task_type="architecture",
        domain="backend",
        task_type_source="inferred",
        domain_source="inferred",
    )
    bias, tier, _reason = _shadow_tier1_only_decision(
        classification,
        "design the multi-region disaster recovery strategy for our payment processing system",
    )

    assert (bias, tier) == ("H", "flagship")


def test_shadow_trusts_an_explicit_caller_provided_type_without_corroboration():
    classification = TaskClassification(
        task_type="architecture",
        domain="backend",
        task_type_source="provided",
        domain_source="provided",
    )
    bias, tier, _reason = _shadow_tier1_only_decision(
        classification, "design a small internal tool"
    )

    assert (bias, tier) == ("H", "flagship")


# --- decision_log wiring ---------------------------------------------------
# route() logs every decision for drift auditing (see router.py's docstring
# and decision_log.py) - these assert the logged fields on a representative
# sample of paths, using conftest.py's autouse `_noop_decision_log` fixture
# explicitly so a signature bug in the log_decision() call isn't silently
# swallowed by route()'s own try/except (which exists so a *real* Postgres
# failure never breaks routing, not to hide a wiring bug from tests).


def test_route_logs_pure_heuristic_decision_with_null_embedding(_noop_decision_log):
    request = TaskRequest(
        description="fix null pointer in login form",
        task_type="triage",
        domain="frontend",
    )
    route(request)

    _noop_decision_log.assert_called_once()
    args, kwargs = _noop_decision_log.call_args
    description, embedding = args
    assert description == "fix null pointer in login form"
    assert embedding is None  # no embeddings.embed() call on the pure-heuristic path
    assert kwargs["resolved_task_type"] == "triage"
    assert kwargs["task_type_source"] == "provided"
    assert kwargs["resolved_is_high_stakes"] is None
    assert kwargs["high_stakes_source"] is None
    assert kwargs["no_signal_llm_used"] is False
    assert kwargs["tier"] == "cheap"
    # provided task_type/domain means shadow and real agree - both hit the same grid cell.
    assert kwargs["shadow_bias"] == "L"
    assert kwargs["shadow_tier"] == "cheap"
    assert "shadow (no tier 2)" in kwargs["shadow_reason"]


def test_route_logs_tier2_nn_task_type_source(_noop_decision_log):
    request = TaskRequest(description="put together a plan for the quarterly numbers")
    with (
        patch("llm_task_router.router.embeddings.embed", return_value=[0.1, 0.2]),
        patch(
            "llm_task_router.router.tier2_classifier.resolve_task_type",
            return_value=tier2_classifier.Tier2Resolution(
                task_type="code_gen", source="nn"
            ),
        ),
        patch("llm_task_router.router.claude_cli.invoke") as mock_invoke,
    ):
        route(request)

    mock_invoke.assert_not_called()
    kwargs = _noop_decision_log.call_args.kwargs
    assert kwargs["task_type_source"] == "tier2_nn"
    assert kwargs["resolved_task_type"] == "code_gen"
    # A de-escalating divergence: without tier 2, both axes are unresolved
    # (domain also falls back to "other" here) and the shadow hedges to mid;
    # tier 2 rescues task_type to code_gen, which the grid maps to cheap.
    assert kwargs["tier"] == "cheap"
    assert kwargs["shadow_bias"] == "M"
    assert kwargs["shadow_tier"] == "mid"


def test_route_logs_keyword_corroborated_high_stakes(_noop_decision_log):
    request = TaskRequest(
        description="design the multi-region disaster recovery strategy for our payment processing system, "
        "given a strict compliance requirement"
    )
    route(request)

    kwargs = _noop_decision_log.call_args.kwargs
    assert kwargs["resolved_is_high_stakes"] is True
    assert kwargs["high_stakes_source"] == "keyword"
    assert kwargs["tier"] == "flagship"


def test_route_logs_tier2_llm_fallback_high_stakes_source(_noop_decision_log):
    request = TaskRequest(
        description="design a scalable, fault-tolerant system for this workload"
    )
    with (
        patch("llm_task_router.router.embeddings.embed", return_value=[0.1, 0.2]),
        patch(
            "llm_task_router.router.tier2_classifier.resolve_high_stakes",
            return_value=HighStakesResolution(
                is_high_stakes=True, source="llm_fallback"
            ),
        ),
    ):
        route(request)

    kwargs = _noop_decision_log.call_args.kwargs
    assert kwargs["resolved_is_high_stakes"] is True
    assert kwargs["high_stakes_source"] == "tier2_llm_fallback"
    assert kwargs["tier"] == "flagship"
    # This is the divergent case shadow evaluation exists to measure: tier 2
    # corroborated high stakes with no keyword signal, so the real decision
    # escalates to flagship while the tier-1-only shadow would have capped
    # at mid - never acted on, only logged (see decision_log.py).
    assert kwargs["shadow_bias"] == "M"
    assert kwargs["shadow_tier"] == "mid"
    assert "capped at mid" in kwargs["shadow_reason"]


def test_route_logs_unavailable_high_stakes_source(_noop_decision_log):
    request = TaskRequest(
        description="design a scalable, fault-tolerant system for this workload"
    )
    with (
        patch("llm_task_router.router.embeddings.embed", return_value=[0.1, 0.2]),
        patch(
            "llm_task_router.router.tier2_classifier.resolve_high_stakes",
            return_value=None,
        ),
    ):
        route(request)

    kwargs = _noop_decision_log.call_args.kwargs
    assert kwargs["resolved_is_high_stakes"] is None
    assert kwargs["high_stakes_source"] == "unavailable"
    assert kwargs["tier"] == "mid"


def test_route_logs_no_signal_llm_used(_noop_decision_log):
    request = TaskRequest(description="repeat this word: hello")
    with (
        patch("llm_task_router.router.embeddings.embed", return_value=[0.0]),
        patch(
            "llm_task_router.router.tier2_classifier.resolve_task_type",
            return_value=None,
        ),
        patch(
            "llm_task_router.router.claude_cli.invoke",
            return_value=ProviderResult(text="CHEAP", cost_usd=0.003, duration_ms=400),
        ),
    ):
        route(request)

    kwargs = _noop_decision_log.call_args.kwargs
    assert kwargs["no_signal_llm_used"] is True
    assert kwargs["tier"] == "cheap"
    # Shadow never invokes tier 3 either (see _shadow_tier1_only_decision's
    # docstring) - it always hedges the no-signal case straight to mid,
    # regardless of what the real tier-3 LLM call answered.
    assert kwargs["shadow_bias"] == "M"
    assert kwargs["shadow_tier"] == "mid"
    assert "no keyword signal on either axis" in kwargs["shadow_reason"]


def test_route_logging_failure_never_breaks_routing():
    """A real Postgres/decision_log failure must degrade to 'no audit trail
    for this call', never to a broken route - same discipline
    tier2_classifier's own Postgres/LLM calls already follow."""
    request = TaskRequest(
        description="fix null pointer in login form",
        task_type="triage",
        domain="frontend",
    )
    with patch(
        "llm_task_router.router.decision_log.log_decision",
        side_effect=RuntimeError("DATABASE_URL not set"),
    ):
        decision = route(request)

    assert decision.tier == "cheap"


def test_route_and_run_invokes_resolved_provider():
    request = TaskRequest(
        description="summarize this report", task_type="summarization", domain="other"
    )
    fake_result = ProviderResult(text="summary here", cost_usd=0.001, duration_ms=200)

    with patch(
        "llm_task_router.router.claude_cli.invoke", return_value=fake_result
    ) as mock_invoke:
        decision, result = route_and_run(request)

    mock_invoke.assert_called_once_with(
        "summarize this report", "haiku", session_id=None, on_event=None
    )
    assert decision.tier == "cheap"
    assert result is fake_result


def test_route_and_run_threads_session_id_through_to_invoke():
    request = TaskRequest(
        description="summarize this report",
        task_type="summarization",
        domain="other",
        session_id="fixed-uuid",
    )
    fake_result = ProviderResult(text="summary here", cost_usd=0.001, duration_ms=200)

    with patch(
        "llm_task_router.router.claude_cli.invoke", return_value=fake_result
    ) as mock_invoke:
        route_and_run(request)

    mock_invoke.assert_called_once_with(
        "summarize this report", "haiku", session_id="fixed-uuid", on_event=None
    )


def test_route_and_run_forwards_on_event_to_invoke():
    request = TaskRequest(
        description="summarize this report", task_type="summarization", domain="other"
    )
    fake_result = ProviderResult(text="summary here", cost_usd=0.001, duration_ms=200)
    on_event = lambda event: None  # noqa: E731

    with patch(
        "llm_task_router.router.claude_cli.invoke", return_value=fake_result
    ) as mock_invoke:
        route_and_run(request, on_event=on_event)

    assert mock_invoke.call_args.kwargs["on_event"] is on_event


def test_route_and_run_calls_on_decision_before_invoke_with_resolved_decision():
    """on_decision must fire before the (potentially long) provider call, not
    after - that's the whole point of adding it, so repl.py can print a
    routing header immediately instead of only once the full response
    lands."""
    request = TaskRequest(
        description="summarize this report", task_type="summarization", domain="other"
    )
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
