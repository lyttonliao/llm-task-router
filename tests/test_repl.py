import uuid
from unittest.mock import Mock, patch

import pytest

from llm_task_router.repl import (
    HELP_TEXT,
    check_provider_auth,
    chat_loop,
    ensure_provider_authenticated,
    format_response,
    main,
    routable_tiers,
    startup_auth_check,
)
from llm_task_router.schema import ProviderResult, RouteDecision
from llm_task_router.tiers import TIER_MODELS


def _fake_module(check_auth_result=(True, ""), check_auth_side_effect=None):
    module = Mock()
    if check_auth_side_effect is not None:
        module.check_auth = Mock(side_effect=check_auth_side_effect)
    else:
        module.check_auth = Mock(return_value=check_auth_result)
    return module


# --- check_provider_auth ---


def test_check_provider_auth_returns_false_reason_on_missing_cli():
    module = Mock()
    module.check_auth = Mock(side_effect=FileNotFoundError("no such file: claude"))

    authenticated, reason = check_provider_auth("claude", module)

    assert authenticated is False
    assert "could not run claude CLI" in reason


# --- ensure_provider_authenticated ---


def test_ensure_provider_authenticated_true_without_prompting_when_already_authenticated():
    module = _fake_module(check_auth_result=(True, ""))
    input_fn = Mock(side_effect=AssertionError("input should not be called"))

    result = ensure_provider_authenticated("claude", module, input_fn=input_fn, print_fn=lambda *a: None)

    assert result is True
    module.login.assert_not_called()


def test_ensure_provider_authenticated_logs_in_on_yes_and_rechecks():
    module = _fake_module(check_auth_side_effect=[(False, "not logged in"), (True, "")])
    input_fn = Mock(return_value="y")

    result = ensure_provider_authenticated("claude", module, input_fn=input_fn, print_fn=lambda *a: None)

    assert result is True
    module.login.assert_called_once()


def test_ensure_provider_authenticated_accepts_empty_answer_as_yes():
    module = _fake_module(check_auth_side_effect=[(False, "not logged in"), (True, "")])
    input_fn = Mock(return_value="")

    result = ensure_provider_authenticated("claude", module, input_fn=input_fn, print_fn=lambda *a: None)

    assert result is True
    module.login.assert_called_once()


def test_ensure_provider_authenticated_declines_on_no():
    module = _fake_module(check_auth_result=(False, "not logged in"))
    input_fn = Mock(return_value="n")

    result = ensure_provider_authenticated("claude", module, input_fn=input_fn, print_fn=lambda *a: None)

    assert result is False
    module.login.assert_not_called()


def test_ensure_provider_authenticated_treats_eof_as_decline():
    module = _fake_module(check_auth_result=(False, "not logged in"))
    input_fn = Mock(side_effect=EOFError())

    result = ensure_provider_authenticated("claude", module, input_fn=input_fn, print_fn=lambda *a: None)

    assert result is False
    module.login.assert_not_called()


def test_ensure_provider_authenticated_returns_false_when_login_does_not_fix_auth():
    module = _fake_module(check_auth_side_effect=[(False, "not logged in"), (False, "still bad")])
    input_fn = Mock(return_value="y")

    result = ensure_provider_authenticated("claude", module, input_fn=input_fn, print_fn=lambda *a: None)

    assert result is False
    module.login.assert_called_once()


def test_ensure_provider_authenticated_handles_login_raising_missing_cli():
    module = _fake_module(check_auth_side_effect=[(False, "not logged in")])
    module.login = Mock(side_effect=FileNotFoundError("no such file: codex"))
    input_fn = Mock(return_value="y")

    result = ensure_provider_authenticated("codex", module, input_fn=input_fn, print_fn=lambda *a: None)

    assert result is False
    module.login.assert_called_once()


# --- startup_auth_check ---


def test_startup_auth_check_preserves_provider_dict_order():
    call_order = []

    def fake_ensure(name, module, *, input_fn=input, print_fn=print):
        call_order.append(name)
        return name == "b"

    providers = {"a": object(), "b": object()}
    with patch("llm_task_router.repl.ensure_provider_authenticated", side_effect=fake_ensure):
        result = startup_auth_check(providers, print_fn=lambda *a: None)

    assert call_order == ["a", "b"]
    assert result == {"b"}


# --- routable_tiers ---


def test_routable_tiers_splits_by_authenticated_provider():
    tier_models = {"cheap": ("claude", "haiku"), "mid": ("codex", "gpt-5.5")}

    routable, unroutable = routable_tiers({"claude"}, tier_models)

    assert routable == {"cheap": ("claude", "haiku")}
    assert unroutable == {"mid": ("codex", "gpt-5.5")}


def test_routable_tiers_against_real_tier_models_with_only_codex_authenticated():
    """Pinned regression test for a real, documented limitation: TIER_MODELS
    maps every tier to "claude" today, so authenticating Codex only yields
    zero routable tiers. If tiers.py ever gets a Codex entry, this
    assertion changes - that's the intended signal the limitation note in
    CLAUDE.md needs revising."""
    routable, unroutable = routable_tiers({"codex"})

    assert routable == {}
    assert unroutable == TIER_MODELS


# --- format_response ---


def test_format_response_success_includes_model_indicator_and_verbatim_text_and_cost_footer():
    decision = RouteDecision(tier="cheap", provider="claude", model="haiku", reason="r")
    result = ProviderResult(text="hello there", cost_usd=0.0012, duration_ms=345)

    out = format_response(decision, result)

    assert "[claude/haiku, tier=cheap]" in out
    assert "hello there" in out
    assert "$0.0012" in out
    assert "345ms" in out


def test_format_response_error_path_omits_text_and_cost_shows_error():
    """Not an exact-equality check anymore: format_response wraps the header
    and error text in ANSI color codes (see tui.py) as of the 2026-07-27
    styling pass, so this asserts on substance (single line, no stray cost
    footer, error text present) rather than a byte-exact string."""
    decision = RouteDecision(tier="cheap", provider="claude", model="haiku", reason="r")
    result = ProviderResult(text="", cost_usd=0.0, duration_ms=0, error="auth check failed: not logged in")

    out = format_response(decision, result)

    assert "[claude/haiku, tier=cheap]" in out
    assert "error: auth check failed: not logged in" in out
    assert "\n" not in out
    assert "cost $" not in out


# --- chat_loop ---


def test_chat_loop_exits_on_slash_exit():
    with patch("llm_task_router.repl.route_and_run") as mock_route:
        chat_loop(input_fn=Mock(side_effect=["/exit"]), print_fn=lambda *a: None)
    mock_route.assert_not_called()


def test_chat_loop_exits_on_slash_quit():
    with patch("llm_task_router.repl.route_and_run") as mock_route:
        chat_loop(input_fn=Mock(side_effect=["/quit"]), print_fn=lambda *a: None)
    mock_route.assert_not_called()


def test_chat_loop_exits_cleanly_on_eof():
    with patch("llm_task_router.repl.route_and_run") as mock_route:
        chat_loop(input_fn=Mock(side_effect=EOFError()), print_fn=lambda *a: None)
    mock_route.assert_not_called()


def test_chat_loop_skips_blank_lines_without_routing():
    with patch("llm_task_router.repl.route_and_run") as mock_route:
        chat_loop(input_fn=Mock(side_effect=["", "   ", "/exit"]), print_fn=lambda *a: None)
    mock_route.assert_not_called()


def test_chat_loop_routes_plain_message_via_route_and_run():
    decision = RouteDecision(tier="cheap", provider="claude", model="haiku", reason="r")
    result = ProviderResult(text="hi", cost_usd=0.001, duration_ms=100)

    with patch("llm_task_router.repl.route_and_run", return_value=(decision, result)) as mock_route:
        chat_loop(input_fn=Mock(side_effect=["fix the bug", "/exit"]), print_fn=lambda *a: None)

    mock_route.assert_called_once()
    (request,), _ = mock_route.call_args
    assert request.description == "fix the bug"
    assert request.task_type is None
    assert request.domain is None
    assert uuid.UUID(request.session_id)  # a real UUID string, not raising


def test_chat_loop_reuses_same_session_id_across_messages():
    """session_id is generated once per chat_loop() run, not per message -
    the whole point of threading it through is that conversation history
    continues even as different messages route to different tiers."""
    decision = RouteDecision(tier="cheap", provider="claude", model="haiku", reason="r")
    result = ProviderResult(text="hi", cost_usd=0.001, duration_ms=100)

    with patch("llm_task_router.repl.route_and_run", return_value=(decision, result)) as mock_route:
        chat_loop(input_fn=Mock(side_effect=["first", "second", "/exit"]), print_fn=lambda *a: None)

    assert mock_route.call_count == 2
    (request_1,), _ = mock_route.call_args_list[0]
    (request_2,), _ = mock_route.call_args_list[1]
    assert uuid.UUID(request_1.session_id)
    assert request_1.session_id == request_2.session_id


def test_chat_loop_prints_header_via_on_decision_before_result_is_known():
    """The header must come from on_decision (fired the instant routing
    resolves), not from the final result - route_and_run is faked here to
    invoke on_decision itself, mirroring what the real function does before
    the (potentially long) provider call starts."""
    decision = RouteDecision(tier="flagship", provider="claude", model="opus", reason="r")
    result = ProviderResult(text="hi", cost_usd=0.001, duration_ms=100)

    def fake_route_and_run(request, *, on_event=None, on_decision=None):
        on_decision(decision)
        return decision, result

    print_fn = Mock()
    with patch("llm_task_router.repl.route_and_run", side_effect=fake_route_and_run):
        chat_loop(input_fn=Mock(side_effect=["fix the bug", "/exit"]), print_fn=print_fn, write_fn=lambda t: None)

    assert any("claude/opus" in str(call) and "flagship" in str(call) for call in print_fn.call_args_list)


def test_chat_loop_streams_body_via_write_fn_and_does_not_reprint_it_via_print_fn():
    """The whole point of streaming: result.text must not be printed a
    second time via print_fn once it was already streamed live through
    write_fn."""
    decision = RouteDecision(tier="cheap", provider="claude", model="haiku", reason="r")
    result = ProviderResult(text="streamed answer", cost_usd=0.001, duration_ms=100)
    written = []

    def fake_route_and_run(request, *, on_event=None, on_decision=None):
        on_event(
            {
                "type": "stream_event",
                "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "streamed answer"}},
            }
        )
        return decision, result

    print_fn = Mock()
    with patch("llm_task_router.repl.route_and_run", side_effect=fake_route_and_run):
        chat_loop(input_fn=Mock(side_effect=["fix the bug", "/exit"]), print_fn=print_fn, write_fn=written.append)

    joined = "".join(written)
    assert "streamed answer" in joined
    assert joined.endswith("streamed answer\n")
    assert not any("streamed answer" in str(call) for call in print_fn.call_args_list)
    assert any("$0.0010" in str(call) or "0.0010" in str(call) for call in print_fn.call_args_list)


def test_chat_loop_error_path_prints_error_not_footer():
    decision = RouteDecision(tier="cheap", provider="claude", model="haiku", reason="r")
    result = ProviderResult(text="", cost_usd=0.0, duration_ms=0, error="auth check failed: not logged in")

    def fake_route_and_run(request, *, on_event=None, on_decision=None):
        on_decision(decision)
        return decision, result

    print_fn = Mock()
    with patch("llm_task_router.repl.route_and_run", side_effect=fake_route_and_run):
        chat_loop(input_fn=Mock(side_effect=["fix the bug", "/exit"]), print_fn=print_fn, write_fn=lambda t: None)

    assert any("error: auth check failed: not logged in" in str(call) for call in print_fn.call_args_list)


def test_chat_loop_prints_divider_between_turns_but_not_before_first_prompt():
    from llm_task_router import tui

    decision = RouteDecision(tier="cheap", provider="claude", model="haiku", reason="r")
    result = ProviderResult(text="hi", cost_usd=0.001, duration_ms=100)

    print_fn = Mock()
    with patch("llm_task_router.repl.route_and_run", return_value=(decision, result)):
        chat_loop(input_fn=Mock(side_effect=["first", "second", "/exit"]), print_fn=print_fn, write_fn=lambda t: None)

    divider_calls = [c for c in print_fn.call_args_list if c.args and tui.DIVIDER_CHAR in str(c.args[0])]
    # three prompts total (first, second, the /exit attempt) -> a divider
    # before every one of them except the very first
    assert len(divider_calls) == 2


def test_chat_loop_prints_blank_gap_after_user_line_before_header():
    decision = RouteDecision(tier="cheap", provider="claude", model="haiku", reason="r")
    result = ProviderResult(text="hi", cost_usd=0.001, duration_ms=100)

    def fake_route_and_run(request, *, on_event=None, on_decision=None):
        on_decision(decision)
        return decision, result

    print_fn = Mock()
    with patch("llm_task_router.repl.route_and_run", side_effect=fake_route_and_run):
        chat_loop(input_fn=Mock(side_effect=["fix the bug", "/exit"]), print_fn=print_fn, write_fn=lambda t: None)

    calls = [c.args[0] if c.args else "" for c in print_fn.call_args_list]
    header_index = next(i for i, c in enumerate(calls) if "claude/haiku" in c)
    assert calls[header_index - 1] == ""  # the blank-gap call immediately precedes the header


def test_chat_loop_no_blank_gap_for_help_command():
    """/help is answered on the same first turn (no divider preamble yet,
    since first_turn is still True) - the very first print_fn call should
    be the help text itself, with no blank-gap call ahead of it."""
    print_fn = Mock()
    with patch("llm_task_router.repl.route_and_run") as mock_route:
        chat_loop(input_fn=Mock(side_effect=["/help", "/exit"]), print_fn=print_fn, write_fn=lambda t: None)

    mock_route.assert_not_called()
    assert print_fn.call_args_list[0].args[0] == HELP_TEXT


def test_chat_loop_calls_renderer_start_from_on_decision_before_first_event():
    decision = RouteDecision(tier="cheap", provider="claude", model="haiku", reason="r")
    result = ProviderResult(text="hi", cost_usd=0.001, duration_ms=100)
    written = []

    def fake_route_and_run(request, *, on_event=None, on_decision=None):
        on_decision(decision)
        on_event(
            {
                "type": "stream_event",
                "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}},
            }
        )
        return decision, result

    with patch("llm_task_router.repl.route_and_run", side_effect=fake_route_and_run):
        chat_loop(input_fn=Mock(side_effect=["fix the bug", "/exit"]), print_fn=lambda *a: None, write_fn=written.append)

    joined = "".join(written)
    assert "connecting" in joined
    assert "\r\x1b[2K" in joined
    assert joined.index("connecting") < joined.index("\r\x1b[2K") < joined.index("hi")


def test_chat_loop_unknown_slash_command_prints_message_and_continues_loop():
    print_fn = Mock()
    with patch("llm_task_router.repl.route_and_run") as mock_route:
        chat_loop(input_fn=Mock(side_effect=["/bogus", "/exit"]), print_fn=print_fn)

    mock_route.assert_not_called()
    assert any("unknown command" in str(call) for call in print_fn.call_args_list)


def test_chat_loop_survives_unexpected_exception_from_route_and_run():
    print_fn = Mock()
    with patch("llm_task_router.repl.route_and_run", side_effect=ValueError("boom")) as mock_route:
        chat_loop(input_fn=Mock(side_effect=["first", "second", "/exit"]), print_fn=print_fn)

    assert mock_route.call_count == 2
    assert any("boom" in str(call) for call in print_fn.call_args_list)


# --- main ---


def test_main_exits_nonzero_when_no_providers_authenticated():
    with (
        patch("llm_task_router.repl.startup_auth_check", return_value=set()),
        patch("llm_task_router.repl.chat_loop") as mock_chat_loop,
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1
    mock_chat_loop.assert_not_called()


def test_main_exits_nonzero_when_no_tiers_routable():
    with (
        patch("llm_task_router.repl.startup_auth_check", return_value={"codex"}),
        patch("llm_task_router.repl.chat_loop") as mock_chat_loop,
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1
    mock_chat_loop.assert_not_called()


def test_main_enters_chat_loop_when_claude_authenticated():
    with (
        patch("llm_task_router.repl.startup_auth_check", return_value={"claude"}),
        patch("llm_task_router.repl.chat_loop") as mock_chat_loop,
    ):
        main()

    mock_chat_loop.assert_called_once_with()
