import os
import uuid
from unittest.mock import Mock, patch

import pytest

from llm_task_router.repl import (
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
    docs/rough-edges.md needs revising."""
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


def test_format_response_wraps_long_text_at_terminal_width(monkeypatch):
    import os
    import shutil

    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback: os.terminal_size((20, 24)))
    decision = RouteDecision(tier="cheap", provider="claude", model="haiku", reason="r")
    result = ProviderResult(text="this response text is long enough to need wrapping", cost_usd=0.0012, duration_ms=345)

    out = format_response(decision, result)

    body_lines = out.split("\n")[1:-1]  # strip header line and cost-footer line
    for line in body_lines:
        assert len(line) <= 20, f"line exceeds terminal width: '{line}' ({len(line)} chars)"


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
    with patch("llm_task_router.repl.route") as mock_route:
        chat_loop(input_fn=Mock(side_effect=["/exit"]), print_fn=lambda *a: None)
    mock_route.assert_not_called()


def test_chat_loop_exits_on_slash_quit():
    with patch("llm_task_router.repl.route") as mock_route:
        chat_loop(input_fn=Mock(side_effect=["/quit"]), print_fn=lambda *a: None)
    mock_route.assert_not_called()


def test_chat_loop_exits_cleanly_on_eof():
    with patch("llm_task_router.repl.route") as mock_route:
        chat_loop(input_fn=Mock(side_effect=EOFError()), print_fn=lambda *a: None)
    mock_route.assert_not_called()


def test_chat_loop_skips_blank_lines_without_routing():
    with patch("llm_task_router.repl.route") as mock_route:
        chat_loop(input_fn=Mock(side_effect=["", "   ", "/exit"]), print_fn=lambda *a: None)
    mock_route.assert_not_called()


def test_chat_loop_routes_plain_message_and_creates_session_then_attaches_then_sends():
    decision = RouteDecision(tier="cheap", provider="claude", model="haiku", reason="r")

    with (
        patch("llm_task_router.repl.route", return_value=decision) as mock_route,
        patch("llm_task_router.repl.terminal.create_session") as mock_create,
        patch("llm_task_router.repl.terminal.attach_terminal") as mock_attach,
        patch("llm_task_router.repl.terminal.send_message") as mock_send,
    ):
        chat_loop(input_fn=Mock(side_effect=["fix the bug", "/exit"]), print_fn=lambda *a: None)

    mock_route.assert_called_once()
    (request,), _ = mock_route.call_args
    assert request.description == "fix the bug"
    assert request.task_type is None
    assert request.domain is None
    assert uuid.UUID(request.session_id)  # a real UUID string, not raising

    mock_create.assert_called_once_with("claude", "haiku", request.session_id, os.getcwd())
    mock_attach.assert_called_once_with(request.session_id, os.getcwd())
    mock_send.assert_called_once_with(request.session_id, "fix the bug")


def test_chat_loop_does_not_block_reading_more_input_after_attaching():
    """attach_terminal() is non-blocking: chat_loop() must keep prompting
    for more messages right after it returns, not wait for or depend on
    the attached session ever exiting."""
    decision = RouteDecision(tier="cheap", provider="claude", model="haiku", reason="r")

    with (
        patch("llm_task_router.repl.route", return_value=decision),
        patch("llm_task_router.repl.terminal.create_session"),
        patch("llm_task_router.repl.terminal.attach_terminal") as mock_attach,
        patch("llm_task_router.repl.terminal.send_message"),
    ):
        chat_loop(input_fn=Mock(side_effect=["first", "second", "/exit"]), print_fn=lambda *a: None)

    # only the first message creates+attaches the session
    mock_attach.assert_called_once()


def test_chat_loop_reuses_same_session_id_across_messages():
    """session_id is generated once per chat_loop() run, not per message -
    the whole point of threading it through is that conversation history
    continues even as different messages route to different tiers."""
    decision = RouteDecision(tier="cheap", provider="claude", model="haiku", reason="r")

    with (
        patch("llm_task_router.repl.route", return_value=decision) as mock_route,
        patch("llm_task_router.repl.terminal.create_session"),
        patch("llm_task_router.repl.terminal.attach_terminal"),
        patch("llm_task_router.repl.terminal.send_message"),
    ):
        chat_loop(input_fn=Mock(side_effect=["first", "second", "/exit"]), print_fn=lambda *a: None)

    assert mock_route.call_count == 2
    (request_1,), _ = mock_route.call_args_list[0]
    (request_2,), _ = mock_route.call_args_list[1]
    assert uuid.UUID(request_1.session_id)
    assert request_1.session_id == request_2.session_id


def test_chat_loop_only_creates_and_attaches_once_across_messages():
    """The whole point of the persistent-session redesign: create_session()
    and attach_terminal() must run exactly once per chat_loop() run, no
    matter how many messages follow - every message after the first is
    delivered via send_message() into the already-running session."""
    decision = RouteDecision(tier="cheap", provider="claude", model="haiku", reason="r")

    with (
        patch("llm_task_router.repl.route", return_value=decision),
        patch("llm_task_router.repl.terminal.create_session") as mock_create,
        patch("llm_task_router.repl.terminal.attach_terminal") as mock_attach,
        patch("llm_task_router.repl.terminal.send_message") as mock_send,
    ):
        chat_loop(input_fn=Mock(side_effect=["first", "second", "third", "/exit"]), print_fn=lambda *a: None)

    mock_create.assert_called_once()
    mock_attach.assert_called_once()
    assert mock_send.call_count == 3


def test_chat_loop_switches_model_before_sending_when_tier_changes():
    """A routed model change mid-run must send /model (via switch_model())
    before the actual message, not spawn a new process."""
    decisions = [
        RouteDecision(tier="cheap", provider="claude", model="haiku", reason="r"),
        RouteDecision(tier="flagship", provider="claude", model="opus", reason="r"),
    ]
    events = []

    with (
        patch("llm_task_router.repl.route", side_effect=decisions),
        patch("llm_task_router.repl.terminal.create_session"),
        patch("llm_task_router.repl.terminal.attach_terminal"),
        patch(
            "llm_task_router.repl.terminal.switch_model",
            side_effect=lambda sid, model: events.append(("switch", model)),
        ),
        patch(
            "llm_task_router.repl.terminal.send_message",
            side_effect=lambda sid, text: events.append(("send", text)),
        ),
    ):
        chat_loop(input_fn=Mock(side_effect=["first", "second", "/exit"]), print_fn=lambda *a: None)

    assert events == [
        ("send", "first"),
        ("switch", "opus"),
        ("send", "second"),
    ]


def test_chat_loop_does_not_switch_model_when_tier_is_unchanged():
    decision = RouteDecision(tier="cheap", provider="claude", model="haiku", reason="r")

    with (
        patch("llm_task_router.repl.route", return_value=decision),
        patch("llm_task_router.repl.terminal.create_session"),
        patch("llm_task_router.repl.terminal.attach_terminal"),
        patch("llm_task_router.repl.terminal.switch_model") as mock_switch,
        patch("llm_task_router.repl.terminal.send_message"),
    ):
        chat_loop(input_fn=Mock(side_effect=["first", "second", "/exit"]), print_fn=lambda *a: None)

    mock_switch.assert_not_called()


def test_chat_loop_prints_sent_to_session_message_after_successful_send():
    decision = RouteDecision(tier="cheap", provider="claude", model="haiku", reason="r")
    print_fn = Mock()

    with (
        patch("llm_task_router.repl.route", return_value=decision),
        patch("llm_task_router.repl.terminal.create_session"),
        patch("llm_task_router.repl.terminal.attach_terminal"),
        patch("llm_task_router.repl.terminal.send_message"),
    ):
        chat_loop(input_fn=Mock(side_effect=["first", "/exit"]), print_fn=print_fn)

    assert any("sent to session" in str(call) for call in print_fn.call_args_list)


def test_chat_loop_prints_header_before_creating_session():
    decision = RouteDecision(tier="flagship", provider="claude", model="opus", reason="r")
    events = []

    print_fn = Mock(side_effect=lambda *a: events.append(("print", a[0] if a else "")))
    with (
        patch("llm_task_router.repl.route", return_value=decision),
        patch(
            "llm_task_router.repl.terminal.create_session",
            side_effect=lambda *a, **k: events.append(("create", a)),
        ),
        patch("llm_task_router.repl.terminal.attach_terminal"),
        patch("llm_task_router.repl.terminal.send_message"),
    ):
        chat_loop(input_fn=Mock(side_effect=["fix the bug", "/exit"]), print_fn=print_fn)

    header_index = next(i for i, (kind, payload) in enumerate(events) if kind == "print" and "claude/opus" in payload)
    create_index = next(i for i, (kind, _) in enumerate(events) if kind == "create")
    assert header_index < create_index
    assert "flagship" in events[header_index][1]


def test_chat_loop_unknown_slash_command_prints_message_and_continues_loop():
    print_fn = Mock()
    with patch("llm_task_router.repl.route") as mock_route:
        chat_loop(input_fn=Mock(side_effect=["/bogus", "/exit"]), print_fn=print_fn)

    mock_route.assert_not_called()
    assert any("unknown command" in str(call) for call in print_fn.call_args_list)


def test_chat_loop_survives_unexpected_exception_from_route():
    print_fn = Mock()
    with (
        patch("llm_task_router.repl.route", side_effect=ValueError("boom")) as mock_route,
        patch("llm_task_router.repl.terminal.create_session") as mock_create,
    ):
        chat_loop(input_fn=Mock(side_effect=["first", "second", "/exit"]), print_fn=print_fn)

    assert mock_route.call_count == 2
    assert any("boom" in str(call) for call in print_fn.call_args_list)
    mock_create.assert_not_called()


def test_chat_loop_survives_exception_from_create_session():
    """A session-creation failure must not crash the loop, and since the
    session never actually got created, the next message must still try
    to create it (not skip straight to send_message() against a session
    that doesn't exist)."""
    decision = RouteDecision(tier="cheap", provider="claude", model="haiku", reason="r")
    print_fn = Mock()

    with (
        patch("llm_task_router.repl.route", return_value=decision),
        patch(
            "llm_task_router.repl.terminal.create_session",
            side_effect=[RuntimeError("no terminal emulator found"), None],
        ) as mock_create,
        patch("llm_task_router.repl.terminal.attach_terminal") as mock_attach,
        patch("llm_task_router.repl.terminal.send_message") as mock_send,
    ):
        chat_loop(input_fn=Mock(side_effect=["first", "second", "/exit"]), print_fn=print_fn)

    assert mock_create.call_count == 2
    session_id_1 = mock_create.call_args_list[0].args[2]
    session_id_2 = mock_create.call_args_list[1].args[2]
    assert session_id_1 == session_id_2
    # first attempt failed before attach/send ever ran; second succeeded
    mock_attach.assert_called_once()
    mock_send.assert_called_once_with(session_id_2, "second")
    assert any("no terminal emulator found" in str(call) for call in print_fn.call_args_list)


def test_chat_loop_survives_exception_from_attach_terminal_without_recreating_session():
    """Real live failure (2026-07-31): create_session() can succeed while
    attach_terminal() fails to actually display anything. A retry must not
    call create_session() again with the same session_id (tmux rejects a
    duplicate new-session name) - it must fall through to send_message()
    against the session that already exists."""
    decision = RouteDecision(tier="cheap", provider="claude", model="haiku", reason="r")
    print_fn = Mock()

    with (
        patch("llm_task_router.repl.route", return_value=decision),
        patch("llm_task_router.repl.terminal.create_session") as mock_create,
        patch(
            "llm_task_router.repl.terminal.attach_terminal",
            side_effect=RuntimeError("no terminal emulator found"),
        ) as mock_attach,
        patch("llm_task_router.repl.terminal.send_message") as mock_send,
    ):
        chat_loop(input_fn=Mock(side_effect=["first", "second", "/exit"]), print_fn=print_fn)

    mock_create.assert_called_once()  # not retried on the second message
    mock_attach.assert_called_once()  # not retried either - already "created"
    mock_send.assert_called_once_with(mock_create.call_args.args[2], "second")
    assert any("no terminal emulator found" in str(call) for call in print_fn.call_args_list)


def test_chat_loop_prints_tmux_session_id_and_manual_attach_fallback():
    decision = RouteDecision(tier="cheap", provider="claude", model="haiku", reason="r")
    print_fn = Mock()

    with (
        patch("llm_task_router.repl.route", return_value=decision) as mock_route,
        patch("llm_task_router.repl.terminal.create_session"),
        patch("llm_task_router.repl.terminal.attach_terminal"),
        patch("llm_task_router.repl.terminal.send_message"),
    ):
        chat_loop(input_fn=Mock(side_effect=["first", "/exit"]), print_fn=print_fn)

    (request,), _ = mock_route.call_args
    printed = [str(c) for c in print_fn.call_args_list]
    assert any(request.session_id in c and "tmux attach -t" in c for c in printed)


def test_chat_loop_prints_divider_between_retries_but_not_before_first_prompt():
    """Dividers separate re-prompts on paths that never spawned a terminal
    (here: two unknown commands in a row before /exit) - there's no
    "between successful turns" case here since these never route at all."""
    from llm_task_router import tui

    print_fn = Mock()
    with patch("llm_task_router.repl.route") as mock_route:
        chat_loop(input_fn=Mock(side_effect=["/bogus", "/also-bogus", "/exit"]), print_fn=print_fn)

    mock_route.assert_not_called()
    divider_calls = [c for c in print_fn.call_args_list if c.args and tui.DIVIDER_CHAR in str(c.args[0])]
    # three prompts total (/bogus, /also-bogus, /exit) -> a divider before
    # every one of them except the very first
    assert len(divider_calls) == 2


def test_chat_loop_prints_blank_gap_after_user_line_before_header():
    decision = RouteDecision(tier="cheap", provider="claude", model="haiku", reason="r")

    print_fn = Mock()
    with (
        patch("llm_task_router.repl.route", return_value=decision),
        patch("llm_task_router.repl.terminal.create_session"),
        patch("llm_task_router.repl.terminal.attach_terminal"),
        patch("llm_task_router.repl.terminal.send_message"),
    ):
        chat_loop(input_fn=Mock(side_effect=["fix the bug", "/exit"]), print_fn=print_fn)

    calls = [c.args[0] if c.args else "" for c in print_fn.call_args_list]
    header_index = next(i for i, c in enumerate(calls) if "claude/haiku" in c)
    assert calls[header_index - 1] == ""  # the blank-gap call immediately precedes the header


# --- main ---


def test_main_exits_nonzero_when_tmux_not_available():
    with (
        patch("llm_task_router.repl.terminal.tmux_available", return_value=False),
        patch("llm_task_router.repl.startup_auth_check") as mock_auth_check,
        patch("llm_task_router.repl.chat_loop") as mock_chat_loop,
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1
    mock_auth_check.assert_not_called()
    mock_chat_loop.assert_not_called()


def test_main_exits_nonzero_when_no_providers_authenticated():
    with (
        patch("llm_task_router.repl.terminal.tmux_available", return_value=True),
        patch("llm_task_router.repl.startup_auth_check", return_value=set()),
        patch("llm_task_router.repl.chat_loop") as mock_chat_loop,
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1
    mock_chat_loop.assert_not_called()


def test_main_exits_nonzero_when_no_tiers_routable():
    with (
        patch("llm_task_router.repl.terminal.tmux_available", return_value=True),
        patch("llm_task_router.repl.startup_auth_check", return_value={"codex"}),
        patch("llm_task_router.repl.chat_loop") as mock_chat_loop,
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1
    mock_chat_loop.assert_not_called()


def test_main_enters_chat_loop_when_claude_authenticated():
    """chat_loop() is called with an explicit input_fn - build_input_fn()'s
    prompt_toolkit-backed callable, not the bare-input() default - so main()
    is asserted against by call count/kwarg presence rather than a fixed
    input_fn identity (constructing a real PromptSession per call is what
    build_input_fn() is for)."""
    with (
        patch("llm_task_router.repl.terminal.tmux_available", return_value=True),
        patch("llm_task_router.repl.startup_auth_check", return_value={"claude"}),
        patch("llm_task_router.repl.chat_loop") as mock_chat_loop,
    ):
        main()

    mock_chat_loop.assert_called_once()
    assert callable(mock_chat_loop.call_args.kwargs.get("input_fn"))
