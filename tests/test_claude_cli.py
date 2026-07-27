import json
import subprocess
from unittest.mock import patch

from llm_task_router.providers import claude_cli as claude_cli_module
from llm_task_router.providers.claude_cli import check_auth, invoke, login


def _completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(args=["claude"], returncode=returncode, stdout=stdout, stderr=stderr)


def _auth_ok():
    return _completed(stdout=json.dumps({"loggedIn": True}))


def _auth_fail():
    return _completed(stdout=json.dumps({"loggedIn": False}))


class _FakeStream:
    """Stand-in for a Popen pipe's text-mode file object: readline() returns
    queued lines in order, then "" forever (EOF) - exactly the contract
    claude_cli._drain() depends on."""

    def __init__(self, lines=()):
        self._lines = list(lines)

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        return ""


class _FakeProcess:
    """Stand-in for subprocess.Popen. Never spawns a real process - patched
    in place of subprocess.Popen for every invoke() test in this file so a
    mocking gap here can't silently fall through to a real, billed `claude
    -p` call the way switching this adapter from subprocess.run to Popen
    once did during development."""

    def __init__(self, stdout_lines=(), stderr_lines=(), returncode=0):
        self.stdout = _FakeStream(stdout_lines)
        self.stderr = _FakeStream(stderr_lines)
        self.returncode = returncode
        self.killed = False

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.killed = True


def _event_line(**payload) -> str:
    return json.dumps(payload) + "\n"


def _result_line(result="ok", total_cost_usd=0.001, duration_ms=500, is_error=False) -> str:
    return _event_line(type="result", result=result, total_cost_usd=total_cost_usd, duration_ms=duration_ms, is_error=is_error)


def _always_ready(select_patch=None):
    """select.select() side_effect that reports every requested stream as
    immediately ready - correct for tests where every fake stream already
    has all its lines queued up front, so there's nothing to actually wait
    on."""
    return lambda rlist, wlist, xlist, timeout: (rlist, [], [])


def test_login_shells_expected_command_with_inherited_stdio():
    """--claudeai (not --console) is the subscription flow, not the API-key
    one - a regression here would silently point users at billed API auth.
    Asserting the exact call_args (positional cmd only, no kwargs) is the
    regression guard for stdio inheritance: capture_output/text/timeout must
    stay absent so the user can actually complete the OAuth flow in the
    terminal instead of it being captured and discarded."""
    with patch("llm_task_router.providers.claude_cli.subprocess.run", return_value=_completed()) as mock_run:
        login()

    assert mock_run.call_args == ((["claude", "auth", "login", "--claudeai"],), {})


def test_login_returns_child_returncode():
    with patch("llm_task_router.providers.claude_cli.subprocess.run", return_value=_completed(returncode=1)):
        assert login() == 1


def test_check_auth_true_when_logged_in():
    with patch("llm_task_router.providers.claude_cli.subprocess.run", return_value=_auth_ok()) as mock_run:
        authenticated, error = check_auth()

    assert authenticated is True
    assert error == ""
    assert mock_run.call_args[0][0] == ["claude", "auth", "status", "--json"]


def test_check_auth_false_when_logged_in_key_false():
    with patch("llm_task_router.providers.claude_cli.subprocess.run", return_value=_auth_fail()):
        authenticated, error = check_auth()

    assert authenticated is False
    assert "not logged in" in error


def test_check_auth_false_against_real_observed_logged_out_shape():
    """Regression guard for the real logged-out payload, not a guess: `env -u
    ANTHROPIC_API_KEY claude --bare auth status` (2026-07-26) - --bare mode
    skips keychain/OAuth reads per `claude --help`, so this exercises the real
    "no credentials resolved" response without touching this account's actual
    stored login."""
    payload = json.dumps({"loggedIn": False, "authMethod": "none", "apiProvider": "firstParty"})
    with patch(
        "llm_task_router.providers.claude_cli.subprocess.run",
        return_value=_completed(stdout=payload, returncode=1),
    ):
        authenticated, error = check_auth()

    assert authenticated is False
    assert "not logged in" in error


def test_check_auth_false_on_unparseable_output():
    with patch("llm_task_router.providers.claude_cli.subprocess.run", return_value=_completed(stdout="not json")):
        authenticated, error = check_auth()

    assert authenticated is False
    assert "could not parse" in error


def test_check_auth_false_on_timeout():
    with patch(
        "llm_task_router.providers.claude_cli.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["claude"], timeout=10),
    ):
        authenticated, error = check_auth()

    assert authenticated is False
    assert "timed out" in error


def test_invoke_short_circuits_and_never_calls_model_when_not_authenticated():
    """The whole point of the auth gate: an unauthenticated run should never
    reach the real `claude -p` call, since that's exactly the shape of the
    bogus-run failure mode CLAUDE.md warns about (every case scoring a
    misleading parse_error instead of a clear, single auth error)."""
    with (
        patch("llm_task_router.providers.claude_cli.subprocess.run", return_value=_auth_fail()),
        patch("llm_task_router.providers.claude_cli.subprocess.Popen") as mock_popen,
    ):
        result = invoke("do the task", model="haiku")

    assert result.error.startswith("auth check failed:")
    assert result.text == ""
    mock_popen.assert_not_called()


def test_builds_expected_command_without_cost_guardrail_flags_by_default():
    """Full-functionality adapter, not cost-minimized: no --disallowed-tools,
    no --strict-mcp-config, no unconditional --system-prompt "" - a real
    interactive client wants real tools/system-prompt/CLAUDE.md, unlike
    llm-eval-harness's stripped-down adapter. Tools instead run under
    --permission-mode bypassPermissions since there's no TTY here to show an
    approval prompt. Assert the exact command list, not just that
    subprocess.Popen was called. stream-json requires --verbose (confirmed
    against real `claude --help`/behavior) and --include-partial-messages is
    what makes assistant text arrive as incremental deltas rather than one
    blob at the end."""
    fake_proc = _FakeProcess(stdout_lines=[_result_line()])
    with (
        patch("llm_task_router.providers.claude_cli.subprocess.run", return_value=_auth_ok()),
        patch("llm_task_router.providers.claude_cli.subprocess.Popen", return_value=fake_proc) as mock_popen,
        patch("llm_task_router.providers.claude_cli.select.select", side_effect=_always_ready()),
    ):
        invoke("do the task", model="haiku")

    args, kwargs = mock_popen.call_args
    cmd = args[0]
    assert cmd == [
        "claude",
        "-p",
        "do the task",
        "--model",
        "haiku",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--verbose",
        "--permission-mode",
        "bypassPermissions",
    ]
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.PIPE
    assert kwargs["text"] is True


def test_invoke_omits_session_flags_when_session_id_not_provided():
    fake_proc = _FakeProcess(stdout_lines=[_result_line()])
    with (
        patch("llm_task_router.providers.claude_cli.subprocess.run", return_value=_auth_ok()),
        patch("llm_task_router.providers.claude_cli.subprocess.Popen", return_value=fake_proc) as mock_popen,
        patch("llm_task_router.providers.claude_cli.select.select", side_effect=_always_ready()),
    ):
        invoke("do the task", model="haiku")

    cmd = mock_popen.call_args[0][0]
    assert "--session-id" not in cmd
    assert "--resume" not in cmd


def test_invoke_includes_session_id_flag_on_first_call():
    claude_cli_module._established_sessions.discard("sid-first")
    fake_proc = _FakeProcess(stdout_lines=[_result_line()])
    with (
        patch("llm_task_router.providers.claude_cli.subprocess.run", return_value=_auth_ok()),
        patch("llm_task_router.providers.claude_cli.subprocess.Popen", return_value=fake_proc) as mock_popen,
        patch("llm_task_router.providers.claude_cli.select.select", side_effect=_always_ready()),
    ):
        invoke("do the task", model="haiku", session_id="sid-first")

    cmd = mock_popen.call_args[0][0]
    assert "--session-id" in cmd
    assert cmd[cmd.index("--session-id") + 1] == "sid-first"
    assert "--resume" not in cmd
    assert "sid-first" in claude_cli_module._established_sessions
    claude_cli_module._established_sessions.discard("sid-first")


def test_invoke_uses_resume_flag_once_session_already_established():
    """Confirmed against real `claude` output (2026-07-26): reusing
    --session-id on a second call FAILS ("Session ID ... is already in
    use"); --resume is the correct flag for every call after the first, and
    it does continue the conversation under a different --model."""
    claude_cli_module._established_sessions.add("sid-second")
    fake_proc = _FakeProcess(stdout_lines=[_result_line()])
    with (
        patch("llm_task_router.providers.claude_cli.subprocess.run", return_value=_auth_ok()),
        patch("llm_task_router.providers.claude_cli.subprocess.Popen", return_value=fake_proc) as mock_popen,
        patch("llm_task_router.providers.claude_cli.select.select", side_effect=_always_ready()),
    ):
        invoke("do the task", model="sonnet", session_id="sid-second")

    cmd = mock_popen.call_args[0][0]
    assert "--resume" in cmd
    assert cmd[cmd.index("--resume") + 1] == "sid-second"
    assert "--session-id" not in cmd
    claude_cli_module._established_sessions.discard("sid-second")


def test_invoke_does_not_mark_session_established_when_call_fails():
    claude_cli_module._established_sessions.discard("sid-fail")
    fake_proc = _FakeProcess(stdout_lines=[], stderr_lines=["boom\n"], returncode=1)
    with (
        patch("llm_task_router.providers.claude_cli.subprocess.run", return_value=_auth_ok()),
        patch("llm_task_router.providers.claude_cli.subprocess.Popen", return_value=fake_proc),
        patch("llm_task_router.providers.claude_cli.select.select", side_effect=_always_ready()),
    ):
        invoke("do the task", model="haiku", session_id="sid-fail")

    assert "sid-fail" not in claude_cli_module._established_sessions


def test_invoke_includes_system_prompt_only_when_explicitly_set():
    fake_proc = _FakeProcess(stdout_lines=[_result_line()])
    with (
        patch("llm_task_router.providers.claude_cli.subprocess.run", return_value=_auth_ok()),
        patch("llm_task_router.providers.claude_cli.subprocess.Popen", return_value=fake_proc) as mock_popen,
        patch("llm_task_router.providers.claude_cli.select.select", side_effect=_always_ready()),
    ):
        invoke("do the task", model="haiku", system_prompt="be terse")

    cmd = mock_popen.call_args[0][0]
    assert "--system-prompt" in cmd
    assert cmd[cmd.index("--system-prompt") + 1] == "be terse"


def test_success_path_parses_cost_duration_and_result():
    fake_proc = _FakeProcess(stdout_lines=[_result_line(result="the answer", total_cost_usd=0.0042, duration_ms=1234)])
    with (
        patch("llm_task_router.providers.claude_cli.subprocess.run", return_value=_auth_ok()),
        patch("llm_task_router.providers.claude_cli.subprocess.Popen", return_value=fake_proc),
        patch("llm_task_router.providers.claude_cli.select.select", side_effect=_always_ready()),
    ):
        result = invoke("do the task", model="haiku")

    assert result.text == "the answer"
    assert result.cost_usd == 0.0042
    assert result.duration_ms == 1234
    assert result.error == ""


def test_nonzero_exit_code_returns_error_from_stderr():
    fake_proc = _FakeProcess(stdout_lines=[], stderr_lines=["rate limited"], returncode=1)
    with (
        patch("llm_task_router.providers.claude_cli.subprocess.run", return_value=_auth_ok()),
        patch("llm_task_router.providers.claude_cli.subprocess.Popen", return_value=fake_proc),
        patch("llm_task_router.providers.claude_cli.select.select", side_effect=_always_ready()),
    ):
        result = invoke("do the task", model="haiku")

    assert result.error == "rate limited"
    assert result.text == ""


def test_timeout_returns_timeout_error_without_ever_calling_readline_past_deadline():
    """Popen has no timeout= kwarg for an incrementally-read stream, so the
    deadline is enforced by comparing time.monotonic() against a target
    computed at the start of _drain() - mock monotonic to jump straight past
    that deadline on the very first loop check, with select.select reporting
    nothing ready (a real hang would never make a stream ready), so this
    exercises the deadline branch deterministically instead of a real 300s
    wait."""
    fake_proc = _FakeProcess(stdout_lines=[], stderr_lines=[])
    with (
        patch("llm_task_router.providers.claude_cli.subprocess.run", return_value=_auth_ok()),
        patch("llm_task_router.providers.claude_cli.subprocess.Popen", return_value=fake_proc),
        patch("llm_task_router.providers.claude_cli.select.select", return_value=([], [], [])),
        patch("llm_task_router.providers.claude_cli.time.monotonic", side_effect=[0, 1000]),
    ):
        result = invoke("do the task", model="haiku")

    assert result.error == "timeout"
    assert result.duration_ms == 300_000
    assert fake_proc.killed is True


def test_no_result_event_received_returns_clear_error():
    """Replaces the old single-blob "could not parse CLI json output" test:
    with NDJSON, a stray non-JSON or malformed line is just skipped (not a
    hard error, see _drain()'s except json.JSONDecodeError: continue) - what
    actually surfaces as an error now is reaching EOF on a 0-exit process
    without ever seeing a `type: result` line."""
    fake_proc = _FakeProcess(stdout_lines=["not json\n"], stderr_lines=[], returncode=0)
    with (
        patch("llm_task_router.providers.claude_cli.subprocess.run", return_value=_auth_ok()),
        patch("llm_task_router.providers.claude_cli.subprocess.Popen", return_value=fake_proc),
        patch("llm_task_router.providers.claude_cli.select.select", side_effect=_always_ready()),
    ):
        result = invoke("do the task", model="haiku")

    assert result.error == "no result event received"
    assert result.text == ""


def test_payload_is_error_true_returns_error_from_result_field():
    fake_proc = _FakeProcess(stdout_lines=[_result_line(result="rate limited", is_error=True)])
    with (
        patch("llm_task_router.providers.claude_cli.subprocess.run", return_value=_auth_ok()),
        patch("llm_task_router.providers.claude_cli.subprocess.Popen", return_value=fake_proc),
        patch("llm_task_router.providers.claude_cli.select.select", side_effect=_always_ready()),
    ):
        result = invoke("do the task", model="haiku")

    assert result.error == "rate limited"
    assert result.text == ""


def test_on_event_called_once_per_parsed_stream_event_in_order():
    """Regression guard for repl.py's live streaming: on_event must see
    every real event (in arrival order) except the raw stderr lines, which
    aren't JSON and were never part of the event stream's contract."""
    events_in = [
        {"type": "system", "subtype": "init"},
        {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}}},
        {"type": "result", "result": "hi", "total_cost_usd": 0.001, "duration_ms": 10, "is_error": False},
    ]
    fake_proc = _FakeProcess(stdout_lines=[json.dumps(e) + "\n" for e in events_in])
    seen = []
    with (
        patch("llm_task_router.providers.claude_cli.subprocess.run", return_value=_auth_ok()),
        patch("llm_task_router.providers.claude_cli.subprocess.Popen", return_value=fake_proc),
        patch("llm_task_router.providers.claude_cli.select.select", side_effect=_always_ready()),
    ):
        invoke("do the task", model="haiku", on_event=seen.append)

    assert seen == events_in


def test_on_event_exception_does_not_break_invoke():
    """A bug in a terminal renderer must not lose an already-in-flight,
    already-paid-for model call - see _drain()'s deliberate broad catch."""
    fake_proc = _FakeProcess(stdout_lines=[_result_line(result="the answer")])
    with (
        patch("llm_task_router.providers.claude_cli.subprocess.run", return_value=_auth_ok()),
        patch("llm_task_router.providers.claude_cli.subprocess.Popen", return_value=fake_proc),
        patch("llm_task_router.providers.claude_cli.select.select", side_effect=_always_ready()),
    ):
        result = invoke("do the task", model="haiku", on_event=lambda e: (_ for _ in ()).throw(ValueError("boom")))

    assert result.text == "the answer"
    assert result.error == ""
