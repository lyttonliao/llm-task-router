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
    with patch("llm_task_router.providers.claude_cli.subprocess.run", return_value=_auth_fail()) as mock_run:
        result = invoke("do the task", model="haiku")

    assert result.error.startswith("auth check failed:")
    assert result.text == ""
    assert mock_run.call_count == 1  # only the auth check, never the real -p call


def test_builds_expected_command_without_cost_guardrail_flags_by_default():
    """Full-functionality adapter, not cost-minimized: no --disallowed-tools,
    no --strict-mcp-config, no unconditional --system-prompt "" - a real
    interactive client wants real tools/system-prompt/CLAUDE.md, unlike
    llm-eval-harness's stripped-down adapter. Tools instead run under
    --permission-mode bypassPermissions since there's no TTY here to show an
    approval prompt. Assert the exact command list, not just that
    subprocess.run was called."""
    payload = json.dumps({"result": "ok", "total_cost_usd": 0.001, "duration_ms": 500})
    with patch(
        "llm_task_router.providers.claude_cli.subprocess.run",
        side_effect=[_auth_ok(), _completed(stdout=payload)],
    ) as mock_run:
        invoke("do the task", model="haiku")

    args, kwargs = mock_run.call_args_list[1]
    cmd = args[0]
    assert cmd == [
        "claude",
        "-p",
        "do the task",
        "--model",
        "haiku",
        "--output-format",
        "json",
        "--permission-mode",
        "bypassPermissions",
    ]
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["timeout"] == 300


def test_invoke_omits_session_flags_when_session_id_not_provided():
    payload = json.dumps({"result": "ok", "total_cost_usd": 0.001, "duration_ms": 500})
    with patch(
        "llm_task_router.providers.claude_cli.subprocess.run",
        side_effect=[_auth_ok(), _completed(stdout=payload)],
    ) as mock_run:
        invoke("do the task", model="haiku")

    cmd = mock_run.call_args_list[1][0][0]
    assert "--session-id" not in cmd
    assert "--resume" not in cmd


def test_invoke_includes_session_id_flag_on_first_call():
    claude_cli_module._established_sessions.discard("sid-first")
    payload = json.dumps({"result": "ok", "total_cost_usd": 0.001, "duration_ms": 500})
    with patch(
        "llm_task_router.providers.claude_cli.subprocess.run",
        side_effect=[_auth_ok(), _completed(stdout=payload)],
    ) as mock_run:
        invoke("do the task", model="haiku", session_id="sid-first")

    cmd = mock_run.call_args_list[1][0][0]
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
    payload = json.dumps({"result": "ok", "total_cost_usd": 0.001, "duration_ms": 500})
    with patch(
        "llm_task_router.providers.claude_cli.subprocess.run",
        side_effect=[_auth_ok(), _completed(stdout=payload)],
    ) as mock_run:
        invoke("do the task", model="sonnet", session_id="sid-second")

    cmd = mock_run.call_args_list[1][0][0]
    assert "--resume" in cmd
    assert cmd[cmd.index("--resume") + 1] == "sid-second"
    assert "--session-id" not in cmd
    claude_cli_module._established_sessions.discard("sid-second")


def test_invoke_does_not_mark_session_established_when_call_fails():
    claude_cli_module._established_sessions.discard("sid-fail")
    with patch(
        "llm_task_router.providers.claude_cli.subprocess.run",
        side_effect=[_auth_ok(), _completed(returncode=1, stderr="boom")],
    ):
        invoke("do the task", model="haiku", session_id="sid-fail")

    assert "sid-fail" not in claude_cli_module._established_sessions


def test_invoke_includes_system_prompt_only_when_explicitly_set():
    payload = json.dumps({"result": "ok", "total_cost_usd": 0.001, "duration_ms": 500})
    with patch(
        "llm_task_router.providers.claude_cli.subprocess.run",
        side_effect=[_auth_ok(), _completed(stdout=payload)],
    ) as mock_run:
        invoke("do the task", model="haiku", system_prompt="be terse")

    cmd = mock_run.call_args_list[1][0][0]
    assert "--system-prompt" in cmd
    assert cmd[cmd.index("--system-prompt") + 1] == "be terse"


def test_success_path_parses_cost_duration_and_result():
    payload = json.dumps({"result": "the answer", "total_cost_usd": 0.0042, "duration_ms": 1234})
    with patch(
        "llm_task_router.providers.claude_cli.subprocess.run",
        side_effect=[_auth_ok(), _completed(stdout=payload)],
    ):
        result = invoke("do the task", model="haiku")

    assert result.text == "the answer"
    assert result.cost_usd == 0.0042
    assert result.duration_ms == 1234
    assert result.error == ""


def test_nonzero_exit_code_returns_error_from_stderr():
    with patch(
        "llm_task_router.providers.claude_cli.subprocess.run",
        side_effect=[_auth_ok(), _completed(returncode=1, stderr="rate limited")],
    ):
        result = invoke("do the task", model="haiku")

    assert result.error == "rate limited"
    assert result.text == ""


def test_timeout_expired_returns_timeout_error():
    with patch(
        "llm_task_router.providers.claude_cli.subprocess.run",
        side_effect=[_auth_ok(), subprocess.TimeoutExpired(cmd=["claude"], timeout=300)],
    ):
        result = invoke("do the task", model="haiku")

    assert result.error == "timeout"
    assert result.duration_ms == 300_000


def test_stdout_not_valid_json_returns_parse_error():
    with patch(
        "llm_task_router.providers.claude_cli.subprocess.run",
        side_effect=[_auth_ok(), _completed(stdout="not json")],
    ):
        result = invoke("do the task", model="haiku")

    assert result.error == "could not parse CLI json output"
    assert result.text == "not json"


def test_payload_is_error_true_returns_error_from_result_field():
    payload = json.dumps({"is_error": True, "result": "rate limited"})
    with patch(
        "llm_task_router.providers.claude_cli.subprocess.run",
        side_effect=[_auth_ok(), _completed(stdout=payload)],
    ):
        result = invoke("do the task", model="haiku")

    assert result.error == "rate limited"
    assert result.text == ""
