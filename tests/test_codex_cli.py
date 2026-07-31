import subprocess
from pathlib import Path
from unittest.mock import patch

from llm_task_router.providers.codex_cli import check_auth, invoke, login


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=["codex"], returncode=returncode, stdout=stdout, stderr=stderr)


def _auth_ok():
    return _completed(stdout="Logged in using ChatGPT\n")


def _auth_fail():
    """Matches the real observed logged-out shape: `CODEX_HOME=<empty dir>
    codex login status` (2026-07-26) - pointing CODEX_HOME at a directory
    with no stored credentials, without touching this account's actual
    ~/.codex login - prints plain text "Not logged in" at exit 1."""
    return _completed(stdout="Not logged in\n", returncode=1)


def _with_auth_ok(fake_run):
    """Wrap a fake_run(cmd, **kwargs) that only knows how to handle the real
    `codex exec` call so it also answers the `codex login status` pre-flight
    check invoke() now makes first."""

    def wrapped(cmd, **kwargs):
        if cmd[:3] == ["codex", "login", "status"]:
            return _auth_ok()
        return fake_run(cmd, **kwargs)

    return wrapped


def test_login_shells_default_command_with_inherited_stdio():
    """No capture_output/text/timeout - stdio must stay inherited so the
    user can actually complete the browser flow in the terminal."""
    with patch("llm_task_router.providers.codex_cli.subprocess.run", return_value=_completed()) as mock_run:
        login()

    assert mock_run.call_args == ((["codex", "login"],), {})


def test_login_with_device_auth_appends_flag():
    with patch("llm_task_router.providers.codex_cli.subprocess.run", return_value=_completed()) as mock_run:
        login(device_auth=True)

    assert mock_run.call_args == ((["codex", "login", "--device-auth"],), {})


def test_login_returns_child_returncode():
    with patch("llm_task_router.providers.codex_cli.subprocess.run", return_value=_completed(returncode=1)):
        assert login() == 1


def test_check_auth_true_when_logged_in():
    with patch("llm_task_router.providers.codex_cli.subprocess.run", return_value=_auth_ok()) as mock_run:
        authenticated, error = check_auth()

    assert authenticated is True
    assert error == ""
    assert mock_run.call_args[0][0] == ["codex", "login", "status"]


def test_check_auth_false_when_not_logged_in():
    with patch("llm_task_router.providers.codex_cli.subprocess.run", return_value=_auth_fail()):
        authenticated, error = check_auth()

    assert authenticated is False
    assert "not logged in" in error.lower()


def test_check_auth_false_on_timeout():
    with patch(
        "llm_task_router.providers.codex_cli.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["codex"], timeout=10),
    ):
        authenticated, error = check_auth()

    assert authenticated is False
    assert "timed out" in error


def test_invoke_short_circuits_and_never_calls_model_when_not_authenticated():
    """Same rationale as claude_cli.py's equivalent test: an unauthenticated
    run must never reach the real `codex exec` call."""
    with patch("llm_task_router.providers.codex_cli.subprocess.run", return_value=_auth_fail()) as mock_run:
        result = invoke("do the task", model="gpt-5")

    assert result.error.startswith("auth check failed:")
    assert result.text == ""
    assert mock_run.call_count == 1  # only the auth check, never the real exec call


def test_builds_expected_command_with_readonly_sandbox_flag():
    """--sandbox read-only is the closest analog to claude_cli.py's
    --disallowed-tools "*" - a regression here would let the router shell out
    with write access. Note --ask-for-approval is deliberately absent: it's a
    top-level `codex` option, not a valid `codex exec` flag (confirmed by a
    real run failing with "unexpected argument" when it was passed) - exec
    auto-defaults to never-ask on its own. Assert the exact command list,
    matching test_claude_cli.py's style."""
    with patch(
        "llm_task_router.providers.codex_cli.subprocess.run",
        side_effect=_with_auth_ok(lambda cmd, **kwargs: _completed()),
    ) as mock_run:
        invoke("do the task", model="gpt-5")

    args, kwargs = mock_run.call_args_list[1]
    cmd = args[0]
    assert cmd[:4] == ["codex", "exec", "do the task", "--model"]
    assert cmd[4] == "gpt-5"
    assert "--sandbox" in cmd and cmd[cmd.index("--sandbox") + 1] == "read-only"
    assert "--ask-for-approval" not in cmd
    assert "--skip-git-repo-check" in cmd
    assert "--output-last-message" in cmd
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["timeout"] == 60
    assert kwargs["stdin"] == subprocess.DEVNULL


def test_success_path_reads_text_from_output_last_message_file():
    def fake_run(cmd, **kwargs):
        path = cmd[cmd.index("--output-last-message") + 1]
        with open(path, "w") as f:
            f.write("the answer\n")
        return _completed()

    with patch("llm_task_router.providers.codex_cli.subprocess.run", side_effect=_with_auth_ok(fake_run)):
        result = invoke("do the task", model="gpt-5")

    assert result.text == "the answer"
    assert result.error == ""


def test_last_message_file_is_cleaned_up_after_success():
    captured_path = {}

    def fake_run(cmd, **kwargs):
        path = cmd[cmd.index("--output-last-message") + 1]
        captured_path["path"] = path
        with open(path, "w") as f:
            f.write("the answer")
        return _completed()

    with patch("llm_task_router.providers.codex_cli.subprocess.run", side_effect=_with_auth_ok(fake_run)):
        invoke("do the task", model="gpt-5")

    assert not Path(captured_path["path"]).exists()


def test_nonzero_exit_code_returns_error_from_stderr():
    with patch(
        "llm_task_router.providers.codex_cli.subprocess.run",
        side_effect=_with_auth_ok(lambda cmd, **kwargs: _completed(returncode=1, stderr="rate limited")),
    ):
        result = invoke("do the task", model="gpt-5")

    assert result.error == "rate limited"
    assert result.text == ""


def test_timeout_expired_returns_timeout_error():
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["codex"], timeout=60)

    with patch("llm_task_router.providers.codex_cli.subprocess.run", side_effect=_with_auth_ok(fake_run)):
        result = invoke("do the task", model="gpt-5")

    assert result.error == "timeout"
    assert result.duration_ms == 60_000


def test_missing_output_file_returns_empty_text():
    with patch(
        "llm_task_router.providers.codex_cli.subprocess.run",
        side_effect=_with_auth_ok(lambda cmd, **kwargs: _completed()),
    ):
        result = invoke("do the task", model="gpt-5")

    assert result.text == ""
    assert result.error == ""


def test_invoke_accepts_and_ignores_session_id():
    """codex exec has no flag to pre-assign a session id (confirmed via
    `codex exec --help`) - session_id is accepted only for Provider-protocol
    conformance and must never leak into the built command."""
    with patch(
        "llm_task_router.providers.codex_cli.subprocess.run",
        side_effect=_with_auth_ok(lambda cmd, **kwargs: _completed()),
    ) as mock_run:
        result = invoke("do the task", model="gpt-5", session_id="some-uuid")

    assert result.error == ""
    cmd = mock_run.call_args_list[1][0][0]
    assert "some-uuid" not in cmd


def test_invoke_accepts_and_ignores_permission_mode():
    """codex exec has no plan-mode analog to claude_cli.py's
    --permission-mode - permission_mode is accepted only for
    Provider-protocol conformance and must never leak into the built
    command."""
    with patch(
        "llm_task_router.providers.codex_cli.subprocess.run",
        side_effect=_with_auth_ok(lambda cmd, **kwargs: _completed()),
    ) as mock_run:
        result = invoke("do the task", model="gpt-5", permission_mode="plan")

    assert result.error == ""
    cmd = mock_run.call_args_list[1][0][0]
    assert "plan" not in cmd
    assert "--permission-mode" not in cmd


def test_invalid_model_name_returns_error_from_real_observed_failure_shape():
    """Regression guard for a real failure seen against a live account:
    an unsupported model name gets a non-zero exit and no output file, with
    the API's 400 invalid_request_error surfaced on stderr - not a crash or
    a silently empty success."""
    stderr = (
        'ERROR: {"type":"error","status":400,"error":{"type":"invalid_request_error",'
        '"message":"The \'not-a-real-model\' model is not supported when using Codex '
        'with a ChatGPT account."}}'
    )
    with patch(
        "llm_task_router.providers.codex_cli.subprocess.run",
        side_effect=_with_auth_ok(lambda cmd, **kwargs: _completed(returncode=1, stderr=stderr)),
    ):
        result = invoke("do the task", model="not-a-real-model")

    assert result.text == ""
    assert "not supported" in result.error
