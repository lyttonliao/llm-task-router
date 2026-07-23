import subprocess
from pathlib import Path
from unittest.mock import patch

from llm_task_router.providers.codex_cli import invoke


def _completed(returncode=0, stderr=""):
    return subprocess.CompletedProcess(args=["codex"], returncode=returncode, stdout="", stderr=stderr)


def test_builds_expected_command_with_readonly_sandbox_flag():
    """--sandbox read-only is the closest analog to claude_cli.py's
    --disallowed-tools "*" - a regression here would let the router shell out
    with write access. Note --ask-for-approval is deliberately absent: it's a
    top-level `codex` option, not a valid `codex exec` flag (confirmed by a
    real run failing with "unexpected argument" when it was passed) - exec
    auto-defaults to never-ask on its own. Assert the exact command list,
    matching test_claude_cli.py's style."""
    with patch("llm_task_router.providers.codex_cli.subprocess.run", return_value=_completed()) as mock_run:
        invoke("do the task", model="gpt-5")

    args, kwargs = mock_run.call_args
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

    with patch("llm_task_router.providers.codex_cli.subprocess.run", side_effect=fake_run):
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

    with patch("llm_task_router.providers.codex_cli.subprocess.run", side_effect=fake_run):
        invoke("do the task", model="gpt-5")

    assert not Path(captured_path["path"]).exists()


def test_nonzero_exit_code_returns_error_from_stderr():
    with patch(
        "llm_task_router.providers.codex_cli.subprocess.run",
        return_value=_completed(returncode=1, stderr="auth error: not logged in"),
    ):
        result = invoke("do the task", model="gpt-5")

    assert result.error == "auth error: not logged in"
    assert result.text == ""


def test_timeout_expired_returns_timeout_error():
    with patch(
        "llm_task_router.providers.codex_cli.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["codex"], timeout=60),
    ):
        result = invoke("do the task", model="gpt-5")

    assert result.error == "timeout"
    assert result.duration_ms == 60_000


def test_missing_output_file_returns_empty_text():
    with patch("llm_task_router.providers.codex_cli.subprocess.run", return_value=_completed()):
        result = invoke("do the task", model="gpt-5")

    assert result.text == ""
    assert result.error == ""


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
        return_value=_completed(returncode=1, stderr=stderr),
    ):
        result = invoke("do the task", model="not-a-real-model")

    assert result.text == ""
    assert "not supported" in result.error
