import subprocess
from pathlib import Path
from unittest.mock import patch

from llm_task_router.providers.codex_cli import invoke


def _completed(returncode=0, stderr=""):
    return subprocess.CompletedProcess(args=["codex"], returncode=returncode, stdout="", stderr=stderr)


def test_builds_expected_command_with_readonly_sandbox_flags():
    """--sandbox read-only/--ask-for-approval never is the closest analog to
    claude_cli.py's --disallowed-tools "*" - a regression here would let the
    router shell out with write access or block waiting on human approval.
    Assert the exact command list, matching test_claude_cli.py's style."""
    with patch("llm_task_router.providers.codex_cli.subprocess.run", return_value=_completed()) as mock_run:
        invoke("do the task", model="gpt-5")

    args, kwargs = mock_run.call_args
    cmd = args[0]
    assert cmd[:4] == ["codex", "exec", "do the task", "--model"]
    assert cmd[4] == "gpt-5"
    assert "--sandbox" in cmd and cmd[cmd.index("--sandbox") + 1] == "read-only"
    assert "--ask-for-approval" in cmd and cmd[cmd.index("--ask-for-approval") + 1] == "never"
    assert "--skip-git-repo-check" in cmd
    assert "--output-last-message" in cmd
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["timeout"] == 60


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
