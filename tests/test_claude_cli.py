import json
import subprocess
from unittest.mock import patch

from llm_task_router.providers.claude_cli import invoke


def _completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(args=["claude"], returncode=returncode, stdout=stdout, stderr=stderr)


def test_builds_expected_command_with_cost_guardrail_flags():
    """--disallowed-tools "*" and --strict-mcp-config strip the default Claude
    Code system prompt - a regression here silently reintroduces the ~$0.07/call
    cost this adapter exists to avoid. Assert the exact command list, not just
    that subprocess.run was called."""
    payload = json.dumps({"result": "ok", "total_cost_usd": 0.001, "duration_ms": 500})
    with patch("llm_task_router.providers.claude_cli.subprocess.run", return_value=_completed(stdout=payload)) as mock_run:
        invoke("do the task", model="haiku")

    args, kwargs = mock_run.call_args
    cmd = args[0]
    assert cmd == [
        "claude",
        "-p",
        "do the task",
        "--system-prompt",
        "",
        "--disallowed-tools",
        "*",
        "--strict-mcp-config",
        "--model",
        "haiku",
        "--output-format",
        "json",
    ]
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["timeout"] == 60


def test_success_path_parses_cost_duration_and_result():
    payload = json.dumps({"result": "the answer", "total_cost_usd": 0.0042, "duration_ms": 1234})
    with patch("llm_task_router.providers.claude_cli.subprocess.run", return_value=_completed(stdout=payload)):
        result = invoke("do the task", model="haiku")

    assert result.text == "the answer"
    assert result.cost_usd == 0.0042
    assert result.duration_ms == 1234
    assert result.error == ""


def test_nonzero_exit_code_returns_error_from_stderr():
    with patch(
        "llm_task_router.providers.claude_cli.subprocess.run",
        return_value=_completed(returncode=1, stderr="auth error: not logged in"),
    ):
        result = invoke("do the task", model="haiku")

    assert result.error == "auth error: not logged in"
    assert result.text == ""


def test_timeout_expired_returns_timeout_error():
    with patch(
        "llm_task_router.providers.claude_cli.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["claude"], timeout=60),
    ):
        result = invoke("do the task", model="haiku")

    assert result.error == "timeout"
    assert result.duration_ms == 60_000


def test_stdout_not_valid_json_returns_parse_error():
    with patch("llm_task_router.providers.claude_cli.subprocess.run", return_value=_completed(stdout="not json")):
        result = invoke("do the task", model="haiku")

    assert result.error == "could not parse CLI json output"
    assert result.text == "not json"


def test_payload_is_error_true_returns_error_from_result_field():
    payload = json.dumps({"is_error": True, "result": "rate limited"})
    with patch("llm_task_router.providers.claude_cli.subprocess.run", return_value=_completed(stdout=payload)):
        result = invoke("do the task", model="haiku")

    assert result.error == "rate limited"
    assert result.text == ""
