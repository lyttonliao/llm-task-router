import os
import shlex
from unittest.mock import Mock, patch

import pytest

from llm_task_router.terminal import (
    provider_cli_name,
    spawn_provider_session,
)

# --- provider_cli_name ---


def test_provider_cli_name_known_providers():
    assert provider_cli_name("claude") == "claude"
    assert provider_cli_name("codex") == "codex"


def test_provider_cli_name_unknown_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        provider_cli_name("gemini")


# --- helpers ---


def _capture_script_side_effect(captured: dict, script_arg_index: int):
    """Returns a subprocess.run side_effect that reads back the wrapper
    script's contents into captured["script"] before the (non-blocking)
    call returns - since this module no longer waits for anything, tests
    read the script directly rather than simulating a spawned process."""

    def _side_effect(cmd, *args, **kwargs):
        with open(cmd[script_arg_index]) as f:
            captured["script"] = f.read()
        return Mock(returncode=0)

    return _side_effect


# --- spawn_provider_session: command construction + macOS dispatch ---


@patch("llm_task_router.terminal.platform.system", return_value="Darwin")
@patch("llm_task_router.terminal.subprocess.run")
def test_spawn_macos_uses_open_with_command_script(mock_run, mock_system):
    result = spawn_provider_session("claude", "sonnet", "sid-123", "hello world")

    assert result is None  # non-blocking: nothing to report back
    launch_cmd = mock_run.call_args.args[0]
    assert launch_cmd[0] == "open"
    assert launch_cmd[1].endswith(".command")


@patch("llm_task_router.terminal.platform.system", return_value="Darwin")
@patch("llm_task_router.terminal.subprocess.run")
def test_spawn_macos_script_contains_session_id_flag(mock_run, mock_system):
    captured = {}
    mock_run.side_effect = _capture_script_side_effect(captured, script_arg_index=1)

    spawn_provider_session("claude", "sonnet", "sid-123", "hello world", resume=False)

    assert "claude --model sonnet --session-id sid-123" in captured["script"]
    assert "hello world" in captured["script"]


@patch("llm_task_router.terminal.platform.system", return_value="Darwin")
@patch("llm_task_router.terminal.subprocess.run")
def test_spawn_macos_script_cds_into_callers_cwd(mock_run, mock_system):
    """Regression test: the first real macOS run (2026-07-31) launched
    claude in the user's home directory instead of the repo llm-chat was
    started from, since a freshly opened Terminal window doesn't inherit
    this process's cwd on its own - the wrapper script must cd there
    explicitly."""
    captured = {}
    mock_run.side_effect = _capture_script_side_effect(captured, script_arg_index=1)

    spawn_provider_session("claude", "sonnet", "sid-123", "hi")

    lines = captured["script"].splitlines()
    assert lines[1] == f"cd -- {shlex.quote(os.getcwd())}"
    # the cd must run before the actual provider command
    assert lines.index(lines[1]) < next(i for i, line in enumerate(lines) if line.startswith("claude "))


@patch("llm_task_router.terminal.platform.system", return_value="Darwin")
@patch("llm_task_router.terminal.subprocess.run")
def test_spawn_macos_resume_uses_resume_flag(mock_run, mock_system):
    captured = {}
    mock_run.side_effect = _capture_script_side_effect(captured, script_arg_index=1)

    spawn_provider_session("claude", "sonnet", "sid-123", "hi", resume=True)

    assert "--resume sid-123" in captured["script"]
    assert "--session-id" not in captured["script"]


@patch("llm_task_router.terminal.platform.system", return_value="Darwin")
@patch("llm_task_router.terminal.subprocess.run")
def test_spawn_macos_script_deletes_itself_as_last_line(mock_run, mock_system):
    """Non-blocking means this module never learns when the spawned
    process finishes, so it can't safely delete the wrapper script itself
    after launching - the script must clean up after its own run instead."""
    captured = {}
    mock_run.side_effect = _capture_script_side_effect(captured, script_arg_index=1)

    spawn_provider_session("claude", "sonnet", "sid-123", "hi")

    script_path = mock_run.call_args.args[0][1]
    lines = captured["script"].splitlines()
    assert lines[-1] == f"rm -f -- {shlex.quote(script_path)}"


@patch("llm_task_router.terminal.platform.system", return_value="Darwin")
@patch("llm_task_router.terminal.subprocess.run")
def test_spawn_macos_does_not_wait_for_subprocess_run_to_report_completion(mock_run, mock_system):
    """The whole point of the non-blocking revision: spawn_provider_session
    must return as soon as the launcher call (`open ...`) itself returns,
    without any further polling/sleeping - confirmed by never patching in
    a sentinel file or any completion signal at all and still returning
    cleanly."""
    mock_run.return_value = Mock(returncode=0)

    spawn_provider_session("claude", "sonnet", "sid-123", "hi")  # must not hang

    mock_run.assert_called_once()


# --- Linux dispatch ---


@patch("llm_task_router.terminal.platform.system", return_value="Linux")
@patch("llm_task_router.terminal.shutil.which")
@patch("llm_task_router.terminal.subprocess.run")
def test_spawn_linux_uses_first_available_terminal(mock_run, mock_which, mock_system):
    mock_which.side_effect = lambda name: "/usr/bin/gnome-terminal" if name == "gnome-terminal" else None

    spawn_provider_session("claude", "sonnet", "sid-123", "hi")

    launch_cmd = mock_run.call_args.args[0]
    assert launch_cmd[0] == "gnome-terminal"
    assert launch_cmd[1] == "--"


@patch("llm_task_router.terminal.platform.system", return_value="Linux")
@patch("llm_task_router.terminal.shutil.which", return_value=None)
def test_spawn_linux_raises_when_no_terminal_found(mock_which, mock_system):
    with pytest.raises(RuntimeError, match="no supported terminal emulator"):
        spawn_provider_session("claude", "sonnet", "sid-123", "hi")


# --- Windows dispatch ---


@patch("llm_task_router.terminal.platform.system", return_value="Windows")
@patch("llm_task_router.terminal.subprocess.run")
def test_spawn_windows_uses_cmd_start(mock_run, mock_system):
    spawn_provider_session("claude", "sonnet", "sid-123", "hi")

    launch_cmd = mock_run.call_args.args[0]
    assert launch_cmd[:3] == ["cmd", "/c", "start"]


@patch("llm_task_router.terminal.platform.system", return_value="Windows")
@patch("llm_task_router.terminal.subprocess.run")
def test_spawn_windows_script_cds_into_callers_cwd(mock_run, mock_system):
    captured = {}

    def _side_effect(cmd, *args, **kwargs):
        script_path = next(p for p in cmd if p.endswith(".bat"))
        with open(script_path) as f:
            captured["script"] = f.read()
        return Mock(returncode=0)

    mock_run.side_effect = _side_effect

    spawn_provider_session("claude", "sonnet", "sid-123", "hi")

    assert f'cd /d "{os.getcwd()}"' in captured["script"]


# --- Unsupported platform ---


@patch("llm_task_router.terminal.platform.system", return_value="Plan9")
def test_spawn_unsupported_platform_raises(mock_system):
    with pytest.raises(ValueError, match="unsupported platform"):
        spawn_provider_session("claude", "sonnet", "sid-123", "hi")
