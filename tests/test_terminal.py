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


def _sentinel_path_for(script_path: str) -> str:
    return os.path.splitext(script_path)[0] + ".sentinel"


def _write_sentinel_side_effect(exit_code: int = 0):
    """Returns a subprocess.run side_effect that, given the launcher cmd,
    locates the script path passed to it and writes exit_code to that
    script's sentinel file - simulating the spawned terminal running the
    wrapper script and finishing instantly, so the real polling loop in
    _poll_sentinel exits immediately instead of the test sleeping."""

    def _side_effect(cmd, *args, **kwargs):
        script_path = next(p for p in cmd if p.endswith((".command", ".sh", ".bat")))
        with open(_sentinel_path_for(script_path), "w") as f:
            f.write(str(exit_code))
        return Mock(returncode=0)

    return _side_effect


# --- spawn_provider_session: command construction + macOS dispatch ---


@patch("llm_task_router.terminal.platform.system", return_value="Darwin")
@patch("llm_task_router.terminal.subprocess.run")
def test_spawn_macos_uses_open_with_command_script(mock_run, mock_system):
    mock_run.side_effect = _write_sentinel_side_effect(exit_code=0)

    code = spawn_provider_session("claude", "sonnet", "sid-123", "hello world")

    assert code == 0
    launch_cmd = mock_run.call_args.args[0]
    assert launch_cmd[0] == "open"
    assert launch_cmd[1].endswith(".command")


@patch("llm_task_router.terminal.platform.system", return_value="Darwin")
@patch("llm_task_router.terminal.subprocess.run")
def test_spawn_macos_script_contains_session_id_flag(mock_run, mock_system):
    captured = {}

    def _side_effect(cmd, *args, **kwargs):
        script_path = cmd[1]
        with open(script_path) as f:
            captured["script"] = f.read()
        with open(_sentinel_path_for(script_path), "w") as f:
            f.write("0")
        return Mock(returncode=0)

    mock_run.side_effect = _side_effect

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

    def _side_effect(cmd, *args, **kwargs):
        script_path = cmd[1]
        with open(script_path) as f:
            captured["script"] = f.read()
        with open(_sentinel_path_for(script_path), "w") as f:
            f.write("0")
        return Mock(returncode=0)

    mock_run.side_effect = _side_effect

    spawn_provider_session("claude", "sonnet", "sid-123", "hi")

    lines = captured["script"].splitlines()
    assert lines[1] == f"cd -- {shlex.quote(os.getcwd())}"
    # the cd must run before the actual provider command
    assert lines.index(lines[1]) < next(i for i, line in enumerate(lines) if line.startswith("claude "))


@patch("llm_task_router.terminal.platform.system", return_value="Darwin")
@patch("llm_task_router.terminal.subprocess.run")
def test_spawn_macos_resume_uses_resume_flag(mock_run, mock_system):
    captured = {}

    def _side_effect(cmd, *args, **kwargs):
        script_path = cmd[1]
        with open(script_path) as f:
            captured["script"] = f.read()
        with open(_sentinel_path_for(script_path), "w") as f:
            f.write("0")
        return Mock(returncode=0)

    mock_run.side_effect = _side_effect

    spawn_provider_session("claude", "sonnet", "sid-123", "hi", resume=True)

    assert "--resume sid-123" in captured["script"]
    assert "--session-id" not in captured["script"]


@patch("llm_task_router.terminal.platform.system", return_value="Darwin")
@patch("llm_task_router.terminal.subprocess.run")
def test_spawn_macos_returns_real_exit_code(mock_run, mock_system):
    mock_run.side_effect = _write_sentinel_side_effect(exit_code=7)

    code = spawn_provider_session("claude", "sonnet", "sid-123", "hi")

    assert code == 7


@patch("llm_task_router.terminal.platform.system", return_value="Darwin")
@patch("llm_task_router.terminal.subprocess.run")
def test_spawn_macos_cleans_up_temp_files(mock_run, mock_system):
    paths = {}

    def _side_effect(cmd, *args, **kwargs):
        script_path = cmd[1]
        sentinel_path = _sentinel_path_for(script_path)
        paths["script"] = script_path
        paths["sentinel"] = sentinel_path
        with open(sentinel_path, "w") as f:
            f.write("0")
        return Mock(returncode=0)

    mock_run.side_effect = _side_effect

    spawn_provider_session("claude", "sonnet", "sid-123", "hi")

    assert not os.path.exists(paths["script"])
    assert not os.path.exists(paths["sentinel"])


# --- Linux dispatch ---


@patch("llm_task_router.terminal.platform.system", return_value="Linux")
@patch("llm_task_router.terminal.shutil.which")
@patch("llm_task_router.terminal.subprocess.run")
def test_spawn_linux_uses_first_available_terminal(mock_run, mock_which, mock_system):
    mock_which.side_effect = lambda name: "/usr/bin/gnome-terminal" if name == "gnome-terminal" else None
    mock_run.side_effect = _write_sentinel_side_effect(exit_code=0)

    code = spawn_provider_session("claude", "sonnet", "sid-123", "hi")

    assert code == 0
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
    def _side_effect(cmd, *args, **kwargs):
        script_path = next(p for p in cmd if p.endswith(".bat"))
        with open(_sentinel_path_for(script_path), "w") as f:
            f.write("0")
        return Mock(returncode=0)

    mock_run.side_effect = _side_effect

    code = spawn_provider_session("claude", "sonnet", "sid-123", "hi")

    assert code == 0
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
        with open(_sentinel_path_for(script_path), "w") as f:
            f.write("0")
        return Mock(returncode=0)

    mock_run.side_effect = _side_effect

    spawn_provider_session("claude", "sonnet", "sid-123", "hi")

    assert f'cd /d "{os.getcwd()}"' in captured["script"]


# --- Unsupported platform ---


@patch("llm_task_router.terminal.platform.system", return_value="Plan9")
def test_spawn_unsupported_platform_raises(mock_system):
    with pytest.raises(ValueError, match="unsupported platform"):
        spawn_provider_session("claude", "sonnet", "sid-123", "hi")
