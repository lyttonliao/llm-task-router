import shlex
from unittest.mock import Mock, patch

import pytest

from llm_task_router.terminal import (
    attach_terminal,
    create_session,
    provider_cli_name,
    send_message,
    switch_model,
    tmux_available,
)

# --- provider_cli_name ---


def test_provider_cli_name_known_providers():
    assert provider_cli_name("claude") == "claude"
    assert provider_cli_name("codex") == "codex"


def test_provider_cli_name_unknown_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        provider_cli_name("gemini")


# --- tmux_available ---


@patch("llm_task_router.terminal.shutil.which", return_value="/opt/homebrew/bin/tmux")
def test_tmux_available_true_when_on_path(mock_which):
    assert tmux_available() is True
    mock_which.assert_called_once_with("tmux")


@patch("llm_task_router.terminal.shutil.which", return_value=None)
def test_tmux_available_false_when_missing(mock_which):
    assert tmux_available() is False


# --- create_session ---


@patch("llm_task_router.terminal.subprocess.run")
def test_create_session_builds_detached_new_session_command(mock_run):
    create_session("claude", "sonnet", "sid-123", "/repo/dir")

    cmd = mock_run.call_args.args[0]
    assert cmd == [
        "tmux", "new-session", "-d", "-s", "sid-123", "-c", "/repo/dir",
        "--", "claude", "--model", "sonnet", "--session-id", "sid-123",
    ]
    assert mock_run.call_args.kwargs.get("check") is True


@patch("llm_task_router.terminal.subprocess.run")
def test_create_session_uses_provider_cli_name(mock_run):
    create_session("codex", "gpt-5.5", "sid-456", "/repo/dir")

    cmd = mock_run.call_args.args[0]
    assert cmd[cmd.index("--") + 1] == "codex"


@patch("llm_task_router.terminal.subprocess.run")
def test_create_session_unknown_provider_raises_before_running_anything(mock_run):
    with pytest.raises(ValueError, match="unknown provider"):
        create_session("gemini", "some-model", "sid-123", "/repo/dir")
    mock_run.assert_not_called()


# --- send_message ---


@patch("llm_task_router.terminal.subprocess.run")
def test_send_message_loads_pastes_then_sends_enter_as_three_calls(mock_run):
    send_message("sid-123", "hello world")

    assert mock_run.call_count == 3
    load_call, paste_call, enter_call = mock_run.call_args_list
    assert load_call.args[0] == ["tmux", "load-buffer", "-"]
    assert load_call.kwargs.get("input") == b"hello world"
    assert paste_call.args[0] == ["tmux", "paste-buffer", "-p", "-d", "-t", "sid-123"]
    assert enter_call.args[0] == ["tmux", "send-keys", "-t", "sid-123", "Enter"]


@patch("llm_task_router.terminal.subprocess.run")
def test_send_message_load_call_happens_before_paste_and_enter_calls(mock_run):
    send_message("sid-123", "hi")

    assert mock_run.call_args_list[0].args[0][:2] == ["tmux", "load-buffer"]
    assert mock_run.call_args_list[1].args[0][:2] == ["tmux", "paste-buffer"]
    assert mock_run.call_args_list[2].args[0][-1] == "Enter"


@patch("llm_task_router.terminal.subprocess.run")
def test_send_message_does_not_reinterpret_special_characters(mock_run):
    """load-buffer reads text from stdin as opaque bytes, so a message that
    looks like tmux key syntax (semicolons, a leading '-', a 'C-c'-looking
    substring) is never at risk of being parsed as flags or key names - the
    same guarantee `-l --` used to provide for send-keys."""
    send_message("sid-123", "-rf; C-c looks scary")

    load_call = mock_run.call_args_list[0]
    assert load_call.args[0] == ["tmux", "load-buffer", "-"]
    assert load_call.kwargs.get("input") == b"-rf; C-c looks scary"


@patch("llm_task_router.terminal.subprocess.run")
def test_send_message_preserves_embedded_newlines_as_a_single_paste(mock_run):
    """The whole point of routing through paste-buffer -p instead of
    send-keys -l: a multi-line message (pasted content, or typed via
    llm-chat's Alt+Enter) must land as one paste, not per-line Enter
    presses in the remote provider CLI - see terminal.py's send_message()
    docstring for the live gap this closes."""
    send_message("sid-123", "line one\nline two\nline three")

    load_call = mock_run.call_args_list[0]
    assert load_call.kwargs.get("input") == b"line one\nline two\nline three"


# --- switch_model ---


@patch("llm_task_router.terminal.time.sleep")
@patch("llm_task_router.terminal.subprocess.run")
def test_switch_model_sends_model_slash_command_via_send_message(mock_run, mock_sleep):
    switch_model("sid-123", "opus")

    load_call = mock_run.call_args_list[0]
    assert load_call.args[0] == ["tmux", "load-buffer", "-"]
    assert load_call.kwargs.get("input") == b"/model opus"
    paste_call = mock_run.call_args_list[1]
    assert paste_call.args[0] == ["tmux", "paste-buffer", "-p", "-d", "-t", "sid-123"]
    enter_call = mock_run.call_args_list[2]
    assert enter_call.args[0] == ["tmux", "send-keys", "-t", "sid-123", "Enter"]


@patch("llm_task_router.terminal.time.sleep")
@patch("llm_task_router.terminal.subprocess.run")
def test_switch_model_sends_a_second_enter_to_confirm_the_switch_dialog(mock_run, mock_sleep):
    """Real live bug (2026-07-31): /model <name> opens a confirmation
    dialog rather than switching immediately - without this second Enter,
    the caller's next send_message() call types the real message into that
    dialog instead of a text box and it's silently lost."""
    switch_model("sid-123", "opus")

    assert mock_run.call_count == 4
    confirm_call = mock_run.call_args_list[3]
    assert confirm_call.args[0] == ["tmux", "send-keys", "-t", "sid-123", "Enter"]
    mock_sleep.assert_called_once()


@patch("llm_task_router.terminal.time.sleep")
@patch("llm_task_router.terminal.subprocess.run")
def test_switch_model_waits_before_sending_the_confirmation_enter(mock_run, mock_sleep):
    events = []
    mock_run.side_effect = lambda *a, **k: events.append("run")
    mock_sleep.side_effect = lambda *a, **k: events.append("sleep")

    switch_model("sid-123", "opus")

    assert events == ["run", "run", "run", "sleep", "run"]


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


# --- attach_terminal: command construction + macOS dispatch ---


@patch("llm_task_router.terminal.platform.system", return_value="Darwin")
@patch("llm_task_router.terminal.subprocess.run")
def test_attach_macos_uses_open_with_command_script(mock_run, mock_system):
    result = attach_terminal("sid-123", "/repo/dir")

    assert result is None  # non-blocking: nothing to report back
    launch_cmd = mock_run.call_args.args[0]
    assert launch_cmd[0] == "open"
    assert launch_cmd[1].endswith(".command")


@patch("llm_task_router.terminal.platform.system", return_value="Darwin")
@patch("llm_task_router.terminal.subprocess.run")
def test_attach_macos_script_runs_tmux_attach(mock_run, mock_system):
    captured = {}
    mock_run.side_effect = _capture_script_side_effect(captured, script_arg_index=1)

    attach_terminal("sid-123", "/repo/dir")

    assert "tmux attach -t sid-123" in captured["script"]


@patch("llm_task_router.terminal.platform.system", return_value="Darwin")
@patch("llm_task_router.terminal.subprocess.run")
def test_attach_macos_script_cds_into_callers_cwd(mock_run, mock_system):
    captured = {}
    mock_run.side_effect = _capture_script_side_effect(captured, script_arg_index=1)

    attach_terminal("sid-123", "/repo/dir")

    lines = captured["script"].splitlines()
    assert lines[1] == f"cd -- {shlex.quote('/repo/dir')}"
    assert lines.index(lines[1]) < next(i for i, line in enumerate(lines) if line.startswith("tmux "))


@patch("llm_task_router.terminal.platform.system", return_value="Darwin")
@patch("llm_task_router.terminal.subprocess.run")
def test_attach_macos_script_deletes_itself_as_last_line(mock_run, mock_system):
    captured = {}
    mock_run.side_effect = _capture_script_side_effect(captured, script_arg_index=1)

    attach_terminal("sid-123", "/repo/dir")

    script_path = mock_run.call_args.args[0][1]
    lines = captured["script"].splitlines()
    assert lines[-1] == f"rm -f -- {shlex.quote(script_path)}"


@patch("llm_task_router.terminal.platform.system", return_value="Darwin")
@patch("llm_task_router.terminal.subprocess.run")
def test_attach_macos_does_not_wait_for_subprocess_run_to_report_completion(mock_run, mock_system):
    mock_run.return_value = Mock(returncode=0)

    attach_terminal("sid-123", "/repo/dir")  # must not hang

    mock_run.assert_called_once()


# --- Linux dispatch ---


@patch("llm_task_router.terminal.platform.system", return_value="Linux")
@patch("llm_task_router.terminal.shutil.which")
@patch("llm_task_router.terminal.subprocess.run")
def test_attach_linux_uses_first_available_terminal(mock_run, mock_which, mock_system):
    mock_which.side_effect = lambda name: "/usr/bin/gnome-terminal" if name == "gnome-terminal" else None

    attach_terminal("sid-123", "/repo/dir")

    launch_cmd = mock_run.call_args.args[0]
    assert launch_cmd[0] == "gnome-terminal"
    assert launch_cmd[1] == "--"


@patch("llm_task_router.terminal.platform.system", return_value="Linux")
@patch("llm_task_router.terminal.shutil.which", return_value=None)
def test_attach_linux_raises_when_no_terminal_found(mock_which, mock_system):
    with pytest.raises(RuntimeError, match="no supported terminal emulator"):
        attach_terminal("sid-123", "/repo/dir")


# --- Windows dispatch ---


@patch("llm_task_router.terminal.platform.system", return_value="Windows")
@patch("llm_task_router.terminal.subprocess.run")
def test_attach_windows_uses_cmd_start(mock_run, mock_system):
    attach_terminal("sid-123", "/repo/dir")

    launch_cmd = mock_run.call_args.args[0]
    assert launch_cmd[:3] == ["cmd", "/c", "start"]


@patch("llm_task_router.terminal.platform.system", return_value="Windows")
@patch("llm_task_router.terminal.subprocess.run")
def test_attach_windows_script_cds_into_callers_cwd(mock_run, mock_system):
    captured = {}

    def _side_effect(cmd, *args, **kwargs):
        script_path = next(p for p in cmd if p.endswith(".bat"))
        with open(script_path) as f:
            captured["script"] = f.read()
        return Mock(returncode=0)

    mock_run.side_effect = _side_effect

    attach_terminal("sid-123", "/repo/dir")

    assert 'cd /d "/repo/dir"' in captured["script"]


# --- Unsupported platform ---


@patch("llm_task_router.terminal.platform.system", return_value="Plan9")
def test_attach_unsupported_platform_raises(mock_system):
    with pytest.raises(ValueError, match="unsupported platform"):
        attach_terminal("sid-123", "/repo/dir")
