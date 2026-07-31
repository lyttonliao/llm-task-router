"""Platform-dispatch terminal spawn for llm-chat's routing pivot: classify
in-process, then hand the actual provider call to a real native terminal so
the user gets full native Claude Code/Codex UX instead of this repo
re-rendering it - see CLAUDE.md's "Next step" section and the full design at
~/.claude/plans/what-s-our-next-goal-jazzy-tome.md (this machine/user's
plans directory, not in-repo; describes an earlier blocking version of this
module, since revised - see "Non-blocking" below).

This module is deliberately narrow: it only launches a terminal running the
provider CLI and returns. It does NOT decide provider/model (router.py
already does that) and does NOT track which session ids have had their
establishing --session-id call yet (unlike providers/claude_cli.py's
_established_sessions - callers own that decision here and pass resume=
accordingly).

Non-blocking (revised 2026-07-31, after an earlier blocking version):
spawn_provider_session() launches the terminal and returns as soon as the
launch itself succeeds, without waiting for the spawned claude/codex
session to exit. The earlier version blocked on a disposable wrapper-script
+ sentinel-file poll loop (since `open -a Terminal`/equivalents don't block
on their own) so repl.py's chat_loop() could return to its own prompt only
once the spawned session ended - live use surfaced that as real friction
twice (the terminal appearing to open in the wrong directory looked like a
hang, and being unable to type into llm-chat until fully exiting the
spawned session, including once requiring Ctrl-C to escape a stuck wait,
felt broken even after being confirmed as the intended design). The wrapper
script this module still writes (needed regardless of blocking, to `cd`
into the caller's cwd and construct the actual command - see below) now
deletes itself as its own last action instead of this module waiting to
know when that's safe to do from the polling loop's completion.

`cd` into the caller's cwd, added 2026-07-31: `open`/terminal-emulator
launches start the new window's shell at its own default startup directory
(the user's home directory on macOS), NOT the calling process's cwd -
confirmed against a real spawn: the spawned session's own startup banner
reported "launched claude in your home directory" instead of the repo
llm-chat was run from. Each wrapper script below `cd`s explicitly before
running the actual provider command.

Verified: macOS only (this machine's daily dev environment), by direct
manual testing. NOT verified at all: Linux (gnome-terminal/konsole/xterm
dispatch is written to the same pattern documented above but never run
against a real Linux desktop) and Windows (same - untested against a real
`cmd`/`start` invocation). Both are best-effort, matching this repo's
existing convention for flagging unverified platform behavior (see
docs/rough-edges.md's Windows select() note for the streaming transport)."""

import os
import platform
import shlex
import shutil
import subprocess
import tempfile
import uuid

_PROVIDER_CLI = {"claude": "claude", "codex": "codex"}

_LINUX_TERMINALS = ("gnome-terminal", "konsole", "xterm")


def provider_cli_name(provider: str) -> str:
    try:
        return _PROVIDER_CLI[provider]
    except KeyError:
        raise ValueError(f"unknown provider: {provider!r}") from None


def spawn_provider_session(
    provider: str, model: str, session_id: str, message: str, *, resume: bool = False
) -> None:
    """Launches a new native terminal running:
        <provider cli> --model <model> (--session-id|--resume) <session_id> <message>
    and returns as soon as that launch succeeds - does NOT wait for the
    spawned session to exit (see module docstring, "Non-blocking").

    resume mirrors claude_cli.py's --session-id-first-call / --resume-
    after-that distinction - pass resume=True once the caller knows this
    session_id has already had its establishing call (that bookkeeping
    lives with the caller, not this module - see module docstring). Because
    this is non-blocking, a caller that fires a --resume call before an
    earlier --session-id call has actually finished being established by
    the spawned CLI is a real, currently-untested race - accepted
    deliberately rather than reintroducing blocking to avoid it."""
    cli = provider_cli_name(provider)
    session_flag = "--resume" if resume else "--session-id"
    cmd = [cli, "--model", model, session_flag, session_id, message]
    cwd = os.getcwd()

    system = platform.system()
    if system == "Darwin":
        _spawn_macos(cmd, cwd)
    elif system == "Linux":
        _spawn_linux(cmd, cwd)
    elif system == "Windows":
        _spawn_windows(cmd, cwd)
    else:
        raise ValueError(f"unsupported platform for terminal spawning: {system}")


def _spawn_macos(cmd: list[str], cwd: str) -> None:
    """Writes a .command file (macOS Terminal's double-click/open
    association for shell scripts) and opens it via `open`. The script
    deletes itself as its last line, once the provider command has
    finished - not this function's job, since it doesn't wait around to
    find out when that is."""
    run_id = uuid.uuid4().hex
    script_path = os.path.join(tempfile.gettempdir(), f"llm-chat-spawn-{run_id}.command")

    script = (
        f"#!/bin/bash\ncd -- {shlex.quote(cwd)}\n{shlex.join(cmd)}\n"
        f"rm -f -- {shlex.quote(script_path)}\n"
    )
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(script_path, 0o700)

    subprocess.run(["open", script_path], check=True)


def _spawn_linux(cmd: list[str], cwd: str) -> None:
    """Best-effort, unverified against a real Linux desktop (see module
    docstring). Picks the first available terminal emulator on PATH from
    _LINUX_TERMINALS; same self-deleting wrapper-script pattern
    _spawn_macos uses."""
    terminal = next((t for t in _LINUX_TERMINALS if shutil.which(t)), None)
    if terminal is None:
        raise RuntimeError(
            "no supported terminal emulator found on PATH "
            f"(tried: {', '.join(_LINUX_TERMINALS)})"
        )

    run_id = uuid.uuid4().hex
    script_path = os.path.join(tempfile.gettempdir(), f"llm-chat-spawn-{run_id}.sh")

    script = (
        f"#!/bin/bash\ncd -- {shlex.quote(cwd)}\n{shlex.join(cmd)}\n"
        f"rm -f -- {shlex.quote(script_path)}\n"
    )
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(script_path, 0o700)

    subprocess.run([terminal, "--", "bash", script_path])


def _spawn_windows(cmd: list[str], cwd: str) -> None:
    """Best-effort, unverified against real Windows (see module docstring).
    `.bat` equivalent of the self-deleting wrapper-script pattern the other
    platforms use; `cmd /c start "" cmd /c <script>` opens a new console
    window. `cd /d` (not plain `cd`) so this also switches drive letters
    when cwd is on a different one than the new console's own default."""
    run_id = uuid.uuid4().hex
    script_path = os.path.join(tempfile.gettempdir(), f"llm-chat-spawn-{run_id}.bat")

    quoted_cmd = " ".join(f'"{part}"' if " " in part else part for part in cmd)
    script = (
        f'@echo off\r\ncd /d "{cwd}"\r\n{quoted_cmd}\r\ndel "{script_path}"\r\n'
    )
    with open(script_path, "w") as f:
        f.write(script)

    subprocess.run(["cmd", "/c", "start", "", "cmd", "/c", script_path])
