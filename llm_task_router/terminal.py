"""Platform-dispatch terminal spawn for the llm-chat pivot to spawn-per-
message: classify in-process, then hand the actual provider call to a real
native terminal so the user gets full native Claude Code/Codex UX instead of
this repo re-rendering it - see CLAUDE.md's "Next step" section and the full
design at ~/.claude/plans/what-s-our-next-goal-jazzy-tome.md (this machine/
user's plans directory, not in-repo).

This module is deliberately narrow: it only spawns a terminal and blocks
until the provider CLI inside it exits, returning that exit code. It does
NOT decide provider/model (router.py already does that), does NOT track
which session ids have had their establishing --session-id call yet (unlike
providers/claude_cli.py's _established_sessions - callers own that decision
here and pass resume= accordingly), and is NOT wired into repl.py's
chat_loop() yet - that's a separate, follow-on change once this primitive is
verified on its own.

Blocking mechanism: `open -a Terminal` (macOS) and OS-level "launch a
terminal emulator" calls generally do NOT block until the spawned window
closes, and Terminal.app windows don't auto-close after their command exits
regardless (that depends on user profile settings this repo doesn't
control). So instead of relying on the launcher call itself to block, each
platform writes a disposable wrapper script that runs the actual provider
CLI command and then writes its exit code to a sentinel file; this module
polls for that sentinel file's existence and reads the code from it. This
works identically across platforms - only "how do you open a new terminal
window" differs.

No poll timeout, matching providers/claude_cli.py's _resolve_timeout_s()
reasoning: a spawned session has a human physically present, Ctrl-C already
interrupts a blocking loop (time.sleep raises KeyboardInterrupt same as
select.select() does there), and a wall-clock guess would be as likely to
kill a legitimately long-running session as an actually-stuck one.

Verified: macOS only (this machine's daily dev environment). A real
end-to-end run (2026-07-31, once wired into repl.py:chat_loop()) confirmed
the terminal genuinely opens and the message reaches a real interactive
claude session - but also caught a real bug the mocked tests couldn't:
the spawned shell's default startup directory is the user's home
directory, not the caller's cwd, so the first real run launched claude in
the wrong place entirely. Fixed by an explicit `cd` in the wrapper script
before the actual command (see _spawn_macos()) - the sentinel-file polling
logic itself was already covered by mocked tests (tests/test_terminal.py),
but "what directory does the spawned process start in" was not something
those tests exercised, since they never actually launch a shell.

NOT verified at all: Linux (gnome-terminal/konsole/xterm dispatch is
written to the same pattern documented above but never run against a real
Linux desktop) and Windows (same - untested against a real `cmd`/`start`
invocation). Both are best-effort, matching this repo's existing convention
for flagging unverified platform behavior (see docs/rough-edges.md's
Windows select() note for the streaming transport)."""

import os
import platform
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid

_PROVIDER_CLI = {"claude": "claude", "codex": "codex"}

_POLL_INTERVAL_S = 0.5

_LINUX_TERMINALS = ("gnome-terminal", "konsole", "xterm")


def provider_cli_name(provider: str) -> str:
    try:
        return _PROVIDER_CLI[provider]
    except KeyError:
        raise ValueError(f"unknown provider: {provider!r}") from None


def spawn_provider_session(
    provider: str, model: str, session_id: str, message: str, *, resume: bool = False
) -> int:
    """Spawns a new native terminal running:
        <provider cli> --model <model> (--session-id|--resume) <session_id> <message>
    blocks until that process exits, and returns its exit code.

    resume mirrors claude_cli.py's --session-id-first-call / --resume-
    after-that distinction - pass resume=True once the caller knows this
    session_id has already had its establishing call (that bookkeeping
    lives with the caller, not this module - see module docstring)."""
    cli = provider_cli_name(provider)
    session_flag = "--resume" if resume else "--session-id"
    cmd = [cli, "--model", model, session_flag, session_id, message]
    cwd = os.getcwd()

    system = platform.system()
    if system == "Darwin":
        return _spawn_macos(cmd, cwd)
    if system == "Linux":
        return _spawn_linux(cmd, cwd)
    if system == "Windows":
        return _spawn_windows(cmd, cwd)
    raise ValueError(f"unsupported platform for terminal spawning: {system}")


def _poll_sentinel(sentinel_path: str) -> int:
    while not os.path.exists(sentinel_path):
        time.sleep(_POLL_INTERVAL_S)
    with open(sentinel_path) as f:
        return int(f.read().strip())


def _spawn_macos(cmd: list[str], cwd: str) -> int:
    """Writes a .command file (macOS Terminal's double-click/open
    association for shell scripts) and opens it via `open`, which launches
    a new Terminal window running the script - launcher call itself
    returns immediately, so completion is detected via the sentinel file
    the script writes on exit, not via `open`'s own return.

    `open`/Terminal.app start the new window's shell at its own default
    startup directory (the user's home directory), NOT the calling
    process's cwd - confirmed against a real spawn (2026-07-31): the
    spawned session's own startup banner reported "launched claude in your
    home directory" instead of the repo llm-chat was run from. The `cd`
    below is what actually fixes that, not something `open` does for
    free."""
    run_id = uuid.uuid4().hex
    tmp_dir = tempfile.gettempdir()
    script_path = os.path.join(tmp_dir, f"llm-chat-spawn-{run_id}.command")
    sentinel_path = os.path.join(tmp_dir, f"llm-chat-spawn-{run_id}.sentinel")

    script = (
        f"#!/bin/bash\ncd -- {shlex.quote(cwd)}\n{shlex.join(cmd)}\n"
        f"echo $? > {shlex.quote(sentinel_path)}\n"
    )
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(script_path, 0o700)

    try:
        subprocess.run(["open", script_path], check=True)
        return _poll_sentinel(sentinel_path)
    finally:
        for path in (script_path, sentinel_path):
            try:
                os.remove(path)
            except OSError:
                pass


def _spawn_linux(cmd: list[str], cwd: str) -> int:
    """Best-effort, unverified against a real Linux desktop (see module
    docstring). Picks the first available terminal emulator on PATH from
    _LINUX_TERMINALS; each is invoked to run the same disposable wrapper
    script + sentinel-file pattern _spawn_macos uses, since none of these
    emulators' "wait for exit" flags (e.g. newer gnome-terminal's --wait)
    are universal enough to depend on. Same explicit `cd` as _spawn_macos -
    a freshly opened terminal emulator window has no reason to inherit this
    process's cwd on its own."""
    terminal = next((t for t in _LINUX_TERMINALS if shutil.which(t)), None)
    if terminal is None:
        raise RuntimeError(
            "no supported terminal emulator found on PATH "
            f"(tried: {', '.join(_LINUX_TERMINALS)})"
        )

    run_id = uuid.uuid4().hex
    tmp_dir = tempfile.gettempdir()
    script_path = os.path.join(tmp_dir, f"llm-chat-spawn-{run_id}.sh")
    sentinel_path = os.path.join(tmp_dir, f"llm-chat-spawn-{run_id}.sentinel")

    script = (
        f"#!/bin/bash\ncd -- {shlex.quote(cwd)}\n{shlex.join(cmd)}\n"
        f"echo $? > {shlex.quote(sentinel_path)}\n"
    )
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(script_path, 0o700)

    try:
        subprocess.run([terminal, "--", "bash", script_path])
        return _poll_sentinel(sentinel_path)
    finally:
        for path in (script_path, sentinel_path):
            try:
                os.remove(path)
            except OSError:
                pass


def _spawn_windows(cmd: list[str], cwd: str) -> int:
    """Best-effort, unverified against real Windows (see module docstring).
    `.bat` equivalent of the sentinel-file pattern the other platforms use;
    `cmd /c start "" cmd /c <script>` opens a new console window. `cd /d`
    (not plain `cd`) so this also switches drive letters when cwd is on a
    different one than the new console's own default - same "don't inherit
    the caller's cwd for free" gap as the other two platforms."""
    run_id = uuid.uuid4().hex
    tmp_dir = tempfile.gettempdir()
    script_path = os.path.join(tmp_dir, f"llm-chat-spawn-{run_id}.bat")
    sentinel_path = os.path.join(tmp_dir, f"llm-chat-spawn-{run_id}.sentinel")

    quoted_cmd = " ".join(f'"{part}"' if " " in part else part for part in cmd)
    script = (
        f'@echo off\r\ncd /d "{cwd}"\r\n{quoted_cmd}\r\n'
        f"echo %errorlevel% > {sentinel_path}\r\n"
    )
    with open(script_path, "w") as f:
        f.write(script)

    try:
        subprocess.run(["cmd", "/c", "start", "", "cmd", "/c", script_path])
        return _poll_sentinel(sentinel_path)
    finally:
        for path in (script_path, sentinel_path):
            try:
                os.remove(path)
            except OSError:
                pass
