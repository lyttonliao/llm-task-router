"""Platform-dispatch terminal spawn for llm-chat's routing pivot: classify
in-process, then hand the actual provider call to a real native terminal so
the user gets full native Claude Code/Codex UX instead of this repo
re-rendering it - see CLAUDE.md's "Next step" section and the full design at
~/.claude/plans/what-s-our-next-goal-jazzy-tome.md (this machine/user's
plans directory, not in-repo; describes an earlier blocking version of this
module, since revised - see docs/llm-chat.md for the full history).

This module is deliberately narrow: it only launches/drives a terminal
running the provider CLI and does NOT decide provider/model (router.py
already does that).

Persistent tmux session, not spawn-per-message (revised 2026-07-31, fourth
same-day revision): earlier revisions spawned a brand new terminal window
and a brand new provider CLI process for EVERY message, using
`--session-id` on the first spawn and `--resume` on every spawn after so
each new process reloaded the same session's transcript from disk. Live use
showed this was the wrong model - it opened a new window per message
instead of continuing in one place, and because spawning was non-blocking,
a fast second message could fire its `--resume` call before the first
spawn's `--session-id` call had actually finished registering the session
on disk, an accepted-but-untested race that looked exactly like "every
message starts a new session."

The fix: spawn exactly ONE interactive provider CLI process per
`repl.chat_loop()` run, keep it alive under `tmux`, and deliver every
message (including the first) by injecting it into that same live process
via `tmux send-keys` - the same mechanism as a human typing into the
attached terminal. There is only ever one provider process, so the
`--resume` race is structurally gone, not just mitigated. `tmux` (not
`screen`) was chosen because each pane gets a real pty, so the provider
CLI's interactive raw-mode TUI behaves exactly as if a human were typing,
and its scripting interface (`send-keys -l` for literal text, `new-session
-d` for detached creation) is far less fragile than `screen -X stuff`. This
is a new external-CLI dependency (like `claude`/`codex` themselves, not a
new Python package) - `tmux_available()` below lets callers fail clearly at
startup instead of crashing mid-session.

Three functions replace the old single `spawn_provider_session()`:
`create_session()` (synchronous, starts the provider CLI detached under
tmux - no visible window yet), `attach_terminal()` (the ONE external
terminal spawn per run, non-blocking exactly as before, just running `tmux
attach` instead of the provider CLI directly), and `send_message()` (used
for every message after the session exists, and for `/model` switches via
`switch_model()` - see repl.py's chat_loop for the controller logic that
decides when each is called).

`cd` into the caller's cwd, added 2026-07-31: `open`/terminal-emulator
launches start the new window's shell at its own default startup directory
(the user's home directory on macOS), NOT the calling process's cwd -
confirmed against a real spawn: the spawned session's own startup banner
reported "launched claude in your home directory" instead of the repo
llm-chat was run from. `attach_terminal()`'s wrapper script still `cd`s
explicitly before running `tmux attach` for this reason (tmux's own
`new-session -c` handles the provider CLI's own cwd separately, in
`create_session()`).

Verified: macOS only (this machine's daily dev environment), by direct
manual testing. NOT verified at all: Linux (gnome-terminal/konsole/xterm
dispatch is written to the same pattern documented above but never run
against a real Linux desktop) and Windows (same - untested against a real
`cmd`/`start` invocation). Both are best-effort, matching this repo's
existing convention for flagging unverified platform behavior (see
docs/rough-edges.md's Windows select() note for the streaming transport).
Also NOT verified live yet as of this revision: whether `/model <name>`
(sent via `switch_model()`) needs a settling beat before the next
`send-keys` call lands cleanly, and whether tiers.py's stored model
strings are accepted as-is by `/model` (confirmed only that `/model
<name>` accepts a direct argument at all, via the installed claude
binary's own usage string - not confirmed end to end under tmux
injection)."""

import os
import platform
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid

_PROVIDER_CLI = {"claude": "claude", "codex": "codex"}

_LINUX_TERMINALS = ("gnome-terminal", "konsole", "xterm")

# Confirmed live (2026-07-31) against real claude 2.1.220: `/model <name>`
# doesn't switch immediately - it opens a "Switch model?" confirmation
# dialog ("1. Yes, switch ... / 2. No, go back") that needs a SECOND Enter
# to accept before the pane is a plain text input again. Sending the next
# message's keystrokes before that second Enter types them into the dialog
# instead of a text box - they're silently lost, even though the switch
# itself still goes through (option 1 is the default selection, so a lone
# Enter accepts it). This delay is a live-confirmed-necessary but
# not-stress-tested guess at how long the dialog takes to render and accept
# input - revisit if a switch is ever still observed swallowing the next
# message.
_MODEL_SWITCH_CONFIRM_DELAY_S = 0.5


def provider_cli_name(provider: str) -> str:
    try:
        return _PROVIDER_CLI[provider]
    except KeyError:
        raise ValueError(f"unknown provider: {provider!r}") from None


def tmux_available() -> bool:
    """Preflight check - callers (repl.main()) should check this once at
    startup and fail clearly rather than let create_session() raise a raw
    FileNotFoundError mid-session."""
    return shutil.which("tmux") is not None


def create_session(provider: str, model: str, session_id: str, cwd: str) -> None:
    """Starts the provider CLI interactively under a detached tmux session
    - a real pty exists (tmux always gives panes one) but no terminal
    window is visible yet, that's attach_terminal()'s job. Synchronous and
    local (no external window spawn), so unlike attach_terminal() this is
    safe to just call and wait for - it returns almost instantly.

    No message is baked into this command - the first message is delivered
    via send_message() exactly like every message after it, so callers
    never special-case "the first call" the way the old
    session_established/resume bookkeeping had to."""
    cli = provider_cli_name(provider)
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_id, "-c", cwd, "--", cli, "--model", model, "--session-id", session_id],
        check=True,
    )


def send_message(session_id: str, text: str) -> None:
    """Delivers text into the live tmux session's pane as a real tmux
    *paste* (load-buffer + paste-buffer -p), not `send-keys -l` (used by
    every earlier revision of this function) - deliberately changed
    2026-08-01 after a live gap surfaced: repl.py's input side moved to
    prompt_toolkit so a multi-line/pasted message no longer auto-submits
    line-by-line INTO llm-chat's own prompt, but `send-keys -l` sends raw
    literal bytes with no paste framing at all - any embedded '\\n' in
    `text` would still have hit the provider CLI's own pty as a bare
    linefeed, which its line editor (the same class of tool as
    prompt_toolkit) reads as a real Enter keypress, reintroducing the
    identical auto-submit bug one hop downstream, per-line inside the
    remote session instead of locally. `paste-buffer -p` fixes this at the
    transport: it wraps the buffer in the bracketed-paste escape sequences
    (ESC[200~ ... ESC[201~) that tell the receiving app "this is a paste,
    not typing" - but only actually emits them if that app has itself
    requested bracketed paste mode; confirmed live against a plain `cat`
    target (which never requests it) that the wrapping is silently skipped
    there while the literal bytes/newlines still land correctly, so this
    is safe to use unconditionally regardless of what's listening on the
    other end. `-d` deletes the paste buffer immediately after so buffers
    don't accumulate over a long-running session. Loading via stdin
    (`load-buffer -`) rather than a temp file/CLI arg sidesteps shell
    quoting entirely - text is opaque bytes to tmux either way, so a
    message that happens to contain '-' or ';' or a 'C-c'-looking
    substring is never at risk of being read as tmux flag/key syntax, the
    same guarantee `-l --` used to provide."""
    subprocess.run(["tmux", "load-buffer", "-"], input=text.encode(), check=True)
    subprocess.run(["tmux", "paste-buffer", "-p", "-d", "-t", session_id], check=True)
    subprocess.run(["tmux", "send-keys", "-t", session_id, "Enter"], check=True)


def switch_model(session_id: str, model: str) -> None:
    """Sends the provider CLI's own `/model <name>` slash command through
    the same send_message() injection primitive, then a second bare Enter
    to accept the confirmation dialog it opens (see
    _MODEL_SWITCH_CONFIRM_DELAY_S above - confirmed live 2026-07-31: without
    this second Enter, the switch itself still happens, but the caller's
    very next send_message() call types the real message into the
    confirmation dialog instead of a text box and it's silently lost).
    Kept as a separate named function from send_message() since it's
    conceptually a control command, not a user message, even though the
    text-injection mechanism is shared."""
    send_message(session_id, f"/model {model}")
    time.sleep(_MODEL_SWITCH_CONFIRM_DELAY_S)
    subprocess.run(["tmux", "send-keys", "-t", session_id, "Enter"], check=True)


def attach_terminal(session_id: str, cwd: str) -> None:
    """Launches a new native terminal running `tmux attach -t <session_id>`
    and returns as soon as that launch succeeds - does NOT wait for the
    attached session to exit (same non-blocking contract the old
    spawn_provider_session() had). This is the ONE external terminal spawn
    per repl.chat_loop() run - every message after the first is delivered
    via send_message()/switch_model() directly, with no further terminal
    spawning at all."""
    cmd = ["tmux", "attach", "-t", session_id]

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
    deletes itself as its last line, once the launched command has
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
