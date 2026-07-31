"""Thin wrapper around headless `claude -p`, adapted from llm-eval-harness's
eval_harness/claude_cli.py. Same "subscription CLI, not API key" rationale,
but unlike that adapter this one is deliberately full-functionality, not
cost-minimized: llm-chat is a real interactive client, not an offline
benchmarking harness, so this file does NOT strip the system prompt or
disable tools/MCP the way eval_harness/claude_cli.py does. Real tools, real
system prompt, real CLAUDE.md/hooks all work here, at the real per-call cost
that comes with that (~$0.07-0.30/call observed, vs. ~$0.003-0.005/call
stripped - see docs/llm-chat.md for the full tradeoff). This file is
independent from llm-eval-harness's copy (see that repo's "Why it's built
this way") - changing this adapter has no effect on that repo's calibration
cost model.

Session continuity: `--session-id <uuid>` establishes a new conversation;
reusing that same flag on a later call FAILS ("Session ID ... is already in
use" - confirmed against real output, not guessed, see docs/llm-chat.md). The
correct mechanism for every call after the first is `--resume <uuid>`, which
does correctly continue the conversation even across a --model change.
`_established_sessions` tracks, per Python process, which session ids have
already had their first (`--session-id`) call so later calls with the same
id switch to `--resume` automatically - callers just pass the same
session_id every time and don't need to know which flag that becomes.

Tool calls run under `--permission-mode bypassPermissions` - confirmed
against real output to execute real commands (e.g. `pwd`) with zero approval
prompts (`permission_denials: []`). This is the user's own machine/account,
the same trust model this repo's adapters already extend to `claude -p`/
`codex exec` elsewhere - not something this file second-guesses per call.

check_auth() gates invoke() so an unauthenticated CLI fails fast with a clear
error instead of falling through to `claude -p`'s own nonzero-exit path (which
already surfaces auth errors, just after paying for the attempt and without a
consistent message - see llm-eval-harness's CLAUDE.md for what an
unauthenticated/inaccessible-account run looks like if it isn't caught: every
case comes back a parse_error with a misleadingly bad-looking aggregate
score). `claude auth status`'s logged-out shape is confirmed against real
output, not guessed: `env -u ANTHROPIC_API_KEY claude --bare auth status`
(2026-07-26) returns `{"loggedIn": false, "authMethod": "none",
"apiProvider": "firstParty"}` at exit 1 - `--bare` mode explicitly skips
keychain/OAuth reads per `claude --help`, so this exercises the real
"no credentials resolved" code path without touching this account's actual
stored login. Not tested: whether plain `claude auth status --json` (no
`--bare`) on a genuinely logged-out machine emits byte-identical JSON -
inferred to be the same schema, not separately confirmed.

Transport: `--output-format stream-json`, not `json`, added 2026-07-27.
`--output-format=stream-json` in print mode requires `--verbose` (confirmed:
omitting it is a hard CLI error, "requires --verbose") and, with
`--include-partial-messages`, emits one JSON event per line (NDJSON) instead
of one blob at EOF - confirmed against a real haiku call: `system`/init and
`status` events, then per-content-block `stream_event`s (`content_block_start`
with `content_block.type` in `{"thinking","text","tool_use"}`,
`content_block_delta` with `delta.type` in `{"thinking_delta","text_delta",
"signature_delta","input_json_delta"}`), and a final line with
`"type":"result"` carrying the same `result`/`total_cost_usd`/`duration_ms`/
`is_error` fields the old single-blob `json` format put at top level. `invoke()`
now reads NDJSON lines via `_drain()` instead of `json.loads()`-ing one
`proc.stdout` blob, and forwards every parsed event to an optional `on_event`
callback (used by `repl.py`'s live-streaming terminal renderer) before
checking `event["type"] == "result"`.

`_drain()` uses `select.select()` over *both* `stdout` and `stderr` pipes,
not a plain `for line in proc.stdout` loop - the latter would reintroduce the
classic subprocess pipe-deadlock (child blocks writing to a full stderr pipe
while we block waiting for stdout EOF that never comes), the exact failure
class `subprocess.run(capture_output=True)` avoided for free internally. Only
tested on Unix (`select()` doesn't support pipes on Windows) - fine for this
repo's Darwin dev environment, not verified elsewhere. There is no default
wall-clock deadline (see `_resolve_timeout_s()` / `LLM_CHAT_TIMEOUT_S`) -
`select.select()`'s own `timeout=` kwarg is passed `None` and blocks
indefinitely unless a caller opts into a ceiling via the env var; not
`subprocess.run`'s `timeout=` kwarg either way, since Popen has no
equivalent for an incrementally-read stream.
"""

import json
import os
import select
import subprocess
import time
from collections.abc import Callable

from llm_task_router.schema import ProviderResult


def login() -> int:
    """Shells to `claude auth login --claudeai` - the subscription login
    flow, not `--console` (API-key/billing), matching this repo's no-API-key
    philosophy. Deliberately no capture_output/text/timeout: stdio is
    inherited from the parent process so the user completes the OAuth flow
    (browser prompt, confirmation) directly in the same terminal, same as
    running the command themselves. The returned exit code is NOT trusted as
    proof the login actually succeeded - it's unconfirmed whether this can
    exit 0 without the flow completing (e.g. the user closes the browser tab
    partway through). Callers must re-run check_auth() afterward as the real
    source of truth, the same way invoke() already does before every call.
    """
    proc = subprocess.run(["claude", "auth", "login", "--claudeai"])
    return proc.returncode


def check_auth() -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["claude", "auth", "status", "--json"], capture_output=True, text=True, timeout=10
        )
    except subprocess.TimeoutExpired:
        return False, "auth check timed out"

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return False, "could not parse `claude auth status` output"

    if payload.get("loggedIn"):
        return True, ""
    return False, "not logged in - run `claude auth login`"


# Session ids that have already had their establishing (--session-id) call
# made in this process - see module docstring's "Session continuity" section.
_established_sessions: set[str] = set()


def _resolve_timeout_s() -> int | None:
    """Recomputed per call, not cached at import - same reasoning as
    tui.ansi_enabled(): a running session should be able to change this via
    env var without a restart. No default ceiling at all (returns None,
    meaning `_drain()` blocks indefinitely) - this went through two fixed
    defaults (60s, then 300s) that both turned out to be arbitrary guesses a
    real session blew past: a genuine multi-file tool-use turn got killed
    mid-task with the whole result discarded, and separately, real sessions
    here have run past 30 minutes on legitimate work (e.g. driving a full
    test sweep). `llm-chat` is an interactive terminal client - the user is
    physically present and Ctrl-C (which interrupts a blocking
    `select.select()` call directly, no cooperation needed from this
    function) is already the real way to bail out of a stuck call, making an
    internal wall-clock guess redundant at best and actively harmful when it
    fires on a call that was simply still working. LLM_CHAT_TIMEOUT_S is an
    opt-in ceiling for callers that do want one - e.g. a non-interactive
    script with nobody watching to press Ctrl-C."""
    raw = os.environ.get("LLM_CHAT_TIMEOUT_S")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def invoke(
    prompt: str,
    model: str,
    system_prompt: str = "",
    *,
    session_id: str | None = None,
    on_event: Callable[[dict], None] | None = None,
    disable_tools: bool = False,
) -> ProviderResult:
    """disable_tools (default False - every existing call site/test is
    unaffected) adds --disallowed-tools "*" --strict-mcp-config, the same
    two flags eval_harness's own stripped claude_cli.py uses, at that
    repo's documented ~$0.003-0.005/call vs this adapter's normal
    ~$0.07-0.30/call full-functionality cost. This exists for internal,
    one-shot calls (e.g. router.py's tier-3 classification fallback) that
    have no business running tools and shouldn't inherit llm-chat's
    full-functionality cost profile just because they share this adapter -
    llm-chat's own real messages must never pass this, by design."""
    authenticated, auth_error = check_auth()
    if not authenticated:
        return ProviderResult(text="", cost_usd=0.0, duration_ms=0, error=f"auth check failed: {auth_error}")

    cmd = [
        "claude",
        "-p",
        prompt,
        "--model",
        model,
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--verbose",
        "--permission-mode",
        "bypassPermissions",
    ]
    if disable_tools:
        cmd += ["--disallowed-tools", "*", "--strict-mcp-config"]
    if system_prompt:
        cmd += ["--system-prompt", system_prompt]

    resuming = session_id is not None and session_id in _established_sessions
    if session_id:
        cmd += ["--resume", session_id] if resuming else ["--session-id", session_id]

    timeout_s = _resolve_timeout_s()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    result_event, stderr_text, timed_out = _drain(proc, on_event, timeout_s)

    if timed_out:
        proc.kill()
        proc.wait()
        return ProviderResult(
            text="", cost_usd=0.0, duration_ms=(timeout_s or 0) * 1000, error="timeout"
        )

    proc.wait()
    if proc.returncode != 0 and result_event is None:
        return ProviderResult(text="", cost_usd=0.0, duration_ms=0, error=stderr_text.strip() or "nonzero exit")

    if result_event is None:
        return ProviderResult(text="", cost_usd=0.0, duration_ms=0, error="no result event received")

    if result_event.get("is_error"):
        return ProviderResult(
            text="",
            cost_usd=0.0,
            duration_ms=result_event.get("duration_ms", 0),
            error=result_event.get("result", "unknown CLI error"),
        )

    # Only mark the session established once we know the call that was
    # supposed to create it actually succeeded - see module docstring.
    if session_id and not resuming:
        _established_sessions.add(session_id)

    return ProviderResult(
        text=result_event.get("result", ""),
        cost_usd=result_event.get("total_cost_usd", 0.0),
        duration_ms=result_event.get("duration_ms", 0),
    )


def _drain(
    proc: subprocess.Popen, on_event: Callable[[dict], None] | None, timeout_s: int | None
) -> tuple[dict | None, str, bool]:
    """Reads stdout (NDJSON events) and stderr concurrently via select() so
    neither pipe's OS buffer can fill and deadlock the child - see module
    docstring. Returns (result_event, stderr_text, timed_out); result_event
    is the parsed `{"type": "result", ...}` line if one arrived before EOF/
    timeout, else None.

    timeout_s=None (the default, see _resolve_timeout_s()) means no deadline
    at all: `remaining` stays None and is passed straight through to
    select.select()'s own timeout arg, which blocks indefinitely - fine here
    since Ctrl-C still interrupts a blocking select() call directly."""
    deadline = time.monotonic() + timeout_s if timeout_s is not None else None
    open_streams = {proc.stdout, proc.stderr}
    result_event = None
    stderr_chunks: list[str] = []

    while open_streams:
        remaining = None
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return result_event, "".join(stderr_chunks), True

        ready, _, _ = select.select(list(open_streams), [], [], remaining)
        for stream in ready:
            line = stream.readline()
            if line == "":
                open_streams.discard(stream)
                continue
            if stream is proc.stderr:
                stderr_chunks.append(line)
                continue

            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if on_event:
                # Broad, deliberate catch: a bug in a terminal renderer
                # shouldn't lose an already-in-flight, already-paid-for model
                # call - same "an interactive session dying entirely over one
                # unanticipated exception is worse UX" reasoning repl.py's
                # chat_loop already applies to route_and_run().
                try:
                    on_event(event)
                except Exception:
                    pass

            if event.get("type") == "result":
                result_event = event

    return result_event, "".join(stderr_chunks), False
