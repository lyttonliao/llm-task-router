"""Wrapper around headless `codex exec`, OpenAI Codex CLI's non-interactive
mode - the ChatGPT-subscription equivalent of `claude -p` (ie no separately
billed API key).

Verified end to end against a real install and a real authenticated call
(`codex-cli 0.145.0`, ChatGPT-account login) on 2026-07-22:
  - `--json` on `codex exec` is NOT a single result payload like claude's
    `--output-format json`. It streams one JSON *event* per line (JSONL) -
    a very different shape from claude_cli.py's single `json.loads(stdout)`.
  - `--output-last-message <FILE>` writes just the agent's final text to a
    file, which is a much simpler way to get the answer than parsing the
    JSONL event stream, so that's what this uses instead of `--json`.
  - `--skip-git-repo-check` matters because `codex exec` otherwise expects
    to run inside a git repo - this router has no reason to require that.
  - `--ask-for-approval` is a top-level `codex` option, NOT a valid `codex
    exec` flag - passing it fails with "unexpected argument". `codex exec`
    is non-interactive by construction and its session banner confirms it
    auto-defaults to `approval: never`, so nothing needs to be passed for
    this; only `--sandbox read-only` is needed as the closest analog to
    claude_cli.py's `--disallowed-tools "*"`. Still NOT equivalent: Codex
    has no flag that fully disables tool/shell use the way Claude Code's
    does - the model can still choose to run read-only shell commands (ls,
    cat, rg, ...) to gather context before answering, so a "simple" prompt
    can still cost more turns than claude_cli.py's single-shot completion.
    Don't assume cost/latency parity between the two adapters.
  - Which model names are valid depends on the auth mode: a ChatGPT-account
    login (used here) rejects some model names outright (confirmed: got a
    400 invalid_request_error for a guessed name that doesn't exist on this
    account's plan) - the default model when `--model` is omitted (seen in
    a real run here: `gpt-5.6-terra`) is a safer bet than guessing until
    tiers.py has real per-model quality-floor data to justify hardcoding one.

Still NOT verified:
  - There's no dollar-cost field anywhere in `codex exec`'s output. A real
    run's stderr banner does print a plain-text "tokens used\n<N>" line, but
    it's unstructured (not in `--output-last-message`'s file, not JSON) and
    converting token count to a dollar cost still needs per-model pricing
    this CLI doesn't expose - cost_usd/duration_ms stay 0.0/0 placeholders.
    Parsing that stderr line is possible but fragile; not done here.
  - Behavior on an actual model error/refusal (partial file vs. empty vs.
    non-zero exit) is covered for the "invalid model name" case seen above
    (non-zero exit, no file), but not for a genuine content refusal.

check_auth()'s two shapes are both confirmed against real output, not
guessed - a real logged-in run prints plain text "Logged in using ChatGPT"
at exit 0 (confirmed 2026-07-22); a real logged-out run, exercised via
`CODEX_HOME=<empty dir> codex login status` (2026-07-26) - which points the
CLI at a directory with no stored credentials without touching this
account's actual `~/.codex` login - prints plain text "Not logged in" at
exit 1. check_auth() treats anything containing "not logged in" as
unauthenticated and anything else without "logged in" in it (or a nonzero
exit) as unauthenticated too, rather than matching one exact string.
"""

import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from llm_task_router.schema import ProviderResult


def login(device_auth: bool = False) -> int:
    """Shells to `codex login` (browser flow) or `codex login --device-auth`
    when device_auth is set, for a future headless/SSH login path - nothing
    in this repo passes device_auth=True yet, it's an intentionally unwired
    parameter. Same "inherited stdio, don't trust the exit code" shape as
    claude_cli.login(): no capture_output/text/timeout so the user completes
    the flow directly in the terminal, and the caller must re-run
    check_auth() afterward rather than trust this return code."""
    cmd = ["codex", "login"]
    if device_auth:
        cmd.append("--device-auth")
    proc = subprocess.run(cmd)
    return proc.returncode


def check_auth() -> tuple[bool, str]:
    try:
        proc = subprocess.run(["codex", "login", "status"], capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return False, "auth check timed out"

    output = (proc.stdout + proc.stderr).strip()
    normalized = output.lower()
    if proc.returncode == 0 and "logged in" in normalized and "not logged in" not in normalized:
        return True, ""
    return False, output or "not logged in - run `codex login`"


def invoke(
    prompt: str,
    model: str,
    *,
    session_id: str | None = None,
    on_event: Callable[[dict], None] | None = None,
    permission_mode: str = "bypassPermissions",
) -> ProviderResult:
    """session_id is accepted for Provider-protocol conformance but ignored:
    unlike `claude -p --session-id`/`--resume`, `codex exec` has no flag to
    pre-assign a session id up front (confirmed via `codex exec --help`) -
    Codex allocates its own id on first run, and continuing one is the
    separate `codex exec resume <id> <prompt>` subcommand, not a flag on this
    call. Wiring real Codex continuity would need callers to branch on "is
    this the first message" and call a different entry point - out of scope
    while no Codex model is even routable (tiers.TIER_MODELS is Claude-only,
    see that module's docstring).

    on_event is likewise accepted-but-ignored, for the same Provider-protocol
    conformance reason: this adapter uses `--output-last-message` (a single
    file written at exit), not `--json`'s per-line event stream (see module
    docstring's "NOT verified"/shape notes) - there is nothing to stream from
    here yet. Wiring it would mean switching this adapter to parse `codex
    exec --json`'s JSONL instead, which is a real, not-yet-done piece of
    work, same as claude_cli.py's stream-json switch was - not worth doing
    speculatively while no tier routes to Codex at all.

    permission_mode is accepted-but-ignored too: `codex exec` already runs
    non-interactively at `approval: never` by default (see module docstring)
    and has no plan-mode analog to claude_cli.py's `--permission-mode plan`
    - repl.py's /plan command is unreachable through Codex anyway since no
    tier routes to it."""
    return _invoke_without_session(prompt, model)


def _invoke_without_session(prompt: str, model: str) -> ProviderResult:
    authenticated, auth_error = check_auth()
    if not authenticated:
        return ProviderResult(text="", cost_usd=0.0, duration_ms=0, error=f"auth check failed: {auth_error}")

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        last_message_path = Path(tmp.name)

    cmd = [
        "codex",
        "exec",
        prompt,
        "--model",
        model,
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--output-last-message",
        str(last_message_path),
    ]
    try:
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL
            )
        except subprocess.TimeoutExpired:
            return ProviderResult(text="", cost_usd=0.0, duration_ms=60_000, error="timeout")

        if proc.returncode != 0:
            return ProviderResult(text="", cost_usd=0.0, duration_ms=0, error=proc.stderr.strip() or "nonzero exit")

        text = last_message_path.read_text().strip() if last_message_path.exists() else ""
        return ProviderResult(text=text, cost_usd=0.0, duration_ms=0)
    finally:
        last_message_path.unlink(missing_ok=True)
