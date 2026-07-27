"""Thin wrapper around headless `claude -p`, adapted from llm-eval-harness's
eval_harness/claude_cli.py. Same "subscription CLI, not API key" rationale,
but unlike that adapter this one is deliberately full-functionality, not
cost-minimized: llm-chat is a real interactive client, not an offline
benchmarking harness, so this file does NOT strip the system prompt or
disable tools/MCP the way eval_harness/claude_cli.py does. Real tools, real
system prompt, real CLAUDE.md/hooks all work here, at the real per-call cost
that comes with that (~$0.07-0.30/call observed, vs. ~$0.003-0.005/call
stripped - see CLAUDE.md, "llm-chat" for the full tradeoff). This file is
independent from llm-eval-harness's copy (see that repo's "Why it's built
this way") - changing this adapter has no effect on that repo's calibration
cost model.

Session continuity: `--session-id <uuid>` establishes a new conversation;
reusing that same flag on a later call FAILS ("Session ID ... is already in
use" - confirmed against real output, not guessed, see CLAUDE.md). The
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
"""

import json
import subprocess

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


def invoke(prompt: str, model: str, system_prompt: str = "", *, session_id: str | None = None) -> ProviderResult:
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
        "json",
        "--permission-mode",
        "bypassPermissions",
    ]
    if system_prompt:
        cmd += ["--system-prompt", system_prompt]

    resuming = session_id is not None and session_id in _established_sessions
    if session_id:
        cmd += ["--resume", session_id] if resuming else ["--session-id", session_id]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return ProviderResult(text="", cost_usd=0.0, duration_ms=300_000, error="timeout")

    if proc.returncode != 0:
        return ProviderResult(text="", cost_usd=0.0, duration_ms=0, error=proc.stderr.strip() or "nonzero exit")

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return ProviderResult(text=proc.stdout, cost_usd=0.0, duration_ms=0, error="could not parse CLI json output")

    if payload.get("is_error"):
        return ProviderResult(text="", cost_usd=0.0, duration_ms=0, error=payload.get("result", "unknown CLI error"))

    # Only mark the session established once we know the call that was
    # supposed to create it actually succeeded - see module docstring.
    if session_id and not resuming:
        _established_sessions.add(session_id)

    return ProviderResult(
        text=payload.get("result", ""),
        cost_usd=payload.get("total_cost_usd", 0.0),
        duration_ms=payload.get("duration_ms", 0),
    )
