"""Thin wrapper around headless `claude -p`, adapted from llm-eval-harness's
eval_harness/claude_cli.py. Same rationale: run on the existing Claude Code
subscription instead of a separately-billed API key, and strip the default
system prompt + disable tools/MCP so we pay for a plain single-turn completion
(~$0.003-0.005/call) instead of a full agent turn (~$0.07/call).

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


def invoke(prompt: str, model: str, system_prompt: str = "") -> ProviderResult:
    authenticated, auth_error = check_auth()
    if not authenticated:
        return ProviderResult(text="", cost_usd=0.0, duration_ms=0, error=f"auth check failed: {auth_error}")

    cmd = [
        "claude",
        "-p",
        prompt,
        "--system-prompt",
        system_prompt,
        "--disallowed-tools",
        "*",
        "--strict-mcp-config",
        "--model",
        model,
        "--output-format",
        "json",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return ProviderResult(text="", cost_usd=0.0, duration_ms=60_000, error="timeout")

    if proc.returncode != 0:
        return ProviderResult(text="", cost_usd=0.0, duration_ms=0, error=proc.stderr.strip() or "nonzero exit")

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return ProviderResult(text=proc.stdout, cost_usd=0.0, duration_ms=0, error="could not parse CLI json output")

    if payload.get("is_error"):
        return ProviderResult(text="", cost_usd=0.0, duration_ms=0, error=payload.get("result", "unknown CLI error"))

    return ProviderResult(
        text=payload.get("result", ""),
        cost_usd=payload.get("total_cost_usd", 0.0),
        duration_ms=payload.get("duration_ms", 0),
    )
