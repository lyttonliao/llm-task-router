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
"""

import subprocess
import tempfile
from pathlib import Path

from llm_task_router.schema import ProviderResult


def invoke(prompt: str, model: str) -> ProviderResult:
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
