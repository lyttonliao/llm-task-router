"""Wrapper around headless `codex exec`, OpenAI Codex CLI's non-interactive
mode - the ChatGPT-subscription equivalent of `claude -p` (ie no separately
billed API key).

Command shape confirmed against a real install (`codex-cli 0.145.0`, `codex
exec --help`) on 2026-07-22 - not a guess anymore:
  - `--json` on `codex exec` is NOT a single result payload like claude's
    `--output-format json`. It streams one JSON *event* per line (JSONL) -
    a very different shape from claude_cli.py's single `json.loads(stdout)`.
  - `--output-last-message <FILE>` writes just the agent's final text to a
    file, which is a much simpler way to get the answer than parsing the
    JSONL event stream, so that's what this uses instead of `--json`.
  - `--skip-git-repo-check` matters because `codex exec` otherwise expects
    to run inside a git repo - this router has no reason to require that.
  - `--sandbox read-only --ask-for-approval never` is the closest analog to
    claude_cli.py's `--disallowed-tools "*"`, but it is NOT equivalent:
    Codex has no flag that fully disables tool/shell use the way Claude
    Code's does. read-only + never-ask means it can't write files and won't
    block waiting for a human, but the model can still choose to run
    read-only shell commands (ls, cat, rg, ...) to gather context before
    answering - so a "simple" prompt can still cost more turns than the
    single-shot completion claude_cli.py makes. Don't assume cost parity
    between the two adapters.

Still NOT verified - `codex doctor` shows no auth configured on this
machine, so no real authenticated call has been made against this code:
  - cost_usd/duration_ms below are hardcoded to 0.0/0 placeholders. Nothing
    in `codex exec --help` documents a per-call cost/usage field the way
    claude's `total_cost_usd`/`duration_ms` are documented - finding the
    real source (likely inside the `--json` event stream, if it exists at
    all) needs either a logged-in test run or reading Codex's own docs.
  - Whether `--output-last-message` behaves correctly on an error/refusal
    (empty file? partial text?) is unconfirmed.
Run `codex login` and re-verify with a real call before trusting this in
the router for anything but a --dry-run.
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
        "--ask-for-approval",
        "never",
        "--skip-git-repo-check",
        "--output-last-message",
        str(last_message_path),
    ]
    try:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return ProviderResult(text="", cost_usd=0.0, duration_ms=60_000, error="timeout")

        if proc.returncode != 0:
            return ProviderResult(text="", cost_usd=0.0, duration_ms=0, error=proc.stderr.strip() or "nonzero exit")

        text = last_message_path.read_text().strip() if last_message_path.exists() else ""
        return ProviderResult(text=text, cost_usd=0.0, duration_ms=0)
    finally:
        last_message_path.unlink(missing_ok=True)
