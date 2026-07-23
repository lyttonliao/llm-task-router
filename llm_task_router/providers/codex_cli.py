"""Wrapper around headless `codex exec`, OpenAI Codex CLI's non-interactive
mode - the ChatGPT-subscription equivalent of `claude -p` (ie no separately
billed API key).

UNVERIFIED: `codex` isn't installed on the machine this was scaffolded on, so
the exact command shape and JSON output schema below are a best guess by
analogy with claude_cli.py, not something actually run against real `codex`
output. Before routing any real traffic here: install `codex`, run `codex exec
--help`, and confirm/fix the flag names and the parsed keys.
"""

import json
import subprocess

from llm_task_router.schema import ProviderResult


def invoke(prompt: str, model: str) -> ProviderResult:
    cmd = [
        "codex",
        "exec",
        prompt,
        "--model",
        model,
        "--json",
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

    return ProviderResult(
        text=payload.get("result", ""),
        cost_usd=payload.get("total_cost_usd", 0.0),
        duration_ms=payload.get("duration_ms", 0),
        error=payload.get("error", ""),
    )
