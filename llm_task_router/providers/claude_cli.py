"""Thin wrapper around headless `claude -p`, adapted from llm-eval-harness's
eval_harness/claude_cli.py. Same rationale: run on the existing Claude Code
subscription instead of a separately-billed API key, and strip the default
system prompt + disable tools/MCP so we pay for a plain single-turn completion
(~$0.003-0.005/call) instead of a full agent turn (~$0.07/call).
"""

import json
import subprocess

from llm_task_router.schema import ProviderResult


def invoke(prompt: str, model: str, system_prompt: str = "") -> ProviderResult:
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
