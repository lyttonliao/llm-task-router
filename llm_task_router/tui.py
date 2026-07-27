"""Minimal ANSI styling for `llm-chat`'s terminal output - stdlib only (raw
escape codes), no `rich`/`textual`, matching this repo's zero-third-party-
dependency rule (see CLAUDE.md, "Why it's built this way").

This is NOT a pty and not a redraw-based TUI - repl.py still just writes to
ordinary stdout, incrementally. What this module adds is color and live,
token-by-token printing in the same visual language as Claude Code's own
CLI (a colored role bullet, one-line tool-call summaries, a dim cost/
duration footer, a transient "thinking" status) - not a pixel-exact clone of
it, which isn't achievable without reverse-engineering a closed-source
renderer, and was explicitly rejected as a design direction for this repo
(see CLAUDE.md, "llm-chat: interactive terminal client" - a raw PTY takeover
of a live `claude` session was considered and rejected as more fragile than
the current mechanism).
"""

import sys
from collections.abc import Callable

RESET = "\x1b[0m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"

CLAUDE_COLOR = "\x1b[38;5;208m"  # orange
CODEX_COLOR = "\x1b[38;5;36m"  # teal
TOOL_COLOR = "\x1b[38;5;178m"  # amber
ERROR_COLOR = "\x1b[38;5;196m"  # red

PROVIDER_COLORS = {"claude": CLAUDE_COLOR, "codex": CODEX_COLOR}

BULLET = "⏺"


def provider_color(provider: str) -> str:
    return PROVIDER_COLORS.get(provider, "")


def prompt() -> str:
    return f"{BOLD}you>{RESET} "


def header(decision) -> str:
    color = provider_color(decision.provider)
    return f"{color}{BOLD}[{decision.provider}/{decision.model}, tier={decision.tier}]{RESET}"


def error_line(error: str) -> str:
    return f"{ERROR_COLOR}error: {error}{RESET}"


def footer(result) -> str:
    return f"{DIM}(cost ${result.cost_usd:.4f}, {result.duration_ms}ms){RESET}"


def tool_line(name: str) -> str:
    return f"{TOOL_COLOR}{BULLET} {name}{RESET}"


def thinking_status() -> str:
    return f"{DIM}✻ thinking…{RESET}"


def default_write(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


class StreamRenderer:
    """Renders claude_cli's per-line NDJSON events live, in the spirit of
    Claude Code's own CLI: a dim transient "thinking…" status while the
    model reasons (cleared once real output starts, never dumping the raw
    thinking text - that's collapsed in the real product too), one-line
    colored bullets for tool calls as they start, and the final answer
    streamed token-by-token as `text_delta` events arrive. write_fn receives
    raw chunks with no added newlines, so tests can assert exactly what
    would hit the terminal.

    Only two of claude_cli's event shapes are rendered (thinking/tool_use
    content-block starts, text_delta content-block deltas) - the rest
    (`system` init/status, `input_json_delta` partial tool args,
    `signature_delta`, the final `result` line) are real but not useful to
    show in a simple chat client and are silently ignored here, not
    unhandled - `claude_cli.invoke()` already extracts the `result` event
    itself independent of this renderer.
    """

    def __init__(self, write_fn: Callable[[str], None] = default_write):
        self._write = write_fn
        self._thinking_shown = False
        self._text_started = False

    def handle(self, event: dict) -> None:
        if event.get("type") != "stream_event":
            return
        inner = event.get("event", {})
        itype = inner.get("type")
        if itype == "content_block_start":
            self._handle_block_start(inner.get("content_block", {}))
        elif itype == "content_block_delta":
            self._handle_delta(inner.get("delta", {}))

    def _handle_block_start(self, block: dict) -> None:
        btype = block.get("type")
        if btype == "thinking" and not self._thinking_shown and not self._text_started:
            self._write(thinking_status())
            self._thinking_shown = True
        elif btype == "tool_use":
            self._clear_thinking()
            prefix = "\n" if self._text_started else ""
            self._write(f"{prefix}{tool_line(block.get('name', 'tool'))}\n")

    def _handle_delta(self, delta: dict) -> None:
        if delta.get("type") != "text_delta":
            return
        self._clear_thinking()
        self._write(delta.get("text", ""))
        self._text_started = True

    def _clear_thinking(self) -> None:
        if self._thinking_shown:
            self._write("\r\x1b[2K")
            self._thinking_shown = False

    def finish(self) -> None:
        if self._text_started:
            self._write("\n")
