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

Colors/bold/dim are gated behind ansi_enabled() (NO_COLOR env var, isatty()
check) so a piped/logged transcript never fills up with escape-code bytes -
but bullets, the thinking glyph, and blank-line spacing always render
regardless, so the transcript stays structurally readable either way. The
"grayed out" thinking status uses DIM rather than a hardcoded gray color:
DIM already reads as muted in effectively every terminal emulator and,
unlike a fixed 256-color code, stays legible across both light and dark
terminal themes.
"""

import os
import sys
from collections.abc import Callable

RESET = "\x1b[0m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"

CLAUDE_COLOR = "\x1b[38;5;208m"  # orange
CODEX_COLOR = "\x1b[38;5;36m"  # teal
TEXT_COLOR = "\x1b[38;5;255m"  # near-white, for Claude's own text/reasoning bullet
TOOL_COLOR = "\x1b[38;5;34m"  # green, for a running tool call
ERROR_COLOR = "\x1b[38;5;196m"  # red

PROVIDER_COLORS = {"claude": CLAUDE_COLOR, "codex": CODEX_COLOR}

BULLET = "⏺"
THINKING_GLYPH = "*"


def ansi_enabled() -> bool:
    """Per no-color.org: NO_COLOR present (any value) disables styling
    outright; otherwise styling is enabled only when stdout is a real
    terminal, never when piping to a file/log. Recomputed on every call, not
    cached at import - the cost (one env lookup + one isatty() syscall) is
    negligible next to a single provider subprocess call per message, and
    caching would freeze a stale answer for the life of the process (and
    make this untestable via monkeypatch)."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    return sys.stdout.isatty()


def style(code: str) -> str:
    """Gates one raw SGR code (BOLD/DIM/*_COLOR/RESET) behind
    ansi_enabled(). Structure - BULLET, THINKING_GLYPH, divider rule,
    blank-line spacing - is never routed through this: a piped/logged
    transcript should stay readable and greppable, only escape-code bytes
    are stripped."""
    return code if ansi_enabled() else ""


def provider_color(provider: str) -> str:
    return style(PROVIDER_COLORS.get(provider, ""))


def prompt() -> str:
    return f"{style(BOLD)}you>{style(RESET)} "


def header(decision) -> str:
    color = provider_color(decision.provider)
    return f"{color}{style(BOLD)}[{decision.provider}/{decision.model}, tier={decision.tier}]{style(RESET)}"


def error_line(error: str) -> str:
    return f"{style(ERROR_COLOR)}error: {error}{style(RESET)}"


def footer(result) -> str:
    return f"{style(DIM)}(cost ${result.cost_usd:.4f}, {result.duration_ms}ms){style(RESET)}"


def tool_line(name: str) -> str:
    return f"{style(TOOL_COLOR)}{BULLET} {name}{style(RESET)}"


def thinking_status() -> str:
    return f"{style(DIM)}{THINKING_GLYPH} thinking…{style(RESET)}"


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
