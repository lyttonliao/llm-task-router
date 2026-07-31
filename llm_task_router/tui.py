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
(see docs/llm-chat.md - a raw PTY takeover of a live `claude` session was
considered and rejected as more fragile than the current mechanism).

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
import shutil
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
DIVIDER_CHAR = "─"
DIVIDER_FALLBACK_WIDTH = 80


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


def connecting_status() -> str:
    """Transient status for the dead-air window between the routing
    decision being known and the first real stream event arriving (auth
    check + subprocess startup + model time-to-first-byte) - cleared by the
    same \\r\\x1b[2K overwrite StreamRenderer already uses for
    thinking_status()."""
    return f"{style(DIM)}connecting…{style(RESET)}"


def text_bullet() -> str:
    """Prefixes Claude's own streamed text/reasoning output - emitted once
    per text content-block, not once per delta."""
    return f"{style(TEXT_COLOR)}{BULLET}{style(RESET)} "


def divider() -> str:
    """Horizontal rule printed between chat turns. Uses the real terminal
    width purely to size a rule line - NOT for text wrapping, which needs
    raw terminal mode and was declined for the same reason the boxed-input-
    frame idea was (see repl.py's chat_loop docstring)."""
    width = shutil.get_terminal_size(fallback=(DIVIDER_FALLBACK_WIDTH, 24)).columns
    return f"{style(DIM)}{DIVIDER_CHAR * width}{style(RESET)}"


def default_write(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


class StreamRenderer:
    """Renders claude_cli's per-line NDJSON events live, in the spirit of
    Claude Code's own CLI: a dim transient "connecting…"/"thinking…" status
    while nothing has streamed yet (each cleared once real output starts,
    never dumping the raw thinking text - that's collapsed in the real
    product too), one-line colored bullets for tool calls and for Claude's
    own text as each segment starts, blank-line spacing between segments,
    and the final answer streamed token-by-token as `text_delta` events
    arrive. write_fn receives raw chunks with no added newlines, so tests
    can assert exactly what would hit the terminal.

    Only three of claude_cli's event shapes are rendered (thinking/text/
    tool_use content-block starts, text_delta content-block deltas) - the
    rest (`system` init/status, `input_json_delta` partial tool args,
    `signature_delta`, the final `result` line) are real but not useful to
    show in a simple chat client and are silently ignored here, not
    unhandled - `claude_cli.invoke()` already extracts the `result` event
    itself independent of this renderer.
    """

    def __init__(self, write_fn: Callable[[str], None] = default_write):
        self._write = write_fn
        self._status_shown = False
        self._text_ever_started = False
        self._any_output = False
        self._in_text_block = False

    def start(self) -> None:
        """Call once, right after the routing header prints and before the
        (blocking) rest of route_and_run() runs - covers the dead-air
        window from auth-check/subprocess-startup/time-to-first-byte.
        Deliberately not done in __init__: constructing a renderer must have
        no I/O side effect, since existing tests construct one and feed it
        a single event, asserting exact output. Cleared by whichever real
        event arrives first, same as thinking_status()."""
        self._write(connecting_status())
        self._status_shown = True

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
        if btype == "thinking" and not self._status_shown and not self._text_ever_started:
            self._write(thinking_status())
            self._status_shown = True
        elif btype == "text":
            self._start_text_segment()
        elif btype == "tool_use":
            self._clear_status()
            self._separate()
            self._write(tool_line(block.get("name", "tool")))
            self._any_output = True
            self._in_text_block = False

    def _handle_delta(self, delta: dict) -> None:
        if delta.get("type") != "text_delta":
            return
        if not self._in_text_block:
            self._start_text_segment()
        self._write(delta.get("text", ""))

    def _start_text_segment(self) -> None:
        """Shared by an explicit `text` content_block_start and the
        defensive fallback in _handle_delta for a text_delta arriving with
        no preceding content_block_start observed."""
        self._clear_status()
        self._separate()
        self._write(text_bullet())
        self._any_output = True
        self._text_ever_started = True
        self._in_text_block = True

    def _separate(self) -> None:
        """Blank line between this turn's segments (thinking->tool,
        tool->tool, tool->text, text->tool) - not called within one text
        block's own deltas. No-op before the first segment of a turn."""
        if self._any_output:
            self._write("\n\n")

    def _clear_status(self) -> None:
        if self._status_shown:
            self._write("\r\x1b[2K")
            self._status_shown = False

    def finish(self) -> None:
        self._clear_status()
        if self._any_output:
            self._write("\n")
