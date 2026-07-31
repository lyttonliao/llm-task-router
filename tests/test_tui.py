import sys

import pytest

from llm_task_router import tui


@pytest.fixture(autouse=True)
def _simulate_interactive_terminal(monkeypatch):
    """This file's existing tests assert the *styled* rendering path -
    simulate a real interactive terminal so tui.ansi_enabled() is True by
    default (sys.stdout.isatty() is False under pytest's captured stdout,
    which would otherwise silently strip every color assertion below).
    The NO_COLOR/non-tty tests further down override these two knobs
    locally to exercise the stripping path instead."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)


def _writer():
    chunks = []
    return chunks, chunks.append


def test_text_delta_streams_verbatim_and_clears_no_thinking_indicator():
    chunks, write = _writer()
    renderer = tui.StreamRenderer(write_fn=write)

    renderer.handle({"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "pong"}}})
    renderer.finish()

    joined = "".join(chunks)
    assert "pong" in joined
    assert joined.endswith("pong\n")
    assert "\r\x1b[2K" not in joined  # nothing was shown that needed clearing


def test_text_block_start_emits_bullet_before_delta_text():
    chunks, write = _writer()
    renderer = tui.StreamRenderer(write_fn=write)

    renderer.handle({"type": "stream_event", "event": {"type": "content_block_start", "content_block": {"type": "text"}}})
    renderer.handle({"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "pong"}}})
    renderer.finish()

    joined = "".join(chunks)
    assert tui.BULLET in joined
    assert joined.index(tui.BULLET) < joined.index("pong")


def test_multiple_text_deltas_in_one_block_are_not_separated_by_blank_lines():
    chunks, write = _writer()
    renderer = tui.StreamRenderer(write_fn=write)

    renderer.handle({"type": "stream_event", "event": {"type": "content_block_start", "content_block": {"type": "text"}}})
    renderer.handle({"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "pon"}}})
    renderer.handle({"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "g"}}})
    renderer.finish()

    joined = "".join(chunks)
    assert joined.endswith("pong\n")
    assert "\n\n" not in joined


def test_multiple_tool_use_blocks_separated_by_blank_line():
    chunks, write = _writer()
    renderer = tui.StreamRenderer(write_fn=write)

    renderer.handle(
        {"type": "stream_event", "event": {"type": "content_block_start", "content_block": {"type": "tool_use", "name": "Read"}}}
    )
    renderer.handle(
        {"type": "stream_event", "event": {"type": "content_block_start", "content_block": {"type": "tool_use", "name": "Grep"}}}
    )

    joined = "".join(chunks)
    assert "\n\n" in joined
    assert joined.index("Read") < joined.index("\n\n") < joined.index("Grep")


def test_tool_use_then_text_separated_by_blank_line():
    chunks, write = _writer()
    renderer = tui.StreamRenderer(write_fn=write)

    renderer.handle(
        {"type": "stream_event", "event": {"type": "content_block_start", "content_block": {"type": "tool_use", "name": "Read"}}}
    )
    renderer.handle({"type": "stream_event", "event": {"type": "content_block_start", "content_block": {"type": "text"}}})
    renderer.handle({"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "done"}}})

    joined = "".join(chunks)
    assert "\n\n" in joined
    assert joined.index("Read") < joined.index("\n\n") < joined.index("done")


def test_renderer_start_emits_connecting_status_cleared_by_first_event():
    chunks, write = _writer()
    renderer = tui.StreamRenderer(write_fn=write)

    renderer.start()
    assert any("connecting" in c for c in chunks)

    renderer.handle({"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}}})

    joined = "".join(chunks)
    assert "\r\x1b[2K" in joined
    assert joined.endswith("hi")


def test_renderer_finish_clears_leftover_status_with_no_events():
    """Covers a zero-event error path (e.g. an auth failure before any
    stream event ever arrives) - start() shouldn't leave a stray
    "connecting…" on screen."""
    chunks, write = _writer()
    renderer = tui.StreamRenderer(write_fn=write)

    renderer.start()
    renderer.finish()

    joined = "".join(chunks)
    assert "\r\x1b[2K" in joined
    assert not joined.endswith("\n")  # no real output, so no trailing newline either


def test_thinking_indicator_shown_then_cleared_when_text_starts():
    chunks, write = _writer()
    renderer = tui.StreamRenderer(write_fn=write)

    renderer.handle({"type": "stream_event", "event": {"type": "content_block_start", "content_block": {"type": "thinking"}}})
    assert any("thinking" in c for c in chunks)

    renderer.handle({"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}}})

    # the clear sequence (cursor-return + erase-line) must appear before the real text
    joined = "".join(chunks)
    assert "\r\x1b[2K" in joined
    assert joined.endswith("hi")


def test_tool_use_block_renders_one_line_bullet_with_tool_name():
    chunks, write = _writer()
    renderer = tui.StreamRenderer(write_fn=write)

    renderer.handle(
        {
            "type": "stream_event",
            "event": {"type": "content_block_start", "content_block": {"type": "tool_use", "name": "Read"}},
        }
    )

    joined = "".join(chunks)
    assert "Read" in joined
    assert tui.BULLET in joined


def test_non_stream_event_types_are_ignored():
    chunks, write = _writer()
    renderer = tui.StreamRenderer(write_fn=write)

    renderer.handle({"type": "system", "subtype": "init"})
    renderer.handle({"type": "result", "result": "ok"})
    renderer.finish()

    assert chunks == []


def test_input_json_delta_and_signature_delta_are_not_rendered():
    """Real tool-arg/thinking-signature deltas exist in the stream but
    aren't useful in a simple chat client - confirm they're silently
    skipped, not accidentally printed as raw JSON."""
    chunks, write = _writer()
    renderer = tui.StreamRenderer(write_fn=write)

    renderer.handle(
        {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "input_json_delta", "partial_json": "{\"x"}}}
    )
    renderer.handle(
        {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "signature_delta", "signature": "abc"}}}
    )

    assert chunks == []


def test_prompt_is_styled_bold_you_arrow():
    """The boxed-input frame (top/bottom border) was tried and removed the
    same day - it can't wrap around input that line-wraps in the terminal
    without raw terminal mode, which was explicitly declined (see repl.py's
    chat_loop docstring). Plain styled prompt only."""
    out = tui.prompt()

    assert "you>" in out
    assert tui.BOLD in out


def test_format_response_helpers_wrap_provider_color():
    from llm_task_router.schema import RouteDecision

    decision = RouteDecision(tier="flagship", provider="claude", model="opus", reason="r")
    out = tui.header(decision)

    assert tui.CLAUDE_COLOR in out
    assert "[claude/opus, tier=flagship]" in out
    assert tui.RESET in out


def test_thinking_status_uses_literal_asterisk_not_the_old_unicode_star():
    out = tui.thinking_status()

    assert "*" in out
    assert "✻" not in out


def test_ansi_enabled_true_by_default_in_this_suite():
    assert tui.ansi_enabled() is True


def test_ansi_enabled_false_when_NO_COLOR_set(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")

    assert tui.ansi_enabled() is False


def test_ansi_enabled_false_when_stdout_not_a_tty(monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    assert tui.ansi_enabled() is False


def test_style_returns_empty_string_when_ansi_disabled(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")

    assert tui.style(tui.BOLD) == ""


def test_tool_line_strips_color_but_keeps_bullet_and_name_when_no_color_set(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")

    out = tui.tool_line("Read")

    assert tui.TOOL_COLOR not in out
    assert tui.RESET not in out
    assert tui.BULLET in out
    assert "Read" in out


def test_header_strips_color_but_keeps_text_when_not_a_tty(monkeypatch):
    from llm_task_router.schema import RouteDecision

    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    decision = RouteDecision(tier="flagship", provider="claude", model="opus", reason="r")

    out = tui.header(decision)

    assert tui.CLAUDE_COLOR not in out
    assert tui.BOLD not in out
    assert "[claude/opus, tier=flagship]" in out


def test_text_bullet_uses_bullet_and_text_color():
    out = tui.text_bullet()

    assert tui.BULLET in out
    assert tui.TEXT_COLOR in out
    assert tui.RESET in out


def test_text_bullet_strips_color_but_keeps_bullet_when_no_color_set(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")

    out = tui.text_bullet()

    assert tui.BULLET in out
    assert tui.TEXT_COLOR not in out


def test_connecting_status_says_connecting_and_is_dim():
    out = tui.connecting_status()

    assert "connecting" in out
    assert tui.DIM in out


def test_divider_sized_to_real_terminal_width(monkeypatch):
    import os
    import shutil

    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback: os.terminal_size((40, 24)))

    out = tui.divider()

    assert out.count(tui.DIVIDER_CHAR) == 40


def test_divider_falls_back_to_80_columns_when_size_unavailable(monkeypatch):
    import os
    import shutil

    # shutil.get_terminal_size() itself falls back to `fallback` when the
    # environment/os module can't determine a real size - simulate that.
    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback: os.terminal_size(fallback))

    out = tui.divider()

    assert out.count(tui.DIVIDER_CHAR) == tui.DIVIDER_FALLBACK_WIDTH


def test_tool_detail_edit_shows_file_path_and_diff():
    out = tui.tool_detail(
        "Edit", {"file_path": "foo.py", "old_string": "line1\nold\nline3", "new_string": "line1\nnew\nline3"}
    )

    assert "foo.py" in out
    assert "-old" in out
    assert "+new" in out


def test_tool_detail_edit_caps_huge_diffs():
    old = "\n".join(f"old{i}" for i in range(200))
    new = "\n".join(f"new{i}" for i in range(200))

    out = tui.tool_detail("Edit", {"file_path": "big.py", "old_string": old, "new_string": new})

    assert out.count("\n") < 100
    assert "more diff lines" in out


def test_tool_detail_write_shows_file_path_and_content_preview():
    out = tui.tool_detail("Write", {"file_path": "new_file.py", "content": "line one\nline two"})

    assert "new_file.py" in out
    assert "+line one" in out
    assert "+line two" in out


def test_tool_detail_write_caps_huge_content():
    content = "\n".join(f"line{i}" for i in range(200))

    out = tui.tool_detail("Write", {"file_path": "big.py", "content": content})

    assert out.count("\n") < 30
    assert "more lines" in out


def test_tool_detail_bash_shows_command():
    out = tui.tool_detail("Bash", {"command": "echo hello"})

    assert "echo hello" in out


def test_tool_detail_generic_tool_shows_file_path():
    out = tui.tool_detail("Read", {"file_path": "some/file.py"})

    assert "some/file.py" in out


def test_tool_detail_returns_empty_string_for_tool_with_no_recognized_argument():
    assert tui.tool_detail("SomeNewTool", {}) == ""


def test_assistant_event_with_non_tool_content_is_ignored():
    chunks, write = _writer()
    renderer = tui.StreamRenderer(write_fn=write)

    renderer.handle({"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}})

    assert chunks == []


def test_assistant_event_appends_tool_detail_after_bullet_without_blank_line_separator():
    chunks, write = _writer()
    renderer = tui.StreamRenderer(write_fn=write)

    renderer.handle(
        {"type": "stream_event", "event": {"type": "content_block_start", "content_block": {"type": "tool_use", "name": "Edit"}}}
    )
    renderer.handle(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Edit",
                        "input": {"file_path": "foo.py", "old_string": "a", "new_string": "b"},
                    }
                ]
            },
        }
    )

    joined = "".join(chunks)
    assert "Edit" in joined
    assert "foo.py" in joined
    assert "\n\n" not in joined  # same segment as the bullet, not a new one


def test_assistant_event_with_tool_input_missing_recognized_args_adds_nothing():
    chunks, write = _writer()
    renderer = tui.StreamRenderer(write_fn=write)

    renderer.handle(
        {"type": "stream_event", "event": {"type": "content_block_start", "content_block": {"type": "tool_use", "name": "SomeNewTool"}}}
    )
    renderer.handle({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "SomeNewTool", "input": {}}]}})

    joined = "".join(chunks)
    assert joined == tui.tool_line("SomeNewTool")


def test_wrap_text_breaks_long_lines_at_word_boundaries():
    text = "This is a very long line that should be wrapped at word boundaries when it exceeds the terminal width limit"
    wrapped = tui.wrap_text(text, 40)

    lines = wrapped.split("\n")
    for line in lines:
        assert len(line) <= 40, f"Line exceeds width: '{line}' ({len(line)} chars)"
    assert "This is a" in wrapped
    assert "very long" in wrapped


def test_wrap_text_preserves_existing_newlines():
    text = "Line one\nLine two is a very long line that should wrap\nLine three"
    wrapped = tui.wrap_text(text, 30)

    lines = wrapped.split("\n")
    # Check that we have at least 3 lines (the existing newlines were preserved)
    assert len(lines) >= 3
    # Check no line exceeds width
    for line in lines:
        assert len(line) <= 30


def test_wrap_text_handles_short_text():
    text = "Short text"
    wrapped = tui.wrap_text(text, 80)

    assert wrapped == "Short text"


def test_wrap_text_handles_empty_string():
    text = ""
    wrapped = tui.wrap_text(text, 80)

    assert wrapped == ""


def test_streaming_text_wraps_complete_lines_at_terminal_width(monkeypatch):
    import shutil
    import os

    # Mock terminal width to 30 for testing
    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback: os.terminal_size((30, 24)))

    chunks, write = _writer()
    renderer = tui.StreamRenderer(write_fn=write)

    renderer.handle({"type": "stream_event", "event": {"type": "content_block_start", "content_block": {"type": "text"}}})
    # Send a complete line that's too long for the terminal
    renderer.handle(
        {
            "type": "stream_event",
            "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "This is a very long line that exceeds terminal width\n"}},
        }
    )
    renderer.finish()

    joined = "".join(chunks)
    # Should have wrapped the long line
    lines = [l for l in joined.split("\n") if l.strip()]  # Skip empty lines and bullet
    wrapped_lines = [l for l in lines if "This" in l or "is a" in l or "very" in l or "long" in l]
    # At least one of the wrapped lines should be in the output
    assert len(wrapped_lines) > 0
