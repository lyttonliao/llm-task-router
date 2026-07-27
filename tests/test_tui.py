from llm_task_router import tui


def _writer():
    chunks = []
    return chunks, chunks.append


def test_text_delta_streams_verbatim_and_clears_no_thinking_indicator():
    chunks, write = _writer()
    renderer = tui.StreamRenderer(write_fn=write)

    renderer.handle({"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "pong"}}})
    renderer.finish()

    assert "".join(chunks) == "pong\n"


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
