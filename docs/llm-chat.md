# llm-chat: interactive terminal client

`llm-chat` (`repl.py`) is a pure routing/classification layer as of
2026-07-31 (see "Spawn-per-message pivot" below): authenticate each
provider once at startup, then for every typed message classify via
`route()` and spawn a real native terminal (`terminal.py`) running the
routed `claude`/`codex` CLI call — the spawned session, not `llm-chat`,
handles everything else (tool use, follow-up turns, plan mode). Built so
engineers without an API budget get a live-routing chat experience off
existing Claude/ChatGPT subscriptions; no code path touches
`ANTHROPIC_API_KEY`/`OPENAI_API_KEY`.

## Spawn-per-message pivot (2026-07-31)

`chat_loop()` no longer calls `route_and_run()` or wires up
`tui.StreamRenderer` — full design at
`~/.claude/plans/what-s-our-next-goal-jazzy-tome.md` (this machine/user's
plans directory, not in-repo), summarized in this repo's `CLAUDE.md`. New
flow per message: `route()` classifies only (no provider call in-process),
`tui.header(decision)` prints the routing decision, then
`terminal.spawn_provider_session(decision.provider, decision.model,
session_id, message, resume=session_established)` opens a real terminal
window running that CLI call with an inherited TTY and blocks until it
exits — see `terminal.py`'s own module docstring for the disposable
wrapper-script + sentinel-file mechanism this uses to detect "the spawned
process exited" despite `open -a Terminal`/equivalents not blocking
themselves. `session_established` is a local bool in `chat_loop()`, not
`claude_cli.py`'s module-level `_established_sessions` set — each spawned
session is a separate OS process, so per-process in-memory tracking in
`claude_cli.py` doesn't apply to it at all; see `chat_loop()`'s own
docstring for exactly when this flips to `True`.

**Slash commands narrowed to `/exit`/`/quit` only.** `/help`, `/clear`, and
`/plan` are removed outright (not deprecated-in-place) — user's explicit
choice among three options (none at all / `/exit`+`/quit` only /
`/exit`+`/quit`+`/clear`) once `llm-chat` become strictly task routing.
`/plan`'s two-turn flow (see "Plan mode" below) and `/clear`'s session reset
(see "`/clear`" below) are both superseded, not replaced 1:1 — the user now
gets real `/plan` natively inside the spawned interactive session (no
headless `ExitPlanMode` limitation there), and there is no `/clear`
equivalent at all (no way to reset `session_id` mid-run; a fresh
conversation means restarting `llm-chat`).

**Not deleted in this pass**: `tui.StreamRenderer` and `repl.format_response()`
are now provably unreferenced by any application code (confirmed via grep —
`route_and_run()` survives only via `cli.py`'s one-shot `llm-route`
command, which touches neither). Per the referenced plan's explicit
"incrementally, not upfront" removal policy, they stay in the codebase
until a separate, later pass confirms the spawn flow has been used for a
while and removes what's still provably dead then — not as an automatic
follow-on to landing this wiring.

## Session continuity

`TaskRequest.session_id` is generated once per `chat_loop()` run (not per
message) and threaded through `route_and_run() -> provider.invoke(...,
session_id=...)` unconditionally. Every message in a session shares one
`session_id`, so history continues even as different messages route to
different Claude tiers/models — lets `llm-chat` stay a thin router in front
of real Claude Code functionality (tools, system prompt, CLAUDE.md/hooks)
instead of reimplementing an interface that mimics it. (Rejected first: a
bespoke reimplemented chat interface, and a raw PTY takeover injecting
`/model` mid-session — undocumented, more fragile.)

Confirmed against real `claude` 2.1.220 output (2026-07-26): the *first*
call per session uses `--session-id "$SID"`; every call after must use
`--resume "$SID"` instead — reusing `--session-id` on a second call fails
outright, but `--resume` correctly continues history across a `--model`
change. `claude_cli.py` tracks which session ids have had their establishing
call in a module-level `_established_sessions` set so callers just pass the
same `session_id` every time without knowing which flag applies.

## Full functionality, not cost-minimized

A deliberate choice distinct from `llm-eval-harness`'s adapter.
`providers/claude_cli.py` doesn't strip the system prompt or disable
tools/MCP: real tools, system prompt, CLAUDE.md/hooks all work, at real
per-call cost (~$0.07-0.30/call vs. eval_harness's ~$0.003-0.005 stripped).
Tool calls run under `--permission-mode bypassPermissions` since a headless
call has no TTY for an approval prompt — confirmed executing real commands
with zero approval prompts. **No default timeout as of 2026-07-31** — this
went through two fixed defaults (60s, then 300s) that both turned out to be
arbitrary guesses a real session blew past: 300s killed a real multi-file
editing turn mid-task, discarding the whole already-paid-for result even
though the terminal had already rendered everything up to that point live,
and separately, real sessions here have legitimately run well past 30
minutes (e.g. driving a full test sweep). `llm-chat` is interactive — the
user is physically present, and Ctrl-C already interrupts a blocking
`select.select()` call directly, making an internal wall-clock guess
redundant at best. `LLM_CHAT_TIMEOUT_S` is an opt-in ceiling for callers
that do want one (e.g. a non-interactive script with nobody watching to
press Ctrl-C) — resolved fresh per call via `claude_cli._resolve_timeout_s()`,
not cached at import, same reasoning as `tui.ansi_enabled()`.

## Cross-provider continuity

**Cross-provider mid-conversation continuity is still unsolved.**
`codex_cli.invoke` accepts `session_id` for `Provider`-protocol conformance
but ignores it — `codex exec` has no flag to pre-assign a session id
(continuation is the separate `codex exec resume <id>` subcommand). Only
works today because every tier maps to Claude; switching providers
mid-conversation would break continuity regardless, since each CLI's session
state is local to it.

## Streaming transport and ANSI styling

**`chat_loop()` no longer wires any of this up as of the 2026-07-31
spawn-per-message pivot (see above)** — `claude_cli.invoke()`'s
`stream-json` transport and `on_event`/`on_decision` plumbing described
below are still real (still used by `cli.py`'s one-shot `route_and_run()`
path and internally by `router.py`'s tier-3 classification fallback), and
`tui.StreamRenderer` still exists and is still tested, but `chat_loop()`
itself doesn't construct or feed it anymore. Left as history below, not
rewritten.

`claude_cli.py` switched from `--output-format json` to `--output-format
stream-json --include-partial-messages --verbose` (`--verbose` is required
— omitting it is a hard CLI error) — one JSON event per line as it arrives.
`invoke()` uses `subprocess.Popen`, with a `_drain()` helper reading
`stdout`/`stderr` concurrently via `select.select()` (avoids the classic
pipe deadlock a plain `for line in proc.stdout` loop would reintroduce).
With no `LLM_CHAT_TIMEOUT_S` set (see "Full functionality" above),
`select.select()`'s own `timeout=` arg is passed `None` and blocks
indefinitely; only when the env var is set does `_drain()` check a
wall-clock deadline every iteration. Unix-only —
`select()` doesn't support pipes on Windows, unverified there. `invoke()`
gained `on_event: Callable[[dict], None]` (`codex_cli.invoke()` accepts it
for interface conformance but ignores it). `route_and_run()` gained a
matching `on_event` passthrough plus `on_decision:
Callable[[RouteDecision], None]`, fired the instant `route()` resolves.

**Tool-call detail, not just a bare bullet (2026-07-31).** `StreamRenderer`
originally rendered a tool call as just `⏺ Edit` — a colored bullet and the
tool name, nothing else, confirmed to read as "blank" on a long real editing
turn (several Edit/Write calls in a row with no visible progress in between).
`claude_cli`'s NDJSON stream turns out to already include a top-level
`assistant` event per completed content block, carrying that block's fully
reconstructed data — for a `tool_use` block, its complete `input`, already
JSON-parsed by the CLI itself (confirmed against a real stream capture:
these arrive after the block's last `input_json_delta` but before its
`content_block_stop`, i.e. right after `_handle_block_start` already printed
the bullet for that same block). `StreamRenderer._handle_assistant()`
consumes these and calls `tui.tool_detail(name, input)`, which renders a real
unified diff (`difflib.unified_diff`) for `Edit`, a content preview for
`Write`, and the primary argument (command/file_path/pattern/etc.) for
everything else — capped at `MAX_DIFF_LINES`/`MAX_CONTENT_PREVIEW_LINES` so
one huge file doesn't flood the terminal. Appended directly under the
existing bullet (no `_separate()` call), not treated as a new segment.

`tui.py` is a stdlib-only ANSI styling module (escape codes, not
`rich`/`textual`), rendering in the visual spirit of Claude Code's own CLI
— not a pixel-exact clone, not a pty. `chat_loop()` wires a
`tui.StreamRenderer` into `on_event` for live token-by-token streaming;
`format_response()` remains the non-streaming formatter (still the pinned
test contract) but the success path no longer calls it, to avoid
double-printing.

**Restyle pass (2026-07-30).** `tui.py` gained `ansi_enabled()` (`NO_COLOR`
env var + `sys.stdout.isatty()`) and a `style(code)` wrapper every
color/bold/dim-emitting helper routes through — piping `llm-chat` to a file
no longer fills it with escape-code bytes, but bullets/glyphs/spacing always
render regardless, so a logged transcript stays structurally readable.
Recolored the tool-call bullet from amber to green and switched the
thinking glyph from `✻` to a literal `*` (still `DIM`, not a hardcoded gray
color — theme-safe across light/dark terminals). `StreamRenderer` now emits
a white bullet before Claude's own streamed text (previously unprefixed),
separates every segment (thinking→tool, tool→tool, tool→text, text→tool)
with a blank line instead of the old asymmetric one-sided spacing, and
gained a `start()` method that shows a "connecting…" status — cleared by
the same `\r\x1b[2K` trick as `thinking_status()` — for the dead air between
`on_decision` firing and the first real stream event (auth check/subprocess
startup/time-to-first-byte). `chat_loop()` prints a `tui.divider()` (sized
via `shutil.get_terminal_size`, 80-column fallback, rule only — never for
text wrapping) plus a blank line before every `you>` prompt after the
first, and a blank line after the user's line before the header, for
messages that actually route.

## Testing gotcha

Switching `invoke()` from `subprocess.run` to `subprocess.Popen` silently
invalidated every test that patched `subprocess.run` — they kept "passing"
while actually falling through to a real, unmocked call (caught by a
bash-level timeout after ~15-20 real calls had already gone out). Every
`claude_cli.py` test now patches `subprocess.Popen` directly. Re-verify with
the narrowest affected test before trusting a green suite after any future
change to this call mechanism.

## Known limitations and deferred work

**A static input-box frame was tried and reverted the same day.** `input()`
hands line editing to the terminal's own readline layer, which
overwrites/advances at the cursor — no way to keep a border in place once
text wraps. Doing it properly needs raw terminal mode (`termios`/`tty`) with
a hand-rolled line editor — declined twice as a much bigger build than
everything else in this pass. Back to a plain `tui.prompt()`.

**Plan mode (`/plan <description>`), implemented 2026-07-31, removed the
same day by the spawn-per-message pivot (see "Spawn-per-message pivot"
above)** — `/plan` no longer exists in `chat_loop()`; the user gets real
`/plan` natively inside the spawned interactive session instead, without
the headless `ExitPlanMode` limitation described below. Left as history:
two-turn flow sketched above, built once real `claude -p` behavior was
checked instead of assumed. Confirmed against a real call: `--permission-mode
plan` genuinely restricts the model to read-only tools (a `Write` attempt was
refused), but the `ExitPlanMode` tool errors out headlessly ("exists but is
not enabled in this context") — there is no structured plan handoff in `-p`
mode, only whatever the model says in its final text `result` once it can't
call that tool, which `repl.py` shows the user as-is. On approval, the
execute leg deliberately bypasses `route_and_run()`/`route()` entirely and
calls `PROVIDERS[decision.provider].invoke()` directly with the plan call's
already-resolved `decision.model`, `--resume`-ing the same session — a fixed
"proceed" confirmation has no real classification signal of its own and
would misroute on that lack of signal if run back through `route()`.
Consequence: the execute leg's real cost/duration are never written to
`routing_decisions` (no second `route()` call means no second row to attach
them to) — a real gap in shadow-eval cost accounting for any conversation
that uses `/plan`, not yet addressed. `claude_cli.invoke()`/`codex_cli.invoke()`
both gained a `permission_mode: str = "bypassPermissions"` parameter (Codex's
is accepted-but-ignored, same conformance pattern as `on_event` — `codex
exec` has no plan-mode analog) to carry this through
`route_and_run()` unchanged for every existing caller.

**`/clear`, added alongside plan mode (2026-07-31), removed the same day
by the spawn-per-message pivot (see above)** — `chat_loop()` has no
command to reset `session_id` mid-run anymore; a fresh conversation means
restarting `llm-chat`. Left as history: `/clear` used to reassign
`chat_loop()`'s `session_id` local to a fresh uuid, the one deliberate
exception to "generated once per run" (see `chat_loop()`'s docstring). The
next message establishes a brand new `claude` session instead of
`--resume`-ing the old one; no flag or provider-side "forget history" call
exists to mirror, this is purely a client-side pointer swap.

**Login always defers to the provider's own interactive command.**
`claude_cli.login()`/`codex_cli.login()` shell out with inherited stdio to
`claude auth login --claudeai`/`codex login` so the user completes the real
OAuth/device flow in the same terminal — `repl.py` never parses that flow
itself, and never trusts `login()`'s exit code as proof of success
(`check_auth()` afterward is the real source of truth).

**`known_models.py` is informational only** — never consulted by
`route()`/`route_and_run()`. A hardcoded table of model slugs confirmed
reachable via this account's calibration history, used solely for `repl.py`'s
startup summary. Same staleness risk as the Codex slug list it's drawn from.

**Authenticating Codex alone makes zero tiers routable**, since
`tiers.TIER_MODELS` maps every tier to `"claude"` — `main()` refuses to start
rather than letting every message fail individually.
`tests/test_repl.py::test_routable_tiers_against_real_tier_models_with_only_codex_authenticated`
is pinned against the real `TIER_MODELS` so it starts failing (in the good
way) the day a Codex tier gets calibrated in.

## Architectural decision: per-message routing vs. interactive UX

`llm-chat` classifies **every message independently** and routes to the appropriate tier — this is
non-negotiable. Messages within the same session can need different quality floors (a help query
routes cheap, a debug task routes expensive), and this per-message granularity is the project's
core value prop.

This design structurally forecloses full Claude Code interactive UX (arrow-key history, menu
selection, inline diff accept/reject). A session-owned terminal (real Claude Code) and a per-message
router (regaining control after each response to reclassify) are incompatible — one tool owns the TTY
continuously, the other intercepts between messages. Rebuilding Claude Code's interactive layer in
`prompt_toolkit` to bridge that gap trades a one-time engineering cost for permanent maintenance
burden, since the UI layer diverges every time Anthropic ships a feature there.

**The practical answer, as of the 2026-07-31 spawn-per-message pivot** (superseding the paragraph
below, which described the design *before* this shipped): `llm-chat` provides readline history via
`input()` and cost-aware routing, then spawns a real native terminal running the routed
`claude`/`codex` CLI call for **every** message, not just an opt-in escalation — see
"Spawn-per-message pivot" near the top of this doc. The user gets full native UX for every routed
message, not a light tool that occasionally hands off to a heavier one.

*Superseded, kept for history*: the original plan was narrower — basic interactivity plus
slash-command dispatch (`/plan`, `/clear`, etc., all since removed) as `llm-chat`'s own affordances,
with a future `/handoff` command reserved for conversations needing more intense/precise
communication (complex refactoring, high-stakes work, exploration needing iteration) to save the
session and launch a real `claude --resume` session. That framing generalized into spawn-per-message
by default instead of an opt-in escalation — same conclusion (don't rebuild Claude Code's interactive
layer), different delivery mechanism.

This isn't a limitation to work around — it's a design decision with real tradeoffs. Don't revisit
it.
