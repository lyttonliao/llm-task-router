# llm-chat: interactive terminal client

`llm-chat` (`repl.py`) is a pure routing/classification layer as of
2026-07-31 (see "Persistent tmux session" below, current; every other
section down through "Spawn-per-message pivot" is superseded but kept for
history — the design churned through four revisions in one day, each
driven by actually using the previous one): authenticate each provider
once at startup, then for every typed message classify via `route()` and
deliver it into a single persistent `tmux`-backed terminal session
(`terminal.py`) running the routed `claude`/`codex` CLI, without ever
spawning a second process for that run — the tmux session, not
`llm-chat`, handles everything else (tool use, follow-up turns, plan
mode, more turns on the same task), and `llm-chat`'s own prompt is
available again immediately after each message is delivered, not once
that session exits. Built so engineers without an API budget get a
live-routing chat experience off existing Claude/ChatGPT subscriptions;
no code path touches `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`.

## Paste-safe multi-line input (2026-08-01)

**The problem, found by actually pasting into `llm-chat`**: `repl.py`'s
prompt used bare `input()`, which hands line editing straight to the
terminal's own readline layer with no real paste awareness — pasting
multi-line text (resume bullets, a code block) submitted each line the
instant its embedded `\n` hit stdin, instead of landing as one message the
user could review/add context to before sending. This is the same root
limitation the "static input-box frame" attempt below already ran into,
just surfacing through a different symptom.

**Fix, local side**: `repl.py` now builds its `input_fn` via
`build_input_fn()`, backed by `prompt_toolkit.PromptSession` instead of
`input()`. `PromptSession` auto-negotiates bracketed paste on any
vt100-compatible terminal — pasted text (even multi-line) lands as one
literal buffer edit, not per-line Enter presses — and picks up real
arrow-key/Home/End line editing plus session-scoped history (up/down
recalls this run's earlier messages, `InMemoryHistory`, no persistence
across runs). A `Meta+Enter` (`escape`, `enter`) key binding inserts a
literal newline for deliberately composing a multi-line message by hand
(e.g. adding context above/below pasted content) — plain `Enter` still
submits, matching prior behavior for the common single-line case.
`tui.prompt()`'s raw ANSI escape codes are wrapped in `ANSI(...)` so
prompt_toolkit renders them instead of treating the escape bytes as
literal characters (it manages the terminal's screen buffer itself, unlike
`input()` writing straight through). New runtime dependency — the first
addition to `pyproject.toml` outside the tier-2 classifier's already-
discussed exception (`sentence-transformers`/`psycopg`), justified the
same way: multi-line paste-safe editing isn't something worth
hand-rolling raw `termios`/`tty` for, which was already declined once (see
"Known limitations" below).

**Fix, remote side — the less obvious half**: fixing only the local prompt
would have moved the exact same bug one hop downstream. `terminal.py`'s
`send_message()` used to inject text via `tmux send-keys -l` — literal
bytes with no paste framing at all. A multi-line message that now survives
intact locally would still hit the provider CLI's own pty as bare `\n`
bytes, which its line editor reads as real Enter keypresses, splitting the
message into separate submits inside the *remote* session instead of
`llm-chat`'s own prompt. Fixed by switching `send_message()` to tmux's
real paste primitive: `tmux load-buffer -` (reads text from stdin, so
messages containing `-`/`;`/`C-c`-looking substrings stay safe without
needing `-l --`'s guard) followed by `tmux paste-buffer -p -d -t
<session>` — `-p` wraps the buffer in bracketed-paste escape codes
(`ESC[200~`/`ESC[201~`), but tmux only actually emits them if the pane's
application has itself requested bracketed paste mode; confirmed live
against a plain `cat` target (which never requests it) that the wrapping
is silently skipped there while the literal newlines still land correctly
— safe to use unconditionally regardless of what's listening on the other
end. `-d` deletes the buffer immediately after so buffers don't
accumulate over a long session. A separate `tmux send-keys ... Enter`
still follows, unchanged, to submit.

**Live-verified end to end (2026-08-01)**, following this repo's standing
rule that `terminal.py`'s mocked tests only prove command construction,
never real behavior: a real detached session running `claude --model
haiku` was sent `"Compute the following two numbers summed together on
the next line:\n40000 + 2222"` via the new `send_message()`. `tmux
capture-pane` showed both lines under a single `❯` prompt and one correct
answer (`42222`) — confirming the message landed as one paste, not two
separate submits, exactly the failure mode this fix targets.

## Persistent tmux session, not spawn-per-message (2026-07-31, fourth same-day revision)

**Supersedes "Non-blocking spawn" below**, which itself superseded "One
spawn per run", which superseded the original "Spawn-per-message pivot" -
four revisions to this same design landed in one day, each driven by
actually using the previous version.

**What was wrong with spawning per message**: every message (after the
first) opened a *brand new terminal window and a brand new `claude`
process*, using `--resume <session_id>` so the new process reloaded the
prior transcript from disk. Live use surfaced this as the wrong model —
the user wants one continuing place to watch the conversation, not a new
window per message — and because spawning was non-blocking (see "Non-blocking
spawn" below), a fast second message could fire its `--resume` call before
the first spawn's `--session-id` call had actually finished registering
the session on disk. That race was accepted as a documented tradeoff at
the time; it turned out to be indistinguishable, from the user's side,
from "every message starts a new session" - the exact complaint that
triggered this revision.

**The fix**: stop spawning a new provider CLI process per message
entirely. `terminal.py` gained three functions that replace
`spawn_provider_session()`:

- `create_session(provider, model, session_id, cwd)` — synchronous, runs
  once per `chat_loop()` run: `tmux new-session -d -s <session_id> -c
  <cwd> -- <cli> --model <model> --session-id <session_id>`. Starts the
  provider CLI interactively under a *detached* tmux session — tmux always
  gives a pane a real pty, so the CLI's raw-mode interactive TUI works
  exactly as if a human were typing, with no window visible yet.
- `attach_terminal(session_id, cwd)` — the one and only external terminal
  spawn per run, reusing the same non-blocking wrapper-script machinery
  `spawn_provider_session()` used (cwd `cd`, self-deleting script,
  `open`/terminal-emulator/`cmd start` dispatch) - just running `tmux
  attach -t <session_id>` instead of the provider CLI directly.
- `send_message(session_id, text)` — the delivery primitive for every
  message, including the first: `tmux send-keys -t <session_id> -l -- <text>`
  followed by a separate `tmux send-keys -t <session_id> Enter`. This
  injects keystrokes into the *same still-running* process exactly as a
  human typing into the attached window would — indistinguishable from
  the CLI's own perspective.
- `switch_model(session_id, model)` — thin wrapper around `send_message()`
  sending the provider CLI's own `/model <name>` slash command. Confirmed
  against the installed `claude` 2.1.220 binary's own usage string
  (`Usage: /model <name>. Available: ..., default, or a full model ID.`)
  that this takes a direct argument, not just the interactive picker — not
  yet live-verified end to end under tmux injection (see "Not yet
  verified" below).

`chat_loop()` now tracks `session_created` (has `create_session()` +
`attach_terminal()` run yet) and `active_model` (which model the live
session is currently running as) instead of the old
`session_established`/`resume` pair. Per message: if the session doesn't
exist yet, create + attach it; if the routed model differs from
`active_model`, send a `/model` switch first; then `send_message()` the
actual text. Because delivery is now a synchronous call from `chat_loop()`'s
own process rather than an external spawn, and the loop is single-threaded,
there is only ever one live provider process and messages are delivered
strictly in order — the `--resume` race is gone structurally, not
mitigated.

**New hard dependency: `tmux`.** Not a new Python package (the "stdlib-only
CLI layer" rule is unaffected) but a new external CLI, like `claude`/`codex`
themselves. Chosen over `screen` (already present on this machine, no
install needed) because tmux gives each pane a real pty and has a cleaner,
less fragile scripting interface (`send-keys -l`, `new-session -d`) than
`screen -X stuff`. `terminal.tmux_available()` is checked once in `main()`
before `chat_loop()` starts, printing a clear "install tmux" message and
exiting rather than letting `create_session()` raise a raw
`FileNotFoundError` mid-session.

**Known limitation, explicitly not solved by this design**: switching
*provider* (not just model) mid-session doesn't work. `/model` only
operates within a single already-running CLI's own session; there is no
equivalent for jumping from a live `claude` process to `codex`
mid-conversation — that would need an entirely different tmux session
tied to a different CLI. Moot today since `TIER_MODELS` maps every tier to
`claude` (see "Known limitations and deferred work" below and
`docs/rough-edges.md`'s Codex-calibration note), but would need solving
before any Codex tier lands.

**Live-tested and fixed same-day (2026-07-31): `/model <name>` doesn't
switch immediately.** First live end-to-end test (real messages, tier
change mid-run) surfaced this directly: sending `/model sonnet` + `Enter`
opens a "Switch model?" confirmation dialog ("1. Yes, switch to Sonnet 5 /
2. No, go back"), not a completed switch. Because option 1 is the default
selection, the *next* `send_message()` call's `Enter` keypress ends up
accepting that dialog by coincidence — so the model genuinely does switch
— but the real message text typed just before it lands in the dialog, not
a text box, and is silently lost. Confirmed via `tmux capture-pane`
against a real session before and after the fix. `switch_model()` now
sends a second bare `Enter` after `/model <name>` (with a
`_MODEL_SWITCH_CONFIRM_DELAY_S = 0.5`s pause first, a live-confirmed-
necessary but not stress-tested guess at how long the dialog takes to
render) to accept the dialog before returning control to the caller.
Verified against a real session post-fix: model switch + the actual
question both land correctly. `tiers.py`'s stored model strings (`haiku`/
`sonnet`/`opus`) are confirmed accepted as-is by `/model` in this same
test — no longer an open question.

## Non-blocking spawn, per-message loop restored (2026-07-31, third same-day revision, superseded - see "Persistent tmux session" above)

**Supersedes "One spawn per run" below**, which itself superseded the
original "Spawn-per-message pivot" further down - three revisions to this
same design landed in one session, each one driven by actually using the
previous version, not by further planning in the abstract.

**What went wrong with "one spawn per run"**: it fixed the "new window per
message" friction by spawning once and then having `chat_loop()` itself
return - but that made the *real* problem (spawning blocked until the
spawned session fully exited) worse, not better, since now the entire
`llm-chat` process sat there unresponsive for exactly as long as the one
session stayed open. The user hit this twice, once badly enough to need
Ctrl-C to escape a stuck wait - even after separately confirming, the
first time it came up, that blocking was the design they wanted. Living
with it a second time changed that answer.

**The fix**: `terminal.spawn_provider_session()` is now non-blocking - it
launches the terminal and returns as soon as the launch itself succeeds,
never waiting for the spawned session to exit (see `terminal.py`'s own
module docstring for the mechanism: the wrapper script now deletes itself
as its last line, since this module no longer learns when it's safe to
clean up from a poll loop that no longer exists). `chat_loop()` goes back
to looping and re-prompting immediately after every spawn - the
`session_established`/`resume` bookkeeping that "One spawn per run"
deleted is back, since there can be more than one spawn per run again.

**A real, accepted tradeoff**: because spawning doesn't wait, a second
message can now route and spawn *while an earlier spawned session is
still being created*. If that second message needs `--resume` on the same
`session_id`, and the first spawn's `claude`/`codex` process hasn't
actually finished registering that session yet, the `--resume` call could
race it - untested territory (confirmed by the user as an accepted
tradeoff, not overlooked: keeping shared conversation history across
messages was judged more valuable than eliminating this race by giving
every spawn its own independent `session_id` instead).

## One spawn per run, not one per message (2026-07-31, same-day follow-up, superseded - see "Non-blocking spawn" above)

**Supersedes the per-message-spawn design in "Spawn-per-message pivot"
below.** Landing spawn-per-message and actually using it live surfaced two
problems in the same session: (1) the spawned terminal launched in the
user's home directory instead of the repo `llm-chat` was run from (fixed
separately - see `terminal.py`'s `_spawn_macos()`, an explicit `cd` into
the caller's `cwd`), and (2) reclassifying and opening a brand-new terminal
window for *every single message* meant a full exit-and-return cycle even
for an ordinary follow-up on the same task - real friction, not a
hypothetical one, confirmed by using it.

**The fix, as originally shipped (now itself superseded - see above)**:
`chat_loop()` spawned exactly once per run, deleting the
`session_established`/`resume` bookkeeping entirely on the theory that
there'd never be a second spawn in the same run. That held for less than
a day: making the block itself the thing worth fixing (see "Non-blocking
spawn" above) meant a second spawn per run became possible again, and
`session_established`/`resume` came back with it. The rest of this
section's reasoning (once real, now historical) follows.

Originally: The first message classifies and spawns a terminal as before;
once that spawn call succeeds, `chat_loop()` returns immediately instead
of prompting for another message. Everything after that - follow-ups,
exploring, switching tiers via the spawned session's own `/model` -
happens natively inside that one already-open terminal, never through
another round trip to `llm-chat`'s own prompt. A genuinely new top-level
request needs a fresh `llm-chat` run.

**This "honest framing" paragraph described the one-spawn-per-run design
and is now itself historical** - per-message classification is back (see
"Non-blocking spawn" above), so `llm-chat`'s "every message independently"
claim in the "Architectural decision" section below is accurate again at
face value, not just at the run-boundary level this paragraph argued for.
Kept for history since it was real reasoning at the time, not because it
still describes current behavior.

## Spawn-per-message pivot (2026-07-31, superseded same-day - see "One spawn per run" above)

`chat_loop()` no longer calls `route_and_run()` or wires up
`tui.StreamRenderer` — full design at
`~/.claude/plans/what-s-our-next-goal-jazzy-tome.md` (this machine/user's
plans directory, not in-repo), summarized in this repo's `CLAUDE.md`.
**The per-message part of this section is superseded** (see above) - the
mechanics below (route() classifies only, terminal.spawn_provider_session()
opens the terminal via a disposable wrapper-script + sentinel-file poll
loop) are all still accurate and current, but the original code sample
here showed a `resume=session_established` kwarg that no longer exists:
`chat_loop()` now spawns at most once per run, always establishing
(`--session-id`), never resuming, so that bookkeeping was deleted rather
than kept dormant. Flow per run: `route()` classifies only (no provider
call in-process), `tui.header(decision)` prints the routing decision, then
`terminal.spawn_provider_session(decision.provider, decision.model,
session_id, message)` opens a real terminal window running that CLI call
with an inherited TTY and blocks until it exits — see `terminal.py`'s own
module docstring for the disposable wrapper-script + sentinel-file
mechanism this uses to detect "the spawned process exited" despite
`open -a Terminal`/equivalents not blocking themselves.

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

**Superseded by "Persistent tmux session" above** for `chat_loop()`
specifically: there is now exactly one provider CLI process per run
(created via `terminal.create_session()`, never re-spawned), so the
`--session-id`/`--resume` distinction described below no longer applies to
`chat_loop()` at all — `/model` switching, once rejected below as "a raw
PTY takeover ... undocumented, more fragile", is exactly what
`terminal.switch_model()` now does, just via `tmux send-keys` rather than a
hand-rolled PTY. The mechanism described in this section
(`route_and_run() -> provider.invoke(..., session_id=...)`,
`_established_sessions`) is still real and still exercised by `cli.py`'s
one-shot `route_and_run()` path, which can `--resume` a `session_id` passed
in from outside — `chat_loop()` just doesn't use that path anymore.

`TaskRequest.session_id` is generated once per `chat_loop()` run (not per
message) — originally threaded through `route_and_run() ->
provider.invoke(..., session_id=...)`, now threaded into
`terminal.create_session()`/`send_message()` instead. Every message in a
session shares one `session_id`, so history continues even as different
messages route to different Claude tiers/models — lets `llm-chat` stay a
thin router in front of real Claude Code functionality (tools, system
prompt, CLAUDE.md/hooks) instead of reimplementing an interface that mimics
it. (Rejected first: a bespoke reimplemented chat interface.)

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
everything else in this pass. Back to a plain `tui.prompt()`. **Partly
superseded 2026-08-01**: the underlying `input()` line-editing gap (no real
paste awareness, no arrow-key editing) is now fixed by moving to
`prompt_toolkit` — see "Paste-safe multi-line input" above — but the
boxed-frame *visual* idea specifically is still not attempted; the two are
separable and only the editing-capability half was ever the blocker for
paste/arrow-key support.

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

**The current answer, as of the same-day "Persistent tmux session" revision** (see near the top of
this doc): `llm-chat` still classifies **every** message independently — that part of this section
remains non-negotiable and true — but no longer spawns a new terminal/process to deliver each one.
One tmux-backed session is created per run and every message, including the first, is injected into
it via `send-keys`. This actually softens the "structurally forecloses interactive UX" framing two
paragraphs up: `llm-chat` and the attached terminal now cooperatively share one live session rather
than separate processes racing to own the TTY across spawns — the user can still type directly into
the attached window between `llm-chat`-driven messages, same session either way. The "Non-blocking
spawn" and "one spawn per run" paragraphs below describe superseded spawn-per-message mechanics, kept
for history.

**Superseded same-day, kept for history**: in between the original per-message design and the
current one, `llm-chat` briefly spawned exactly one terminal per run and then got out of the way
entirely — see "One spawn per run" near the top of this doc. That fixed the "new window per
message" complaint but made the real problem (spawning *blocking* until the session exited) worse,
not better, since it meant the whole `llm-chat` process sat unresponsive for as long as that one
session stayed open. Reverted the same day once that became clear from actually using it.

**Superseded earlier, kept for history**: the version below this one had `llm-chat` spawn a
terminal for **every** message, not just the first — real native UX per message, but at the cost of
one terminal window per line of dialogue and a full exit/return cycle between them (the *blocking*
part of that cost is what's fixed now; the "one terminal window per message" part was judged, after
living with the alternative, to be the lesser problem).

*Superseded, kept for history*: the original plan was narrower — basic interactivity plus
slash-command dispatch (`/plan`, `/clear`, etc., all since removed) as `llm-chat`'s own affordances,
with a future `/handoff` command reserved for conversations needing more intense/precise
communication (complex refactoring, high-stakes work, exploration needing iteration) to save the
session and launch a real `claude --resume` session. That framing generalized into spawn-per-message
by default instead of an opt-in escalation — same conclusion (don't rebuild Claude Code's interactive
layer), different delivery mechanism.

This isn't a limitation to work around — it's a design decision with real tradeoffs. Don't revisit
it.
