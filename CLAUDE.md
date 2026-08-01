# llm-task-router

Given a task description, classify it and route it to the cheapest model tier
that clears a quality floor for that task category, then actually run it.
The live-routing counterpart to `llm-eval-harness`, which benchmarks
prompt/model quality offline to calibrate the tiers this router picks from.

## Next step (updated 2026-07-31)

**`llm-chat` is now a pure routing/classification layer** — full design at
`~/.claude/plans/what-s-our-next-goal-jazzy-tome.md` (this machine/user's
plans directory, not in-repo; describes the original per-message-spawn
version, since revised four times - see `docs/llm-chat.md`'s "Persistent
tmux session" section for the full history). `llm-chat` no longer renders
provider output itself (the custom `StreamRenderer`/`tui.py` streaming path
was chasing feature parity with Claude Code's own interactive UI — a
permanent maintenance burden against a target this repo doesn't control).

**Current flow, landed 2026-07-31, mocked-suite-green but NOT yet
live-verified** (see `~/.claude/plans/so-right-now-we-re-scalable-reddy.md`
for the plan this landed from): `chat_loop()` classifies every message via
`route()`, then delivers it into a single persistent `tmux`-backed terminal
session instead of spawning a new terminal/process per message.
`terminal.create_session()` starts the provider CLI detached under tmux
once per run (`tmux new-session -d -s <sid> -- claude --model <model>
--session-id <sid>`); `terminal.attach_terminal()` opens the one visible
terminal window (`tmux attach -t <sid>`), non-blocking exactly as spawning
was before; every message after, including the first, is delivered via
`terminal.send_message()` (`tmux send-keys -l` for literal text, then a
separate `Enter`) — literal keystroke injection into that same still-running
process, indistinguishable from a human typing. A tier change mid-run sends
the provider CLI's own `/model <name>` first (`terminal.switch_model()`).
Requires `tmux` on PATH, checked via `terminal.tmux_available()` before
`chat_loop()` starts. Only `/exit`/`/quit` survive of the old slash
commands; `/help`, `/clear`, `/plan` are removed outright.

**Why**: the prior "non-blocking spawn, one window per message" design
(three revisions, see below) used `--resume <session_id>` so each new
window's process reloaded the same transcript — but a fast second message
could fire its `--resume` before the first spawn's `--session-id` call had
actually finished registering the session, an accepted-but-untested race
that looked exactly like "every message opens a new session" once it bit.
The tmux redesign removes the race structurally: there is only ever one
provider process per run, so there's nothing left to race.

**Four revisions landed in one day, each driven by actually using the
previous one, not further planning**: (1) spawn a terminal per message,
blocking until it exits; (2) spawn once per run, then get out of the way
entirely (fixed the "window per message" complaint, made blocking worse);
(3) non-blocking spawn, back to one spawn per message (fixed blocking,
reintroduced the `--resume` race above); (4) **current** - one persistent
tmux session per run, no more spawning after the first message, message
delivery via `send-keys` instead of process respawn. See
`docs/llm-chat.md`'s "Persistent tmux session" section for the full
mechanics and the explicit known limitation (provider switching mid-session
still isn't supported - `/model` only works within one already-running
CLI's session).

**Bugs from earlier revisions, still real history**: the spawned terminal
originally launched in the user's home directory instead of the repo
`llm-chat` was run from — `open`/equivalents don't inherit the caller's
`cwd`, fixed with an explicit `cd` in the wrapper script (this fix carried
forward unchanged into `attach_terminal()`). A live reminder that
`terminal.py`'s mocked tests only ever verify command construction, never
what actually happens once a real shell runs — see that module's own
docstring.

Commits so far (this list will grow once the tmux redesign lands its own
commits - not yet committed as of writing this section): word-wrap rewrite
(`75b33d3`), `terminal.py` spawn primitive (`e2123c1`), `chat_loop()`
wiring + slash-command removal (`0c176fb`), docs (`36751ac`), cwd fix
(`1f2a06f`), one-spawn-per-run redesign (`a7d0740`) then reverted same-day
for non-blocking spawn (`d2ea35e`). `StreamRenderer`/`repl.format_response()`
are provably unreferenced by any application code (confirmed by grep —
`route_and_run()` survives only via `cli.py`'s one-shot `llm-route`
command) but are deliberately **not** deleted yet, per the referenced
plan's "incrementally, not upfront" removal policy. macOS-verified end to
end for the prior three revisions; Linux/Windows terminal-spawning remains
unverified against real installs, same status as this repo's Windows
`select()` gap — see `docs/rough-edges.md`.

**Not done yet, and the next concrete step**: live-verify the tmux redesign
(the mocked suite is green but has never proven the real thing works,
same caveat every prior revision here needed) - `brew install tmux`, run
`llm-chat`, confirm two same-tier messages land as sequential turns in one
window and a tier change actually switches the model via `/model` before
the next message. Also watch for whether `/model` needs a settling beat
before the next `send-keys` call, and whether `tiers.py`'s model strings
are accepted as-is by `/model` (only confirmed it takes a direct argument
at all, via the binary's own usage string, not that it behaves correctly
under tmux injection). Remaining work beyond that is the pre-existing
parked threads below (Codex tier calibration, Windows/Linux verification)
plus whatever surfaces from continued daily use.

Separately still true and unaffected by the above: `audit_tier2`/
`shadow_report` launchd jobs are live on this dev machine
(`~/Library/LaunchAgents/com.llm-task-router.*.plist`, daily 06:00/06:15) and
have had their first unattended fire (06:14/06:25 logs, 2026-07-31).
`routing_decisions` is at 15 rows (13 shadow-scored) — enough for an actual
divergence read now: `llm-route-shadow-report` shows 30.8% divergence
(1 escalated / 3 de-escalated, all task_type-driven) and, more importantly,
that `cheap` only costs ~1.8x less than `mid` per call ($0.1872 vs $0.3433
avg) — real signal that per-call fixed overhead (Claude Code's system
prompt + full tool schemas + global `CLAUDE.md`, sent on every call
regardless of tier) is compressing the tier ladder's cost savings; see
`docs/rough-edges.md` for the diagnosis, not yet acted on. `is_high_stakes`
is separately stuck at 4 labeled rows (needs 6) — passive traffic isn't
producing these; needs deliberate seeding. `audit_tier2.py` has also flagged
its first real suspect row (id=102, stored label disagrees with neighbors at
80%) with no command yet to act on it beyond manual SQL.

One thread stays explicitly on hold in favor of letting the audit schedule
accumulate more data first (deferred four times, most recently 2026-07-31):
**Codex tier calibration** — see `docs/rough-edges.md`. Also parked:
Windows `select()` support, cross-provider session continuity.

Don't re-derive this by re-reading the whole file — start here, then jump to
the referenced doc or plan only if you need the full backstory.

## Why it's built this way

- **Subscription CLIs, not API keys.** Every provider call shells out to a
  headless CLI (`claude -p`, `codex exec`) instead of an API/SDK — same
  cost-avoidance rationale as `llm-eval-harness`: no per-token billing, runs
  on existing subscriptions.
- **Zero third-party dependencies for the CLI/provider-adapter layer** —
  stdlib only (`dataclasses`, `argparse`, `subprocess`, `json`), matching
  `llm-eval-harness`. **No longer true repo-wide**: `sentence-transformers`
  and `psycopg`/`pgvector` are real, permanent runtime dependencies of the
  tier-2 classifier (`tier2_classifier.py`/`vector_store.py`/
  `decision_log.py`) — a deliberate, discussed exception. The rule still
  holds everywhere else.
- **Independent from `llm-eval-harness`.** Own dataclasses and provider
  adapters rather than importing the eval harness as a package — the two
  interact conceptually (the harness's benchmark runs calibrate `tiers.py`),
  not through shared code, since a live-routing runtime and an offline
  benchmark runner are different enough consumption patterns that coupling
  them would be the wrong kind of DRY.

## Architecture

```
llm_task_router/
  schema.py       - TaskRequest, RouteDecision, ProviderResult dataclasses
  providers/
    base.py       - Provider protocol (invoke(prompt, model) -> ProviderResult)
    claude_cli.py - subprocess wrapper around `claude -p`
    codex_cli.py  - subprocess wrapper around `codex exec`, verified end to
                    end against a real install and a real authenticated call
                    (see docs/rough-edges.md for what's still unverified)
  classifier.py   - tier-1 heuristic rule table (type x domain grid)
  embeddings.py   - tier-2: local sentence-transformers embedding wrapper
  vector_store.py - tier-2: pgvector-backed routing_examples store
                    (nearest_neighbors/insert_example/all_labeled_examples) -
                    one of two modules allowed to contain SQL, see
                    decision_log.py below
  tier2_classifier.py - tier-2 orchestration: NN lookup + cheap-LLM
                    fallback-with-write-back (see docs/classifier.md)
  decision_log.py - drift auditing + shadow evaluation: the second (and,
                    for now, final) module allowed to contain SQL -
                    routing_decisions log, one row per route() call, each
                    carrying both the real decision and a free tier-1-only
                    shadow decision (see docs/drift-and-shadow.md)
  db/schema.sql   - routing_examples + routing_decisions DDL, applied
                    manually via psql
  tiers.py        - tier name -> concrete (provider, model) mapping
  known_models.py - static known-model table, display-only, NOT used for
                    routing (see docs/llm-chat.md)
  router.py       - route() classifies + resolves a tier; route_and_run()
                    also invokes the provider
  cli.py          - `llm-route route <description> --type ... --domain ...`
                    (installed console script; `python -m llm_task_router
                    route ...` still works identically)
  repl.py         - `llm-chat`, interactive terminal client: authenticates
                    each provider at startup, then routes each message
                    independently via route() and delivers it into one
                    persistent tmux-backed terminal session (see
                    docs/llm-chat.md)
  terminal.py     - platform-dispatch terminal spawn + tmux session
                    control: create_session/attach_terminal/send_message/
                    switch_model (see docs/llm-chat.md)
  tui.py          - stdlib-only ANSI styling, no longer wired into
                    chat_loop()'s success path (see docs/llm-chat.md)
```

## Where the rest of this lives

This file stays a short index on purpose — the detail (design history,
decisions overturned along the way, real-world gotchas) lives in `docs/`,
one topic per file:

- **`docs/classifier.md`** — tier-1 heuristic grid + tier-2
  continuous-learning classifier (embeddings, pgvector NN lookup,
  `AGREEMENT_THRESHOLD`, LLM-fallback write-back).
- **`docs/drift-and-shadow.md`** — `routing_decisions` audit logging,
  `audit_tier2.py`'s leave-one-out re-validation, and
  `report_shadow_divergence.py`'s tier-1-vs-tier-2 live comparison.
- **`docs/scheduling-audits.md`** — portable cron/launchd/Task Scheduler
  recipes for the two audit scripts above (also see the `schedule-audits`
  skill).
- **`docs/cli-entrypoints.md`** — the four installed console scripts and
  running them from any directory via `uv tool install` (also see the
  `install-cli` skill).
- **`docs/llm-chat.md`** — the interactive terminal client: session
  continuity, streaming transport, `tui.py` styling, known limitations.
- **`docs/auth-and-providers.md`** — the `check_auth()` pre-flight pattern
  and the checklist for adding a new provider (also see the `add-provider`
  skill).
- **`docs/rough-edges.md`** — known gaps and unverified behavior, grouped
  by module.

Adding a provider or touching scheduling/auth should start at the matching
skill (`add-provider`, `schedule-audits`, `install-cli`) before the doc —
skills carry the verification workflow, the docs carry the "why."
