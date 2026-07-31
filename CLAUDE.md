# llm-task-router

Given a task description, classify it and route it to the cheapest model tier
that clears a quality floor for that task category, then actually run it.
The live-routing counterpart to `llm-eval-harness`, which benchmarks
prompt/model quality offline to calibrate the tiers this router picks from.

## Next step (updated 2026-07-31)

**`llm-chat` is being rearchitected into a pure routing/classification layer** —
full design at `~/.claude/plans/what-s-our-next-goal-jazzy-tome.md` (this
machine/user's plans directory, not in-repo). Short version: `llm-chat` stops
trying to render provider output itself (the custom `StreamRenderer`/`tui.py`
streaming path was chasing feature parity with Claude Code's own interactive
UI — menus, inline diffs, arrow-key history — a permanent maintenance burden
against a target this repo doesn't control). New flow: classify the message,
print the routing decision, spawn a **real native terminal** running
`claude --model <tier's model> --session-id <sid> "<message>"` with a real
inherited TTY, let the user drive that session with full native
functionality, and return to `llm-chat`'s prompt when they exit. Per-message
routing is unchanged and non-negotiable (see `docs/llm-chat.md`'s
"Architectural decision" section and this project's memory — don't relitigate
why full-interactive-UX-in-`llm-chat` was rejected, it's a structural
incompatibility, not an oversight). Existing `StreamRenderer`/streaming code
is deliberately **not** being deleted as part of this pivot — it stays until
the spawn model is verified end to end and confirmed unreferenced by an
actual grep, not assumed.

Two of the plan's steps are done: the word-wrap rewrite is committed
(`75b33d3`), and `llm_task_router/terminal.py` — the platform-dispatch spawn
primitive (`spawn_provider_session()`), with a disposable wrapper-script +
sentinel-file poll loop so it can block until the spawned CLI exits despite
`open -a Terminal`/equivalents not blocking themselves — is committed
(`e2123c1`), covered by mocked tests, macOS-verified only (Linux/Windows
dispatch is written but unverified against real installs, same unverified
status as this repo's Windows `select()` gap — see `docs/rough-edges.md`).
**Not done yet, and the next concrete step**: `terminal.py` is not wired into
`repl.py:chat_loop()` — no real user message reaches it. That wiring
(rewriting `chat_loop()`'s control flow to classify → print decision → call
`spawn_provider_session()` → loop, plus what happens to `/model`, `/plan`,
and a possible `/handoff` command under the new flow — see the referenced
plan's steps 2-4) is its own pass, deliberately not bundled with landing the
spawn primitive itself. A real end-to-end smoke test (type a message, watch
a terminal actually open) can't happen until that wiring exists.

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
                    independently via route_and_run() (see docs/llm-chat.md)
  tui.py          - stdlib-only ANSI styling for repl.py's live-streaming
                    terminal output
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
