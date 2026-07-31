# llm-task-router

Given a task description, classify it and route it to the cheapest model tier
that clears a quality floor for that task category, then actually run it.
The live-routing counterpart to `llm-eval-harness`, which benchmarks
prompt/model quality offline to calibrate the tiers this router picks from.

## Next step (updated 2026-07-31)

`audit_tier2`/`shadow_report` launchd jobs are live on this dev machine
(`~/Library/LaunchAgents/com.llm-task-router.*.plist`, daily 06:00/06:15,
`DATABASE_URL=postgresql:///llm_task_router`, logs in
`~/Library/Logs/llm-task-router/`) — verified with one manual
`launchctl start` each before trusting the schedule; as of 2026-07-31 the
schedule hasn't yet had its first unattended fire (next one is today's
06:00/06:15). `routing_decisions` has grown from 1→3 real shadow-scored
rows since setup — still short of a meaningful tier 1 vs. tier 2 divergence
read; periodically re-run `llm-route-shadow-report` (or read
`shadow_report.log`) as more accumulate (see `docs/drift-and-shadow.md`).
`is_high_stakes` is separately stuck at 3 labeled rows (needs 6 for
leave-one-out at all) — unlike `task_type` (101 rows), ordinary traffic may
not organically produce enough high-stakes-labeled examples; may need
deliberate seeding rather than passive waiting. This machine instance
doesn't change the portable recipe in `docs/scheduling-audits.md`.

**`llm-chat` plan mode (`/plan`) and `/clear` shipped 2026-07-31** — see
`docs/llm-chat.md` for the real-CLI verification (`ExitPlanMode` errors out
headlessly; the two-turn `--permission-mode plan` → `--resume` flow works
around that) and the known gap it leaves (execute-leg cost isn't logged to
`routing_decisions`).

One thread stays explicitly on hold in favor of letting the audit schedule
accumulate more data first (deferred three times, most recently
2026-07-29): **Codex tier calibration** (re-probe reachable Codex models,
re-run `llm-eval-harness`'s `calibrate-tier` skill) — see
`docs/rough-edges.md`. Also parked: Windows `select()` support,
cross-provider session continuity.

Don't re-derive this by re-reading the whole file — start here, then jump to
the referenced doc only if you need the full backstory.

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
