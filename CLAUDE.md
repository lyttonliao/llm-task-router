# llm-task-router

Given a task description, classify it and route it to the cheapest model tier
that clears a quality floor for that task category, then actually run it.
The live-routing counterpart to `llm-eval-harness`, which benchmarks
prompt/model quality offline to calibrate the tiers this router picks from.

## Why it's built this way

- **Subscription CLIs, not API keys.** Every provider call shells out to a
  headless CLI (`claude -p`, `codex exec`) instead of an API/SDK, same
  cost-avoidance rationale as `llm-eval-harness`'s `claude_cli.py`: no
  separate per-token billing, runs on existing subscriptions.
- **Zero third-party dependencies** — stdlib only (`dataclasses`, `argparse`,
  `subprocess`, `json`), matching `llm-eval-harness`.
- **Independent from `llm-eval-harness`.** This repo has its own dataclasses
  and its own provider adapters rather than importing the eval harness as a
  package. The two projects interact conceptually (the harness's benchmark
  runs are what should calibrate the tier map in `tiers.py`), not through
  shared code — a live-routing runtime and an offline benchmark runner are
  different enough consumption patterns that coupling them would be the wrong
  kind of DRY.

## Architecture

```
llm_task_router/
  schema.py       - TaskRequest, RouteDecision, ProviderResult dataclasses
  providers/
    base.py       - Provider protocol (invoke(prompt, model) -> ProviderResult)
    claude_cli.py - subprocess wrapper around `claude -p`
    codex_cli.py  - subprocess wrapper around `codex exec`, verified end to
                    end against a real install and a real authenticated call
                    (see rough edges below for what's still unverified)
  classifier.py   - tier-1 heuristic rule table (type x domain grid)
  tiers.py        - tier name -> concrete (provider, model) mapping
  known_models.py - static known-model table, display-only, NOT used for
                    routing (see "llm-chat" below)
  router.py       - route() classifies + resolves a tier; route_and_run()
                    also invokes the provider
  cli.py          - `llm-route route <description> --type ... --domain ...`
                    (installed console script; `python -m llm_task_router
                    route ...` still works identically, cli.py is unchanged
                    either way - see "Installed CLI entrypoint" below)
  repl.py         - `llm-chat`, interactive terminal client: authenticates
                    each provider at startup, then routes each message
                    independently via route_and_run() (stateless,
                    single-shot - see "llm-chat" below)
```

## The classifier is a three-tier cascade; only tier 1 exists

The full design (from the router planning thread) is a confidence cascade:
1. **Heuristic rule table** (`classifier.py`) - cheapest, fires first. This is
   the only tier implemented so far.
2. A **trained model**, cold-started from `llm-eval-harness` golden-set
   labels, for cases the heuristic doesn't confidently cover. Not built.
3. A **cheap LLM call** for the remaining ambiguity band. Not built.

Default-to-escalate under uncertainty, not default-to-cheap - an
underrouted, silently-wrong answer is worse than an overrouted, correct-but-
pricier one, because the former is undetectable without dedicated auditing.

Adding tier 2 or 3 means giving `classifier.classify()` a confidence signal
and having `router.route()` fall through to the next tier when it's low -
don't restructure `route()`'s shape to do this, extend it.

## Adding a provider

See the `add-provider` skill in this repo for the full real-verification
workflow (install, read the actual subcommand's `--help`, check auth, write
the adapter + mocked tests, one real authenticated call, register it). Short
version:

1. Add `llm_task_router/providers/<name>.py` with an `invoke(prompt, model)
   -> ProviderResult` function, following `claude_cli.py`'s shape - including
   a `check_auth() -> tuple[bool, str]` that `invoke()` calls first (see
   "Auth pre-flight check" below). Every existing provider has one; a new one
   without it is a regression, not just an omission.
2. Register it in `router.PROVIDERS` as a **module**, not a pre-grabbed
   function - `PROVIDERS = {"name": module}` then `.invoke(...)` at call
   time, not `{"name": module.invoke}`. The latter early-binds the reference
   at import time and silently defeats `patch("...module.invoke")` in tests -
   this already happened once during `router.py`'s own first draft.
3. Only add entries to `tiers.TIER_MODELS` pointing at it once you have real
   quality-floor calibration data for that provider's models - run
   `llm-eval-harness`'s `calibrate-tier` skill (sibling repo, `../llm-eval-harness`)
   to get it. A tier mapping is only as good as the quality floor behind it.
   As of 2026-07-23 that repo has full judged data for every Codex model this
   account can reach (`gpt-5.4-mini`, `gpt-5.6-luna`, `gpt-5.6-terra`,
   `gpt-5.5`) against Claude's haiku/sonnet/opus on `bug_triage`/`v1_naive`
   (see its CLAUDE.md, "Router tier calibration status") - and the verdict is
   **none of them clear haiku's cheap-tier floor yet**. Don't add a Codex
   entry to `TIER_MODELS` off that data; re-run calibration if a stronger
   Codex model becomes accessible on this account (`gpt-5.6-sol` and several
   others 400 on the current login - see that section for the full list).

## Installed CLI entrypoint

`pyproject.toml` declares `[project.scripts] llm-route = "llm_task_router.cli:main"`
and `[tool.uv] package = true`-equivalent build config (`[build-system]` +
`[tool.uv.build-backend] module-root = ""`, needed because the package lives
at the repo root, not under `src/`). `uv sync` builds and installs it, after
which `uv run llm-route route "<description>" --dry-run` works as a real
command, not just `python -m llm_task_router`. `cli.py`'s `main()` didn't
change shape - only `ArgumentParser(prog=...)` was updated from the old
`"python -m llm_task_router"` to `"llm-route"` so `--help` output matches the
way it's actually invoked now. Both invocation styles keep working
side by side; there's no reason to remove the `-m` path.

## llm-chat: interactive terminal client

`llm-chat` (`repl.py`, registered the same way `llm-route` is - see
"Installed CLI entrypoint") is a real interactive session: authenticate once
per provider, then type messages in a loop and get routed responses printed
back verbatim with a model-used indicator, the terminal-native counterpart
to `llm-route`'s one-shot invocation. Built specifically so engineers
without an API budget can still get a live-routing chat experience off
their existing Claude/ChatGPT subscriptions - no code path here touches
`ANTHROPIC_API_KEY`/`OPENAI_API_KEY`, matching this repo's "subscription
CLIs, not API keys" rationale.

**Stateless, single-shot per message, deliberately, for v1.** Every message
in a session routes independently through `route_and_run()` completely
unmodified - no conversation history is sent to the model, and a session
can land a different message on a different model/tier with zero special
handling. This is a real limitation, not an oversight: both `claude -p` and
`codex exec` already expose session-continuation flags (`--session-id`/
`--continue`/`--resume` and `codex exec resume` respectively) that a v2
could use. The seam for that: `TaskRequest` could gain an optional
`session_id: str | None = None` field (backward-compatible default), and
provider `invoke()` could gain an optional continuation kwarg mapping to
those flags. None of this exists yet - it's a seam, not a partial
implementation, and multi-turn continuity has a real complication worth
solving deliberately rather than bolting on: switching which provider/tier
a message routes to mid-conversation breaks continuity, since each CLI's
session state is local to that CLI.

**Login is always handed off to the provider's own interactive command,
never driven programmatically.** `claude_cli.login()` and `codex_cli.login()`
both shell out with **inherited stdio** (no `capture_output`/`timeout`) to
`claude auth login --claudeai` (the subscription flow, not `--console`) and
`codex login` respectively, so the user completes the real OAuth/device flow
directly in the same terminal - `repl.py` never attempts to complete or
parse that flow itself. This extends the `add-provider` skill's existing
rule ("never attempt to complete an OAuth/browser login flow on their
behalf") to say the *client*, not just the developer working on this repo,
must always defer to the CLI's own login command. Neither `login()`'s exit
code is trusted as proof of success - `repl.py` always re-runs `check_auth()`
afterward as the real source of truth, same discipline `invoke()` already
uses. `codex_cli.login()` accepts an unwired `device_auth: bool = False`
param for a future headless/SSH login prompt; nothing calls it with `True`
yet.

**`known_models.py` is informational only - never consulted by `route()`/
`route_and_run()`.** It's a hardcoded table of model slugs already confirmed
reachable via this account's calibration history (see "Known rough edges"
below for the Codex slug list), used solely so `repl.py`'s startup summary
can tell the user roughly what's usable given who they authenticated with.
Same staleness risk as the Codex slug list it's drawn from - re-verify with
a real call before trusting it for anything beyond display.

**Named limitation: authenticating Codex alone currently makes zero tiers
routable.** `tiers.TIER_MODELS` maps every tier (`cheap`/`mid`/`flagship`) to
`"claude"` today (see "Known rough edges" - no Codex model has cleared a
tier floor yet), so `repl.startup_auth_check()` can report Codex as
authenticated while `repl.routable_tiers()` still returns an empty routable
set, and `main()` refuses to start the chat loop with a clear message rather
than silently letting every message fail one at a time. This is a direct,
expected consequence of `tiers.py`'s current calibration state, not a bug in
`repl.py`, and not something this feature tries to work around.
`tests/test_repl.py::test_routable_tiers_against_real_tier_models_with_only_codex_authenticated`
is pinned against the real `TIER_MODELS` specifically so it starts failing
(in the good way) the day a Codex tier gets calibrated in - that failure is
the signal this paragraph needs updating, not a regression.

## Auth pre-flight check

Both provider adapters (`claude_cli.py`, `codex_cli.py`) now export a
`check_auth() -> tuple[bool, str]` that `invoke()` calls before doing
anything else - `claude auth status --json` (parses the `loggedIn` key) and
`codex login status` (text-matches for "logged in" / "not logged in")
respectively. An unauthenticated call now short-circuits to a single clear
`ProviderResult(error="auth check failed: ...")` and never reaches the real
model subprocess, instead of falling through to whatever failure shape the
underlying CLI produces on its own (a nonzero exit with a stderr message that
varies by CLI, or - in `llm-eval-harness`'s worst observed case for an
inaccessible Codex account - every case coming back a `parse_error` with a
misleadingly bad-looking aggregate score, see that repo's CLAUDE.md,
"Codex model access is account-dependent"). This exists specifically to catch
that failure mode at the single-call layer, before it can propagate into a
multi-case run that looks like a real (terrible) quality result.

Both adapters' logged-out output shapes are confirmed against real output
(2026-07-26), not inferred - and neither required actually logging this dev
account out of Claude/ChatGPT:

- **Claude**: `env -u ANTHROPIC_API_KEY claude --bare auth status` returns
  `{"loggedIn": false, "authMethod": "none", "apiProvider": "firstParty"}` at
  exit 1. `--bare` mode explicitly skips keychain/OAuth reads (per
  `claude --help`), so this exercises the real "no credentials resolved"
  code path without touching the account's actual stored login.
- **Codex**: `CODEX_HOME=<empty dir> codex login status` returns plain text
  "Not logged in" at exit 1. Pointing `CODEX_HOME` at a directory with no
  stored credentials gets the same effect for Codex that `--bare` gets for
  Claude, without touching the account's actual `~/.codex` login.

Reuse these two techniques for testing this gate in future sessions instead
of reaching for a real logout - a real logout on either CLI requires an
interactive re-auth flow to undo, which isn't worth the risk for a read-only
verification. Both `check_auth()` implementations were already written
defensively before this confirmation (treat anything that doesn't clearly
parse as "logged in" as unauthenticated, not the other way around); the real
output matched what was inferred, so no logic changes were needed, only the
docstrings/tests moving from "assumed shape" to "confirmed shape with a
regression test pinned to the real string."

Adding a new provider should include its own `check_auth()` following this
same shape (see "Adding a provider" below) - don't skip it just because the
provider's own nonzero-exit path already surfaces auth errors eventually;
the point is failing fast and consistently, not just failing.

## Known rough edges

- `providers/codex_cli.py` is verified against a real `codex-cli 0.145.0`
  install and a real authenticated (ChatGPT-account) call - not a placeholder
  anymore. Two things still open: (1) there's no dollar-cost field anywhere
  in `codex exec`'s output (a real run's stderr does print an unstructured
  "tokens used" line followed by a token count, but no dollar figure, and
  converting tokens to cost needs per-model pricing this CLI doesn't expose),
  so `cost_usd`/`duration_ms` stay 0.0/0 placeholders; (2) `--output-last-message`'s
  behavior on a genuine content refusal (as opposed to a hard API error,
  which is covered) is unconfirmed. It's currently unreachable through the router in practice
  anyway since no tier in `tiers.py` routes to it yet.
- Codex has no flag equivalent to claude_cli.py's `--disallowed-tools "*"`
  that fully disables tool/shell use - `--sandbox read-only` is the closest
  analog (can't write files), but the model can still choose to run
  read-only shell commands to gather context before answering. Don't assume
  cost/latency parity between the two provider adapters. Also: which model
  names are valid depends on the auth mode - a ChatGPT-account login rejects
  some names outright (confirmed via a real 400 invalid_request_error), so
  don't hardcode a guessed model name into `tiers.py` without checking it
  against the account that will actually run it. As of 2026-07-23, on the
  account used for development: `gpt-5.4-mini`, `gpt-5.6-luna`,
  `gpt-5.6-terra`, and `gpt-5.5` are reachable; `gpt-5.6-sol`, `gpt-5.3-codex`,
  `gpt-5.1-codex-mini`, `gpt-5.4-nano`, `gpt-5.4`, and `gpt-5.2` all 400. A
  different account/plan may differ - re-probe with a single cheap `codex
  exec` call before trusting either list.
- `tiers.TIER_MODELS`'s Claude entries (haiku/sonnet/opus for cheap/mid/
  flagship) are now backed by real judged benchmark data, not a guess - as of
  2026-07-23 `llm-eval-harness` confirmed a clean, monotonic ladder on
  `bug_triage`/`v1_naive` (haiku 60% fully-correct → sonnet 66.7% → opus
  73.3%, judge coherence flat ~0.84-0.85 across all three). No Codex model
  tested clears haiku's floor yet, so the map stays Claude-only for now - see
  the calibration status section in that repo's CLAUDE.md for the full table
  and which Codex model slugs are even reachable on this account.
- Whether task types like `code_gen`/`refactor` should eventually get real
  tool access (vs. the current single-shot, tools-disabled completion every
  provider call makes) is an open, deliberately deferred question - see the
  cost-guardrail flags in `claude_cli.py`.
- No shadow evaluation, no live dual-routing, no drift auditing - this
  scaffold only exploits the current heuristic grid, it doesn't explore or
  self-correct yet.
