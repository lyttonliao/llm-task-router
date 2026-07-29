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
- **Zero third-party dependencies for the CLI/provider-adapter layer** —
  stdlib only (`dataclasses`, `argparse`, `subprocess`, `json`), matching
  `llm-eval-harness`. **No longer true repo-wide**, as of the tier-2
  continuous-learning classifier (2026-07-27, see "Tier 2" below):
  `sentence-transformers` and `psycopg`/`pgvector` are real, permanent
  runtime dependencies of `tier2_classifier.py`/`vector_store.py`/
  `decision_log.py`. Deliberate, discussed choice — building the actual
  long-term product was judged more valuable than staying dependency-free.
  The rule still holds everywhere else; don't read this as license to add
  dependencies elsewhere without the same deliberate discussion.
- **Independent from `llm-eval-harness`.** This repo has its own dataclasses
  and provider adapters rather than importing the eval harness as a package.
  The two projects interact conceptually (the harness's benchmark runs
  calibrate the tier map in `tiers.py`), not through shared code — a
  live-routing runtime and an offline benchmark runner are different enough
  consumption patterns that coupling them would be the wrong kind of DRY.

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
  embeddings.py   - tier-2: local sentence-transformers embedding wrapper
  vector_store.py - tier-2: pgvector-backed routing_examples store
                    (nearest_neighbors/insert_example/all_labeled_examples) -
                    one of two modules allowed to contain SQL, see
                    decision_log.py below
  tier2_classifier.py - tier-2 orchestration: NN lookup + cheap-LLM
                    fallback-with-write-back (see "Tier 2" below)
  decision_log.py - drift auditing + shadow evaluation: the second (and,
                    for now, final) module allowed to contain SQL -
                    routing_decisions log, one row per route() call, each
                    carrying both the real decision and a free tier-1-only
                    shadow decision (see "Drift auditing" and "Shadow
                    evaluation" below)
  db/schema.sql   - routing_examples + routing_decisions DDL, applied
                    manually via psql
  tiers.py        - tier name -> concrete (provider, model) mapping
  known_models.py - static known-model table, display-only, NOT used for
                    routing (see "llm-chat" below)
  router.py       - route() classifies + resolves a tier; route_and_run()
                    also invokes the provider
  cli.py          - `llm-route route <description> --type ... --domain ...`
                    (installed console script; `python -m llm_task_router
                    route ...` still works identically)
  repl.py         - `llm-chat`, interactive terminal client: authenticates
                    each provider at startup, then routes each message
                    independently via route_and_run() (see "llm-chat" below)
  tui.py          - stdlib-only ANSI styling for repl.py's live-streaming
                    terminal output
```

## The classifier: tier-1 heuristic + tier-2 continuous-learning cascade

Originally scoped as three tiers (heuristic → a trained model cold-started
from `llm-eval-harness` golden labels → cheap-LLM fallback). Design review on
2026-07-27 collapsed the last two into one mechanism instead of building the
intermediate static model — see "Tier 2" below. `classifier.py`'s heuristic
grid remains tier 1, unchanged, the free/zero-latency first pass; expected to
be deprecated later (not yet) once tier 2 earns enough coverage to be trusted
alone.

**Escalation to flagship is precision-first, not recall-first** (reversed
2026-07-27 from an original "escalate under uncertainty" default).
`router.route()` requires both a grid cell mapped to `H` and — unless the
caller explicitly provided `task_type` (trusted as a deliberate override) — a
genuine high-stakes signal in the description itself
(`classifier.has_high_stakes_signal()`; see `IMPACT_KEYWORDS`'s docstring for
the deliberately narrow production/security/compliance/irreversibility/scale
vocabulary). A shape-only match (`architecture`'s `TYPE_KEYWORDS` includes
bare "design"/"strategy", which fire on a trivial UI question as readily as a
real one) no longer earns flagship alone. An inferred `H` without keyword
corroboration falls through to tier 2's `resolve_high_stakes()` (below)
rather than capping unconditionally at `mid` — a real escalation path for
genuinely hard tasks that don't use recognized vocabulary (e.g. "migrate the
k8s cluster to a new region"), confirmed against a real run.

**No-signal fallback**: a fully-unresolved description used to inherit
`classify_description()`'s `task_type="architecture"` safety placeholder and
silently escalate via architecture's uniform-`H` row (a real incident: "reply
with exactly the word: pong" routed to opus, ~$0.18 to repeat one word). Both
axes fully unresolved now hedges to `mid` instead of consulting the grid
(`tests/test_router.py::test_route_hedges_to_mid_when_no_keyword_signal_and_tier3_is_unavailable`
pins the fallback-unavailable case).

**Tier 3 (`_classify_via_llm()`)** handles that no-signal band: one stripped
haiku call judges the description's actual difficulty/consequence/ambiguity
directly, rather than a keyword list (tried, rejected — doesn't generalize)
or embeddings (rejected at the time on `llm-eval-harness`'s prior finding
that they don't cleanly separate this kind of soft-semantic distinction at
small scale — tier 2 below revisited that finding for a different question
with a different result). Reuses `claude_cli.invoke(disable_tools=True,
system_prompt=...)`, landing in `eval_harness`'s ~$0.003-0.005/call bracket
instead of `llm-chat`'s ~$0.07-0.30/call. Returns `None` on any failure
(provider error, unparseable text, exception); `route()` falls back to the
`mid` hedge.

## Tier 2: the continuous-learning classifier

Built 2026-07-27. Not a static trained model — embeddings + a pgvector-backed
store of labeled examples (`routing_examples`), falling through to one cheap
LLM call for whatever the store can't confidently answer, which then writes
its own answer back as a new labeled row. `router.route()` calls the same
primitive (`tier2_classifier.py`) against two different questions — task_type
resolution and high-stakes corroboration — not two separate mechanisms.

**Modules**: `embeddings.py` (`embed(text) -> list[float]`, lazy-singleton
`sentence-transformers all-MiniLM-L6-v2`, 384-dim) · `vector_store.py`
(Postgres/pgvector client, `DATABASE_URL`-configured — see "Architecture"
above) · `tier2_classifier.py` (`resolve_task_type()` / `resolve_high_stakes()`,
the orchestration `router.py` calls). `scripts/seed_vector_store.py`
cold-started the store from `llm-eval-harness`'s 98 golden cases (task_type
only — no `is_high_stakes` ground truth exists in that data, so that column
starts empty and fills in only as real `needs_corroboration` requests resolve
through tier 2 over time).

**Write-back is LLM-fallback-only, never on an NN-confident hit** — a
matching neighbor already covers that region of the embedding space;
re-inserting it would add a near-duplicate with no new information. This is
what makes it "continuous learning" rather than a one-time cold start: the
same question never needs a second LLM call.

**`AGREEMENT_THRESHOLD = 0.8` (≥4/5 of `NEIGHBOR_K=5` nearest neighbors must
share a label) came from a real leave-one-out check against the seeded store,
not a guess, and overturned the original design en route:**
1. Raw cosine similarity does **not** cleanly separate same-label from
   different-label neighbors at this data/model scale (same-label median
   ~0.32, cross-label median ~0.29 — heavily overlapping). A hard similarity
   floor (the first draft) would have rejected most true matches — the same
   shape of finding `llm-eval-harness`'s CLAUDE.md already documents for a
   structurally similar problem, confirmed independently a second time.
2. Relative *agreement* among the k neighbors does carry real signal despite
   (1): 72.4% accuracy unconditionally (98 samples, 7 classes, chance ~14%),
   rising monotonically — 78.2% at ≥3/5 (80% coverage), 87.1% at ≥4/5 (32%
   coverage), 100% at 5/5 (8% coverage, too rare alone). Fewer than
   `NEIGHBOR_K` neighbors never counts as confident regardless of agreement
   fraction (true of `is_high_stakes` early on, since it starts empty).

**`needs_corroboration` no longer caps to `mid` unconditionally on a
keyword-negative miss** — it falls through to `resolve_high_stakes()`
(reusing the embedding already computed for task_type when there is one).
`True` escalates to flagship for real; `False` or `None` (tier 2 unavailable)
both still cap to `mid` — tier 2 only ever adds a path to a *correct*
escalation, never removes the safety cap. Both `resolve_task_type()`/
`resolve_high_stakes()` return `None` on any failure (provider error,
unparseable text, exception) — `router.py` treats tier-2-unavailable as "fall
back to heuristic-only," never a crash.

**`resolve_high_stakes()` gained a `source` field, 2026-07-28** —
`HighStakesResolution(is_high_stakes, source)`, symmetric with
`resolve_task_type()`'s `Tier2Resolution(task_type, source)`. Needed by drift
auditing (below): without it, a live decision log couldn't tell whether a
resolution came from the NN vote or the LLM fallback.

## Drift auditing: logging every routing decision

Built 2026-07-28, closing this file's former "no shadow evaluation, no live
dual-routing, no drift auditing" rough edge. Distinct from
`llm-eval-harness`'s `calibrate-tier` skill, which is offline calibration
against a fixed 98-case golden set and never touches live traffic — before
this work, `route()` discarded every decision the instant it was made, so
there was no logged traffic to re-validate `AGREEMENT_THRESHOLD` against or
catch a bad `llm_fallback` write-back with.

**`decision_log.py`** (second SQL module, alongside `vector_store.py`) writes
one row per `route()` call to `routing_decisions`: resolved task_type/domain
and which mechanism produced each (`provided`/`inferred`/`tier2_nn`/
`tier2_llm_fallback`/`fallback` for task_type; `keyword`/`tier2_nn`/
`tier2_llm_fallback`/`unavailable` for high-stakes), plus the final
bias/tier/model/reason. `embedding` is null when the heuristic resolved
everything without ever calling `embeddings.embed()`. Wrapped in
`try/except Exception: pass` in `route()` — a logging failure degrades to "no
audit trail for this call," never a broken route. `tests/conftest.py`
autouse-patches `log_decision` to a no-op for the whole suite; a handful of
`test_router.py` tests override it explicitly to assert logged fields on
representative paths.

**`scripts/audit_tier2.py`** re-runs the same leave-one-out methodology
against the *live* `routing_examples` table (via
`vector_store.all_labeled_examples()`) instead of the frozen seed, printing
the same kind of agreement curve plus a "threshold holds / consider raising"
verdict, and flags `source='llm_fallback'` rows whose neighbors now
confidently disagree with the stored label — printed for manual review only,
never auto-corrected. An opt-in `--judge-flagged` pass gets one cheap second
LLM opinion per flagged row (same call shape as tier 2's own fallback).
Verified against the real local store: 101 `task_type` rows produced a live
curve (73.3%/79.0%/88.2%/100.0%) tracking the original seed-only numbers,
verdict "holds," 0 suspect rows; `is_high_stakes` (3 rows) correctly reported
"not enough rows yet." Manual/periodic, like `seed_vector_store.py` — no cron
wired up yet.

**Gotcha worth restating**: `vector_store.all_labeled_examples()`'s first
draft called `list(row[2])` on the embedding column, assuming an iterable — a
`register_vector()`-registered connection actually returns a
`pgvector.Vector` object there, not iterable; failed with a real `TypeError`
against the real store. Fixed with `Vector.to_list()`. Any future module
reading embeddings back out of Postgres should expect the same and verify
against a real query first.

## Shadow evaluation: live dual-routing without doubling cost

Built 2026-07-28, closing this file's last documented rough edge ("no live
dual-routing — a request never goes to two tiers to compare"). Distinct from
both drift auditing above (which re-validates tier 2's *own* NN/LLM
resolutions against themselves) and `llm-eval-harness`'s offline calibration
(a fixed 98-case golden set) — this compares tier 1 against tier 2 on live
traffic, which is the actual open question blocking tier 1's eventual
deprecation (see "The classifier" above).

**Not a real second model invocation.** The obvious shape of "shadow
evaluation" — actually calling a second model on every request and comparing
outputs — would double real per-call cost for every single routed request,
which contradicts this repo's cost discipline everywhere else (`_classify_via_llm`,
`tier2_classifier`'s LLM fallback, all explicitly reasoned about
dollars-per-call). Instead, `router._shadow_tier1_only_decision()` computes a
second **routing decision**, not a second response: what tier the request
would have gotten if tier 2 had never been consulted at all, using only data
already computed for the real decision (`classification`, the grid, keyword
high-stakes matching) — zero extra embedding calls, zero extra LLM calls,
zero extra Postgres round-trips.

**The no-signal case is a deliberate approximation, not a full replay.** The
real no-signal branch (`_classify_via_llm`, tier 3) makes a genuine LLM call;
re-invoking it a second time just to populate a shadow column would
reintroduce the exact double-cost problem this design avoids. The shadow
reading always hedges the no-signal case straight to `mid` instead — the same
static fallback tier 3 itself uses on its own failure, so this is a
defensible stand-in for "would have escalated," not a fabricated number.

**`route()` logs both, acts on only one.** Every call computes
`shadow_bias`/`shadow_tier`/`shadow_reason` alongside the real decision and
passes both to `decision_log.log_decision()` — the shadow fields never touch
`provider`/`model`/the returned `RouteDecision`, purely an audit-row column.
`decision_log.py` gained `fetch_decisions()` (a `LoggedDecision` NamedTuple
per row) to read the table back in batch, the same one-round-trip-then-
analyze-in-Python shape `vector_store.all_labeled_examples()` already
established.

**`scripts/report_shadow_divergence.py`** reads the live table and reports:
divergence rate (real tier vs. shadow tier), direction (`tiers.TIER_ORDER`
now exists specifically for this — escalated = tier 2 routed more expensively
than tier 1 alone would have; de-escalated = the reverse, e.g. tier 2
correctly resolving a task_type the heuristic couldn't place, avoiding an
unnecessary `mid` hedge), and which `tier2_classifier` axis (task_type
resolution vs. high-stakes corroboration) drove each divergence. Deliberately
does not judge which side was *right* — that needs a ground-truth label or a
judge call this script doesn't make; it only measures how often and in which
direction the two disagree, the input needed for an eventual tier-1
deprecation call.

**Migration note**: `db/schema.sql`'s `CREATE TABLE routing_decisions` now
includes `shadow_bias`/`shadow_tier`/`shadow_reason` (nullable, for
migration compatibility with rows logged before this existed). An existing
live database needs the three commented `ALTER TABLE` statements at the
bottom of that file run manually via `psql` — not applied automatically,
same one-time/not-idempotent discipline the rest of that file already
documents.

## Adding a provider

See the `add-provider` skill for the full real-verification workflow
(install, read `--help`, check auth, write the adapter + mocked tests, one
real authenticated call, register it). Short version:

1. Add `llm_task_router/providers/<name>.py` with `invoke(prompt, model) ->
   ProviderResult` following `claude_cli.py`'s shape, including a
   `check_auth() -> tuple[bool, str]` that `invoke()` calls first (see "Auth
   pre-flight check" below) — every existing provider has one; a new one
   without it is a regression.
2. Register it in `router.PROVIDERS` as a **module**, not a pre-grabbed
   function — `{"name": module}` then `.invoke(...)` at call time, not
   `{"name": module.invoke}`. The latter early-binds at import time and
   silently defeats `patch("...module.invoke")` in tests (already happened
   once in `router.py`'s own first draft).
3. Only add entries to `tiers.TIER_MODELS` once you have real quality-floor
   calibration data (`llm-eval-harness`'s `calibrate-tier` skill). As of
   2026-07-23, no Codex model clears haiku's cheap-tier floor — see "Known
   rough edges" for the current reachable-model list; don't add a Codex entry
   off stale data. Same discipline applies to `classifier.TYPE_DOMAIN_GRID` —
   see `llm-eval-harness/CLAUDE.md`, "Router tier synthesis across all 7
   suites": `code_gen`/`refactor` rows are calibration-derived (uniform `L`),
   the rest is still uncalibrated heuristic. Don't re-derive that table here;
   follow the pointer.

## Installed CLI entrypoint

`pyproject.toml` declares `[project.scripts] llm-route =
"llm_task_router.cli:main"` plus `uv`-package build config (package lives at
repo root, not `src/`, hence `[tool.uv.build-backend] module-root = ""`).
`uv run llm-route route "<description>" --dry-run` works as a real command;
`python -m llm_task_router route ...` still works identically side by side.

## llm-chat: interactive terminal client

`llm-chat` (`repl.py`) is a real interactive session: authenticate each
provider once at startup, then route each typed message independently via
`route_and_run()` — stateless per call, but see session continuity below.
Built so engineers without an API budget get a live-routing chat experience
off existing Claude/ChatGPT subscriptions; no code path touches
`ANTHROPIC_API_KEY`/`OPENAI_API_KEY`.

**Session continuity (2026-07-26).** `TaskRequest.session_id` is generated
once per `chat_loop()` run (not per message) and threaded through
`route_and_run() -> provider.invoke(..., session_id=...)` unconditionally.
Every message in a session shares one `session_id`, so history continues even
as different messages route to different Claude tiers/models — lets
`llm-chat` stay a thin router in front of real Claude Code functionality
(tools, system prompt, CLAUDE.md/hooks) instead of reimplementing an
interface that mimics it. (Rejected first: a bespoke reimplemented chat
interface, and a raw PTY takeover injecting `/model` mid-session —
undocumented, more fragile.)

Confirmed against real `claude` 2.1.220 output (2026-07-26, not guessed — see
verification commands in `providers/claude_cli.py`'s docstring): the *first*
call per session uses `--session-id "$SID"`; every call after must use
`--resume "$SID"` instead — reusing `--session-id` on a second call fails
outright ("Session ID ... is already in use"), but `--resume` correctly
continues history across a `--model` change. `claude_cli.py` tracks which
session ids have had their establishing call in a module-level
`_established_sessions` set so callers just pass the same `session_id` every
time without knowing which flag applies.

**Full functionality, not cost-minimized** — a deliberate choice distinct
from `llm-eval-harness`'s adapter. `providers/claude_cli.py` doesn't strip
the system prompt or disable tools/MCP: real tools, system prompt,
CLAUDE.md/hooks all work, at real per-call cost (~$0.07-0.30/call vs.
eval_harness's ~$0.003-0.005 stripped). Tool calls run under
`--permission-mode bypassPermissions` since a headless call has no TTY for an
approval prompt — confirmed executing real commands with zero approval
prompts. Doesn't affect `llm-eval-harness`, which has its own separate
`claude_cli.py`. Timeout is 300s (up from 60s) to accommodate real tool-use
turns.

**Cross-provider mid-conversation continuity is still unsolved.**
`codex_cli.invoke` accepts `session_id` for `Provider`-protocol conformance
but ignores it — `codex exec` has no flag to pre-assign a session id
(continuation is the separate `codex exec resume <id>` subcommand). Only
works today because every tier maps to Claude; switching providers
mid-conversation would break continuity regardless, since each CLI's session
state is local to it.

**Streaming transport + ANSI styling (2026-07-27).** `claude_cli.py` switched
from `--output-format json` to `--output-format stream-json
--include-partial-messages --verbose` (`--verbose` is required — omitting it
is a hard CLI error) — one JSON event per line as it arrives. `invoke()` now
uses `subprocess.Popen`, with a `_drain()` helper reading `stdout`/`stderr`
concurrently via `select.select()` (not a plain `for line in proc.stdout`
loop, which would reintroduce the classic pipe deadlock
`subprocess.run(capture_output=True)` avoided for free). The 300s timeout is
now a wall-clock deadline checked every `_drain()` iteration. Unix-only —
`select()` doesn't support pipes on Windows, unverified there. `invoke()`
gained `on_event: Callable[[dict], None]`, called once per parsed event
(`codex_cli.invoke()` accepts it for interface conformance but ignores it —
still on `--output-last-message`). `route_and_run()` gained a matching
`on_event` passthrough plus `on_decision: Callable[[RouteDecision], None]`,
fired the instant `route()` resolves, before the provider call starts.

`tui.py` is a stdlib-only ANSI styling module (escape codes, not
`rich`/`textual`) rendering in the visual spirit of Claude Code's own CLI —
not a pixel-exact clone, not a pty. `chat_loop()` wires a `tui.StreamRenderer`
into `on_event` for live token-by-token streaming; `on_decision` prints the
`[provider/model, tier=X]` header immediately. `chat_loop()`'s `write_fn`
param (raw unterminated chunks, for streaming) is kept separate from
`print_fn` (discrete lines, what tests assert on). `format_response()`
remains the non-streaming full-message formatter (still the pinned test
contract) but `chat_loop`'s success path no longer calls it, to avoid
double-printing the streamed answer.

**Testing gotcha, worth restating:** switching `invoke()` from
`subprocess.run` to `subprocess.Popen` silently invalidated every test that
patched `subprocess.run` — they kept "passing" while actually falling through
to a real, unmocked call (caught by a bash-level timeout after ~15-20 real
calls had already gone out). Every `claude_cli.py` test now patches
`subprocess.Popen` directly (`_FakeProcess`/`_FakeStream` in
`tests/test_claude_cli.py`). Re-verify with the narrowest affected test (hard
timeout) before trusting a green suite after any future change to this call
mechanism.

**A static input-box frame was tried and reverted the same day.** `input()`
hands line editing to the terminal's own readline layer, which
overwrites/advances at the cursor — no way to keep a border in place once
text wraps, since the frame was drawn before input started. Doing it properly
needs raw terminal mode (`termios`/`tty`) with a hand-rolled line editor —
declined twice as a much bigger build than everything else in this pass. Back
to a plain `tui.prompt()`.

**Plan mode is explicitly deferred** — a future two-turn flow
(`--permission-mode plan` to produce a plan, a follow-up `--resume` call to
execute it), noted but not designed further until needed.

**Login always defers to the provider's own interactive command.**
`claude_cli.login()`/`codex_cli.login()` shell out with inherited stdio (no
`capture_output`/`timeout`) to `claude auth login --claudeai`/`codex login`
so the user completes the real OAuth/device flow in the same terminal —
`repl.py` never parses that flow itself, and never trusts `login()`'s exit
code as proof of success (`check_auth()` afterward is the real source of
truth). `codex_cli.login()` has an unwired `device_auth: bool = False` param
for a future headless/SSH login prompt.

**`known_models.py` is informational only** — never consulted by
`route()`/`route_and_run()`. A hardcoded table of model slugs confirmed
reachable via this account's calibration history, used solely for `repl.py`'s
startup summary. Same staleness risk as the Codex slug list it's drawn from.

**Authenticating Codex alone makes zero tiers routable**, since
`tiers.TIER_MODELS` maps every tier to `"claude"` (no Codex model has cleared
a floor yet) — `repl.startup_auth_check()` can report Codex authenticated
while `repl.routable_tiers()` returns empty, and `main()` refuses to start
rather than letting every message fail individually.
`tests/test_repl.py::test_routable_tiers_against_real_tier_models_with_only_codex_authenticated`
is pinned against the real `TIER_MODELS` so it starts failing (in the good
way) the day a Codex tier gets calibrated in.

## Auth pre-flight check

Both provider adapters export `check_auth() -> tuple[bool, str]` that
`invoke()` calls first — `claude auth status --json` (parses `loggedIn`) and
`codex login status` (text-matches "logged in"/"not logged in"). An
unauthenticated call short-circuits to `ProviderResult(error="auth check
failed: ...")` before reaching the real model subprocess, instead of falling
through to whatever failure shape the underlying CLI produces on its own (a
nonzero exit with a CLI-specific stderr message, or — `llm-eval-harness`'s
worst observed case — every case coming back a misleadingly bad-looking
`parse_error`).

Both adapters' logged-out shapes are confirmed against real output
(2026-07-26), without actually logging this dev account out:
- **Claude**: `env -u ANTHROPIC_API_KEY claude --bare auth status` →
  `{"loggedIn": false, ...}` at exit 1. `--bare` skips keychain/OAuth reads
  entirely.
- **Codex**: `CODEX_HOME=<empty dir> codex login status` → "Not logged in" at
  exit 1.

Reuse these two techniques for testing this gate instead of a real logout,
which needs an interactive re-auth flow to undo. Both `check_auth()`
implementations were already written defensively (treat anything that
doesn't clearly parse as "logged in" as unauthenticated) before this
confirmation — the real output matched, so only docstrings/tests moved from
"assumed" to "confirmed with a regression test pinned to the real string."

A new provider's `check_auth()` should follow this same shape — don't skip it
just because the provider's own nonzero-exit path eventually surfaces an auth
error; the point is failing fast and consistently.

## Known rough edges

- `providers/codex_cli.py` is verified against a real `codex-cli 0.145.0`
  install and a real authenticated call. Two things still open: (1) no
  dollar-cost field anywhere in `codex exec`'s output (stderr prints an
  unstructured token count, no per-model pricing to convert it), so
  `cost_usd`/`duration_ms` stay 0.0/0 placeholders; (2) `--output-last-message`'s
  behavior on a genuine content refusal (vs. a hard API error, which is
  covered) is unconfirmed. Currently unreachable through the router anyway —
  no tier routes to it yet.
- Codex has no flag equivalent to `--disallowed-tools "*"` — `--sandbox
  read-only` is the closest analog (can't write files) but can still run
  read-only shell commands. Don't assume cost/latency parity between the two
  adapters. Valid model names depend on auth mode (a ChatGPT-account login
  rejects some outright, confirmed via a real 400). As of 2026-07-23,
  reachable on the dev account: `gpt-5.4-mini`, `gpt-5.6-luna`,
  `gpt-5.6-terra`, `gpt-5.5`; not reachable: `gpt-5.6-sol`, `gpt-5.3-codex`,
  `gpt-5.1-codex-mini`, `gpt-5.4-nano`, `gpt-5.4`, `gpt-5.2` (all 400).
  Re-probe with a single cheap `codex exec` call before trusting either list
  on a different account.
- `tiers.TIER_MODELS`'s Claude entries are backed by real judged data, not a
  guess — as of 2026-07-23 a clean, monotonic ladder on `bug_triage`/
  `v1_naive` (haiku 60% fully-correct → sonnet 66.7% → opus 73.3%, judge
  coherence flat ~0.84-0.85). No Codex model clears haiku's floor yet, so the
  map stays Claude-only — see `llm-eval-harness/CLAUDE.md`'s calibration
  status section for the full table.
- Tool access is resolved for every Claude call, not just `llm-chat`'s —
  `llm-route`'s one-shot path goes through the same `claude_cli.invoke()`,
  sharing the same full-functionality/`bypassPermissions` behavior. Confirmed
  deliberately with the user (2026-07-26): one adapter, one behavior, rather
  than threading a cost-mode flag through both call paths.
- ~~No shadow evaluation, no live dual-routing, no drift auditing~~ —
  **closed 2026-07-28** for both halves (see "Drift auditing" and "Shadow
  evaluation" above). Still true: both `audit_tier2.py` and
  `report_shadow_divergence.py` are manual/periodic scripts, not
  self-correcting or cron'd — nothing auto-adjusts `AGREEMENT_THRESHOLD` or
  deprecates tier 1 based on what they report, a human still reads the
  output and decides.
