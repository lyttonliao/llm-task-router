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
  continuous-learning classifier (2026-07-27, see "The classifier is a
  two-tier cascade" below): `sentence-transformers` (local embeddings) and
  `psycopg`/`pgvector` (Postgres client) are real, permanent runtime
  dependencies of `tier2_classifier.py`. This was a deliberate, discussed
  choice, not scope creep — building the actual long-term product (a
  continuous-learning classifier) was judged more valuable than staying
  dependency-free for its own sake. The rule still holds everywhere else in
  this repo; don't read this exception as license to add dependencies
  elsewhere without the same deliberate discussion.
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
  embeddings.py   - tier-2: local sentence-transformers embedding wrapper
  vector_store.py - tier-2: pgvector-backed routing_examples store
                    (nearest_neighbors/insert_example/all_labeled_examples) -
                    one of two modules allowed to contain SQL, see
                    decision_log.py below
  tier2_classifier.py - tier-2 orchestration: NN lookup + cheap-LLM
                    fallback-with-write-back (see "Tier 2: the
                    continuous-learning classifier" below)
  decision_log.py - drift auditing: the second (and, for now, final) module
                    allowed to contain SQL - write-only routing_decisions
                    log, one row per route() call (see "Drift auditing:
                    logging every routing decision" below)
  db/schema.sql   - routing_examples + routing_decisions DDL, applied
                    manually via psql
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
  tui.py          - stdlib-only ANSI styling for repl.py's live-streaming
                    terminal output (see "Streaming transport + ANSI
                    styling" below)
```

## The classifier is a two-tier cascade: heuristic + continuous-learning classifier

The design was originally scoped as a three-tier cascade (heuristic rule
table → a trained model cold-started from `llm-eval-harness` golden-set
labels → a cheap-LLM call for the remaining ambiguity band). That changed
during design review on 2026-07-27: rather than build the originally-scoped
intermediate step (a static offline-trained classifier, then a separate
cheap-LLM-fallback tier), the decision was to build the actual long-term
product directly. **Tier 2 now absorbs what would have been both the
"trained model" and "cheap-LLM fallback" tiers into one mechanism** — see
"Tier 2: the continuous-learning classifier" below for what was actually
built. `classifier.py`'s heuristic rule table remains tier 1, unchanged,
still the free/zero-latency first pass; it's expected to be deprecated later
(not yet) once tier 2 has earned enough coverage to be trusted alone.

~~Default-to-escalate under uncertainty, not default-to-cheap~~ - **superseded
2026-07-27, see below.** This was the original principle (an underrouted,
silently-wrong answer is worse than an overrouted, correct-but-pricier one,
because the former is undetectable without dedicated auditing) and it drove
two fixes made the same day, in this order, before it was corrected outright:

1. **Fix 1 - no-signal fallback.** `classify_description()`'s
   fully-unrecognized description falls back to `task_type="architecture"`
   purely so `classify()` always has a valid grid key (see that function's
   docstring) - that label is not evidence the task is architecture-shaped.
   Feeding it straight into the grid meant "we have zero idea what this is"
   silently became "escalate to flagship", since `architecture`'s row is
   uniform `H`. A trivial toy prompt with no keyword overlap on either axis
   ("reply with exactly the word: pong") routed to opus and cost ~$0.18 to
   repeat one word. Fixed by hedging at `mid` when **both** axes are fully
   unresolved, instead of consulting the grid.
2. **Fix 2 (this superseded the "escalate under uncertainty" framing
   entirely) - flagship needs a real high-stakes signal, not just a shape
   match.** Fix 1 wasn't narrow enough: an *inferred* type/domain match
   still reached the grid's `H` cells on shape alone - `architecture`'s
   `TYPE_KEYWORDS` include bare words like `"design"`/`"strategy"`, which
   match a trivial question ("what's a good design for this button?")
   exactly as readily as a real one ("design the multi-region disaster
   recovery strategy for the payment system"). `router.route()` now requires
   **both** an H-mapped grid cell **and** (unless the caller explicitly
   provided that `task_type` - an explicit override is trusted as a
   deliberate judgment call, not second-guessed) a genuine high-stakes
   signal in the description itself
   (`classifier.has_high_stakes_signal()` - production/security/compliance/
   irreversibility/scale vocabulary, see `IMPACT_KEYWORDS`'s docstring for
   the full list and why it's deliberately narrow). An inferred `H` without
   that corroboration is capped at `mid`.

**The net effect: escalation to flagship is now precision-first, not
recall-first** - the opposite bias from the original principle above. This
was a real, accepted tradeoff at the time: a genuinely hard task that
doesn't happen to use recognized high-stakes vocabulary (e.g. "migrate the
k8s cluster to a new region" - no "multi-region"/"disaster recovery"
wording) would be underrouted to `mid` rather than reaching flagship.
**Closed 2026-07-27** by tier 2's `resolve_high_stakes()` (see "Tier 2: the
continuous-learning classifier" below) - that exact k8s-migration example
now correctly escalates to flagship, confirmed against a real run, not just
a design intention.

**Fix 3 - tier 3 (cheap-LLM fallback) built for the no-signal band,
2026-07-27.** The no-signal hedge (Fix 1) still routed every unclassifiable
description to `mid` uniformly - including genuinely trivial ones. A trivial
toy prompt ("repeat this word: hello") should cost the cheap tier, not
sonnet. A `TRIVIAL_KEYWORDS` list (symmetric to `IMPACT_KEYWORDS`) was tried
and explicitly rejected: it doesn't generalize ("string matching and regex
limitations are proving insufficient" - too many phrasings to enumerate). A
vector-DB/embeddings alternative was considered next and also rejected: it
breaks this repo's zero-third-party-dependency rule, and - more
importantly - `llm-eval-harness`'s own CLAUDE.md already ran this exact
experiment for a structurally similar soft-semantic-matching problem and
found embeddings don't cleanly separate it (concrete-vs-abstract phrasing
scored 0.54-0.60 cosine similarity with no threshold separating real
signal from same-case noise, and a bigger model didn't help - see that
repo's "regex, not AST or embeddings" section). **Built instead: `router.py`'s
own already-planned tier 3** ("a cheap-LLM-call fallback for the remaining
ambiguity band," named in this module's own docstring before this fix
existed). `_classify_via_llm()` makes one haiku call, asking the model to
judge the description's actual difficulty/consequence/ambiguity directly -
a judgment call, which a small model is structurally better suited for than
keyword matching. Scoped to the no-signal branch only; the
`needs_corroboration` branch (Fix 2) is untouched - it has a different
failure mode (underrouting a possibly-hard task, not overrouting a trivial
one) and deserves separate review.

This call reuses `providers/claude_cli.py`'s `invoke()`, but **not**
unchanged - that adapter is deliberately full-functionality for `llm-chat`
(~$0.07-0.30/call, see "llm-chat: interactive terminal client" above), and
calling it as-is for an internal one-word classification would have cost
about as much as the misroute it's meant to prevent, plus left tool
execution enabled on a call with no business touching tools. `invoke()`
gained an opt-in `disable_tools: bool = False` param (default `False` -
every existing call site is unaffected) that adds `--disallowed-tools "*"
--strict-mcp-config`, the same two flags `eval_harness`'s own stripped
adapter uses; `_classify_via_llm()` passes that plus an explicit
classifier-persona `system_prompt`, landing this call in `eval_harness`'s
~$0.003-0.005/call bracket instead of `llm-chat`'s. `_classify_via_llm()`
returns `None` (never raises) on a provider error, unparseable response, or
any exception - `check_auth()`'s `subprocess.run` has no `FileNotFoundError`
guard, so `invoke()` is not exception-safe, and `route()` treats `None` as
"fall back to the existing mid hedge," not as a crash.

**Cost/latency, stated plainly**: every no-signal request now costs one
extra (stripped, cheap-bracket) haiku call and adds latency before the real
task call starts. This is a genuine tradeoff against the zero-cost tier-1
path, accepted because it's what actually lets a trivial request reach the
cheap tier instead of either overpaying at mid (the old hedge) or
misclassifying via a keyword list that won't generalize.

## Tier 2: the continuous-learning classifier

Built 2026-07-27, same day as the design pivot away from the three-tier
scope above. Not a static trained model - a live, growing mechanism:
embeddings + a pgvector-backed store of labeled examples, falling through to
one cheap LLM call (writing its own answer back as a new labeled row) for
whatever the store can't yet answer confidently. Both `route()` call sites
below use the exact same primitive against two different questions - "what
task_type is this?" and "is this genuinely high-stakes?" - rather than two
separate mechanisms.

**New modules** (`llm_task_router/`): `embeddings.py` (`embed(text) ->
list[float]`, lazy-singleton `sentence-transformers` `all-MiniLM-L6-v2`,
384-dim), `vector_store.py` (the only module with SQL - `nearest_neighbors()`
and `insert_example()` against a `routing_examples` table, connection via
`DATABASE_URL`), `tier2_classifier.py` (`resolve_task_type()` /
`resolve_high_stakes()` - the orchestration `router.py` actually calls).
Schema at `llm_task_router/db/schema.sql`, applied manually once
(`psql -f llm_task_router/db/schema.sql`) against a local Postgres with the
`pgvector` extension - not run automatically, no migration framework at this
stage. `scripts/seed_vector_store.py` cold-starts the store from
`llm-eval-harness`'s 98 golden cases (the one place this repo reads the
sibling repo's files, and only offline/one-time - never at runtime), mapping
each case's suite file to a `task_type` label; no `is_high_stakes` ground
truth exists in that data, so that column starts entirely empty and only
fills in as real `needs_corroboration`-shaped requests get resolved through
tier 2 over time.

**Only LLM-fallback resolutions get written back, not NN-confident ones** -
a matching neighbor already covers that region of the embedding space, so
writing it again would add near-duplicate rows without new information.
This is what makes it "continuous learning" rather than a one-time cold
start: the exact same question never needs a second LLM call.

**`AGREEMENT_THRESHOLD` (0.8, i.e. ≥4/5 of the k=5 nearest neighbors must
share a label) came from a real leave-one-out check against the seeded
store, not a guess - and it overturned the original design en route.** Two
findings, both worth keeping since the first one directly contradicts what
was assumed going in:
1. Raw cosine similarity does **not** cleanly separate same-task_type from
   different-task_type neighbors at this data/model scale (same-label pairs
   measured median ~0.32, cross-label median ~0.29 - heavily overlapping). A
   hard similarity floor (the first draft of this module) would have
   rejected most true matches. This is the same shape of finding
   `llm-eval-harness`'s CLAUDE.md already documents for a structurally
   similar problem ("regex, not AST or embeddings") - off-the-shelf
   embeddings not cleanly separating a soft-semantic distinction at small
   data scale, confirmed independently a second time on a different problem.
2. Relative agreement among the k nearest neighbors **does** carry real
   signal despite (1): leave-one-out majority-of-5 vote scored 72.4%
   accuracy unconditionally (98 samples, 7 classes, chance ~14%), rising
   monotonically with agreement - 78.2% at ≥3/5 (80% coverage), 87.1% at
   ≥4/5 (32% coverage), 100% at 5/5 (8% coverage, too rare to rely on
   alone). `AGREEMENT_THRESHOLD=0.8` was chosen from that curve, erring
   toward the more-accurate LLM fallback over a borderline NN vote -
   consistent with this repo's original escalate-under-uncertainty instinct
   (superseded above for the *keyword* gate specifically, but still the
   right default for tier 2's own confidence gate). A neighbor set smaller
   than `NEIGHBOR_K=5` (the store barely has labeled examples yet for that
   axis - true of `is_high_stakes` at seed time) never counts as confident
   regardless of agreement fraction, for the same reason.

**`needs_corroboration` (Fix 2 above) is no longer permanently capped to mid
on a keyword-negative miss.** `router.route()`'s corroboration branch now
falls through to `tier2_classifier.resolve_high_stakes()` (reusing the same
embedding already computed for task_type resolution when there is one,
rather than paying for a second one) instead of capping unconditionally.
Keyword-positive matches still skip this entirely - free evidence, no need
to spend a call re-confirming it. `True` escalates to flagship for real;
`False` or `None` (tier 2 unavailable) both still cap to mid, exactly like
the old unconditional behavior - tier 2 only ever adds a path to a *correct*
escalation, never removes the existing safety cap.

**Both `resolve_task_type()` and `resolve_high_stakes()` return `None`/fall
back on any failure** - provider error, unparseable text, or a raised
exception - same discipline `_classify_via_llm()` already established for
tier 3's no-signal band. `router.py` treats tier-2-unavailable as "fall back
to the existing heuristic-only behavior," never as a crash.

**`resolve_high_stakes()` gained a `source` field, 2026-07-28** -
`HighStakesResolution(is_high_stakes, source)` replaces the bare
`bool | None` it used to return, symmetric with `resolve_task_type()`'s
existing `Tier2Resolution(task_type, source)`. This was needed by the drift
auditing work below, not a standalone cleanup: without it, a live decision
log couldn't tell whether a high-stakes resolution came from the NN vote or
the LLM fallback, only that one of them fired.

## Drift auditing: logging every routing decision

Built 2026-07-28, closing the "no shadow evaluation, no live dual-routing,
no drift auditing" gap this file used to list under "Known rough edges."
Worth being precise about what this is *not*: `llm-eval-harness`'s
`calibrate-tier` skill is offline calibration against a fixed, 98-case
golden set - it answers "does model X clear tier Y's quality floor," never
touches live traffic, and was never a substitute for this. Before this work,
`route()` made a real decision on every call but discarded it the instant
`route_and_run()` returned - there was no logged traffic to check
`tier2_classifier.AGREEMENT_THRESHOLD` (tuned only against the 98-row seed
set) against, and no way to catch a wrong `llm_fallback` write-back short of
noticing bad routing behavior after the fact.

**`decision_log.py`** (new SQL module, see "Architecture" above) writes one
row to a new `routing_decisions` table (`db/schema.sql`) per `route()` call:
the resolved `task_type`/`domain` and *which mechanism* produced each -
`provided` / `inferred` / `tier2_nn` / `tier2_llm_fallback` / `fallback` for
task_type, `keyword` / `tier2_nn` / `tier2_llm_fallback` / `unavailable` for
high-stakes - plus the final bias/tier/model/reason. `embedding` is null
whenever the heuristic grid resolved everything without ever calling
`embeddings.embed()` - logging a pure-heuristic decision doesn't force one
into existence. `route()` calls `log_decision()` wrapped in
`try/except Exception: pass`, same "never let this break routing" discipline
`tier2_classifier`'s own Postgres/LLM calls already follow - a logging
failure degrades to "no audit trail for this call," never to a broken route.
`tests/conftest.py` autouse-patches `log_decision` to a no-op for the whole
suite so the ~140 pre-existing tests needed zero changes; a handful of new
`test_router.py` tests override that fixture explicitly to assert the logged
fields on representative paths.

**`scripts/audit_tier2.py`** re-runs the exact leave-one-out methodology
whose one-time result is hardcoded in `tier2_classifier.py`'s
`AGREEMENT_THRESHOLD` comment (72.4%/78.2%/87.1%/100% accuracy at
unconditional/≥3/5/≥4/5/5/5 agreement on the 98-row seed) against the *live*
`routing_examples` table via the new `vector_store.all_labeled_examples()`,
instead of the frozen seed set - printing the same kind of curve plus an
explicit "current threshold holds / consider raising" verdict. It also flags
`source='llm_fallback'` rows whose neighbors now confidently disagree with
their own stored label - a signal an earlier LLM-fallback write-back was
probably wrong, printed for manual review only, never auto-corrected
(auto-correcting a label from a heuristic risks compounding the exact error
class this is trying to catch). An opt-in `--judge-flagged` pass gets one
cheap second LLM opinion per flagged row, reusing the same
haiku/`disable_tools=True`/stripped-system-prompt call shape
`tier2_classifier.py`'s own fallback already uses - not
`llm-eval-harness`'s `judge_score` (that judges *response quality* against a
task, a different question from "is this classification label correct").
Verified against the real local store the day this was built: 101 seeded +
grown `task_type` rows produced a live curve
(73.3%/79.0%/88.2%/100.0%) closely tracking the original seed-only numbers,
threshold verdict "holds," 0 suspect rows; `is_high_stakes` (3 rows at the
time) correctly reported "not enough rows yet" rather than crashing on too
small a sample. Manual/periodic, like `seed_vector_store.py` - no cron or
automation wired up yet.

**Real bug this surfaced, worth restating so it isn't repeated:**
`vector_store.all_labeled_examples()`'s first draft called `list(row[2])` on
the embedding column returned by a `register_vector()`-registered
connection, assuming it'd behave like an iterable/numpy array the way other
psycopg reads do. It doesn't - psycopg/pgvector returns a `pgvector.Vector`
object there, which isn't iterable; `list(...)` failed with a real
`TypeError` the first time this ran against the actual local Postgres store,
not a hypothetical. Fixed with `Vector.to_list()`. Every other module that
reads embeddings back out of Postgres should expect the same - `Vector`, not
a bare list or array - and verify against a real query before assuming
otherwise.

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
   Same discipline applies to `classifier.TYPE_DOMAIN_GRID` (task_type x
   domain -> tier), not just `TIER_MODELS` - see `llm-eval-harness/CLAUDE.md`,
   "Router tier synthesis across all 7 suites" (2026-07-27) for the current
   state: `code_gen`/`refactor` rows are now calibration-derived (uniform
   `L`, zero discrimination found across all three Claude tiers on hardened
   suites), the rest of the grid is still pillar 22's original heuristic,
   not yet calibration-confirmed. Don't re-derive that table here; follow
   the pointer.

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

**Session continuity, implemented 2026-07-26.** `TaskRequest.session_id` is
generated once per `chat_loop()` run (not per message) and threaded through
`route_and_run()` -> `provider.invoke(..., session_id=...)` unconditionally,
even when `None`. Every message in one `llm-chat` session shares the same
`session_id`, so conversation history continues even as the classifier
routes different messages to different Claude tiers/models mid-session -
built specifically so `llm-chat` could stay a thin router in front of real
Claude Code functionality (tools, real system prompt, CLAUDE.md/hooks, and
eventually plan mode) instead of reimplementing an interface that mimics it.
Two designs were considered and rejected first: a bespoke chat interface
that reimplements CLI functionality (drifts from upstream, throws away
plan mode/slash commands for free), and a raw PTY takeover of a live
interactive `claude` session injecting `/model` mid-session (undocumented,
unconfirmed, more fragile than the mechanism below).

**The real mechanism, confirmed against real `claude` 2.1.220 output on
2026-07-26 (not guessed) - see the verification commands in
`providers/claude_cli.py`'s module docstring:**
```bash
claude -p "Remember this: my favorite fruit is starfruit. Just say OK." \
  --model haiku --session-id "$SID" --output-format json      # establishes the session

claude -p "What did I say my favorite fruit was?" \
  --resume "$SID" --model sonnet --output-format json         # -> "Starfruit."
```
Reusing `--session-id` on a second call FAILS outright ("Session ID ... is
already in use") - the correct flag for every call after the first is
`--resume`, and it does correctly continue history across a `--model`
change. `claude_cli.py` tracks which session ids have already had their
establishing call in a module-level `_established_sessions` set (only added
after a confirmed success, not on error/timeout) so callers never need to
know or care which flag a given call becomes - they just pass the same
`session_id` every time.

**Full functionality, not cost-minimized - a deliberate choice, distinct
from `llm-eval-harness`'s adapter.** `llm-chat` is a real interactive
client, not an offline benchmarking harness, so `providers/claude_cli.py`
no longer strips the system prompt or disables tools/MCP the way
`eval_harness/claude_cli.py` does - real tools, real system prompt, real
CLAUDE.md/hooks all work, at the real per-call cost that comes with that
(~$0.07-0.30/call observed here, vs. ~$0.003-0.005/call stripped). Tool
calls run under `--permission-mode bypassPermissions` since a headless
`subprocess.run()` call has no TTY to show an approval prompt - confirmed
against real output to execute real commands (`pwd`) with zero approval
prompts (`permission_denials: []`). This only touches this repo's own,
independent `claude_cli.py` - `llm-eval-harness` has its own separate copy
(see "Independent from `llm-eval-harness`" above), so there's no cross-repo
cost or behavior change. Timeout bumped from 60s to 300s to accommodate real
tool-use turns instead of stripped single completions.

**Still true, still unsolved:** cross-provider mid-conversation continuity.
`codex_cli.invoke` accepts `session_id` for `Provider`-protocol conformance
but ignores it entirely (`codex exec` has no flag to pre-assign a session id
- confirmed via `codex exec --help` - continuation there is the separate
`codex exec resume <id>` subcommand). This only works today because every
tier in `tiers.TIER_MODELS` maps to Claude - switching which provider a
message routes to mid-conversation would still break continuity, since each
CLI's session state is local to that CLI. Constraining this feature to
Claude-only wasn't a new limitation introduced by this change - it formalizes
what was already true.

**Streaming transport + ANSI styling, added 2026-07-27.** Two related but
separate changes, both scoped to `llm-chat` only (`llm-route`'s one-shot
`cli.py` path is unaffected beyond riding the same transport):

- `claude_cli.py` switched from `--output-format json` (one blob, read after
  the process exits) to `--output-format stream-json --include-partial-messages
  --verbose` (one JSON event per line, readable as they arrive). `--verbose`
  is not optional - confirmed via real output, omitting it is a hard CLI
  error ("requires --verbose"). `invoke()` now shells out via `subprocess.Popen`
  instead of `subprocess.run`, and a new `_drain()` helper reads `stdout`
  (NDJSON events) and `stderr` concurrently via `select.select()` - not a
  plain `for line in proc.stdout` loop, which would reintroduce the classic
  pipe-deadlock `subprocess.run(capture_output=True)` avoided for free
  (child blocks writing to a full stderr pipe while we block waiting for
  stdout EOF that never arrives). The 300s timeout that used to be
  `subprocess.run(timeout=300)` is now a wall-clock deadline checked every
  `_drain()` iteration, since `Popen` has no equivalent for an incrementally
  read stream. Only exercised on Unix - `select()` doesn't support pipes on
  Windows, unverified there. `invoke()` gained an optional
  `on_event: Callable[[dict], None]` param, called once per parsed event
  before the final `"type":"result"` line is recognized - `codex_cli.invoke()`
  and the `Provider` protocol (`base.py`) both accept the same param for
  interface conformance, but `codex_cli.py` ignores it (still on
  `--output-last-message`, not `--json`'s own JSONL stream - real, not-yet-done
  work, same shape as this Claude-side change was before it was done).
  `router.route_and_run()` gained a matching `on_event` passthrough plus a
  separate `on_decision: Callable[[RouteDecision], None]`, fired the instant
  `route()` resolves - before the provider call starts - specifically so a
  caller can show routing info immediately rather than only once the full
  response lands.
- `tui.py` (new) is a stdlib-only ANSI styling module - ANSI escape codes,
  not `rich`/`textual`, matching this repo's zero-dependency rule. It renders
  in the visual spirit of Claude Code's own CLI (a colored provider bullet,
  one-line tool-call summaries, a dim cost/duration footer, a transient
  "thinking…" status cleared the moment real text starts) - deliberately not
  a pixel-exact clone of a closed-source renderer, and not a pty (a raw pty
  takeover was already considered and rejected for `llm-chat`, see above -
  this doesn't revisit that). `repl.py`'s `chat_loop()` wires a
  `tui.StreamRenderer` into `on_event` so the model's answer streams live,
  token-by-token, instead of appearing all at once after the call returns;
  `on_decision` prints the `[provider/model, tier=X]` header the moment
  routing resolves. `chat_loop()` gained a `write_fn` param (default
  `tui.default_write`, raw `sys.stdout.write`+`flush`) separate from
  `print_fn` - `print_fn` still emits discrete, newline-terminated lines
  (what tests assert on), `write_fn` emits raw unterminated chunks for live
  streaming; collapsing these into one parameter would make either the
  streamed text or the line-based assertions awkward to test. `format_response()`
  is kept as the non-streaming full-message formatter (still the pinned text
  contract in tests) but `chat_loop`'s success path no longer calls it -
  doing so would print the answer a second time after it was already
  streamed.

**A mocking gap surfaced once, worth restating so it isn't repeated:**
switching `invoke()` from `subprocess.run` to `subprocess.Popen` silently
invalidated every existing test that patched `subprocess.run` - they kept
"passing" their own mock setup while actually falling through to a real,
unmocked `Popen` call. Caught by a bash-level timeout during development
after ~15-20 real (cheap, haiku/sonnet, trivial-prompt) calls had already
gone out - a real, if small, unintended cost, not a hypothetical one. Every
`claude_cli.py` test now patches `subprocess.Popen` directly (`_FakeProcess`/
`_FakeStream` doubles in `tests/test_claude_cli.py`) and asserts it, not just
`subprocess.run`. Whenever this adapter's underlying subprocess call
mechanism changes again, re-verify by running the single narrowest affected
test first (with a hard timeout) before running the full suite - don't
assume an existing green test file still means what it used to.

**A static input box frame was tried and removed the same day (2026-07-27).**
`chat_loop()` briefly printed a box-drawing top/bottom border around each
`input_fn()` call, width-matched to the terminal via
`shutil.get_terminal_size()`. Rejected once actually used: `input()` hands
line editing to the terminal's own readline layer, which just overwrites/
advances at the cursor as you type - so there's no way to keep a right
border in place, and once typed text line-wraps in the terminal there's no
border around the wrapped lines at all, since the frame was printed before
input even started. A box that genuinely wraps around wrapped text needs
raw terminal mode (`termios`/`tty`) and a hand-rolled line editor rebuilding
backspace/arrows/history/resize-handling from scratch - evaluated and
explicitly declined twice (once before building the static version, again
after seeing its limits) as a substantially bigger build than everything
else in this pass combined. `chat_loop()` is back to a plain styled
`tui.prompt()` with no border.

**Plan mode is explicitly deferred, not part of this pass.** A future
seam - a two-turn flow, one call with `--permission-mode plan` to produce a
plan, a follow-up call in the same session (via `--resume`) to approve/
execute it - noted here the same way multi-turn continuity was noted before
it was implemented, not designed further until it's actually needed.

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
- Tool access is resolved for every Claude call, not just `llm-chat`'s -
  `providers/claude_cli.py` no longer disables tools by default (see
  "llm-chat: interactive terminal client" above), and `llm-route`'s one-shot
  CLI path goes through the same `claude_cli.invoke()`, so it shares the same
  full-functionality/`bypassPermissions` behavior. Confirmed deliberately
  with the user (2026-07-26), not an accidental side effect: one adapter, one
  behavior, rather than threading a cost-mode flag through both call paths.
- ~~No shadow evaluation, no live dual-routing, no drift auditing~~ -
  **closed 2026-07-28** for the auditing half specifically, see "Drift
  auditing: logging every routing decision" above: every `route()` decision
  is now logged, and `scripts/audit_tier2.py` re-validates
  `AGREEMENT_THRESHOLD` and flags suspect `llm_fallback` rows against live
  data. Still true: no live dual-routing (a request is never sent to two
  tiers to compare), and the audit script is manual/periodic, not a running
  process that self-corrects on its own.
