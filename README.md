# llm-task-router

Given a task description, classify it and route it to the cheapest model
tier that clears a quality floor for that task's category, then actually run
it. This is the live-routing counterpart to
[`llm-eval-harness`](../llm-eval-harness), which benchmarks prompt/model
quality offline to produce the calibration data this router's tiers are
built from.

This file is the human-facing pitch and design rationale. For current
implementation state, open questions, and "how do I add X" instructions, see
[`CLAUDE.md`](./CLAUDE.md) — that file is the source of truth for what's
actually built; this one is the source of truth for *why* it's shaped this
way.

## The problem

Calling the biggest model for every task is wasteful; calling the cheapest
model for every task silently produces wrong answers on the tasks it can't
handle. The router's job is to close that gap: pick the cheapest tier that
still clears a per-category quality floor, and never guess at what that
floor is — measure it.

## Design pillars

These were established one at a time across a Socratic planning process
(2026-07-21 through 2026-07-22, three sessions, before any code existed) —
each one is a specific design question that got pressure-tested and
resolved, not a category summary. Reconstructed here from the actual session
transcripts rather than from a compressed memory note, so the numbering and
content match what was actually decided at the time.

**Objective function**

1. **Constrained optimization, not a scalarized weighted score.** Minimize
   cost subject to (a) a hard per-category quality floor and (b) a hard
   per-category latency ceiling. Explicitly rejected `w_q·quality − w_c·cost
   − w_l·latency`: a weighted blend can pay extra for a marginal quality/
   latency gain even after the floor is already cleared, which defeats the
   point of a cost-minimizing router. Making latency a second hard ceiling
   (instead of a soft marginal comparison) closes that gap without
   reintroducing scalarization.

**Role of the eval harness / offline calibration**

2. **The eval harness's job here is offline calibration, not live scoring.**
   Its scorers (`rule_based_score`, `judge_score`) need a completed output
   plus a known golden answer — neither exists at request time. Its real
   role is benchmarking every model tier per task category ahead of time to
   find the minimum-sufficient tier; that becomes `tiers.py`'s calibration
   data, not something computed per-request.

**Routing signal**

3. **Live routing signal is input-only — never a response feature.** Task
   category and structural shape of the request, decided before any model
   has answered. There's no response to inspect at decision time, so the
   classifier can't use anything about output quality to route.

**Asymmetric risk**

4. **Default to escalate under uncertainty, not default to cheap.**
   Underrouting (a confident-but-wrong cheap answer) is a silent,
   undetectable failure; overrouting (a correct but pricier answer) is a
   visible, bounded cost that just shows up on the bill. Given that
   asymmetry, low classifier confidence should push a task up a tier, not
   down.

**Shadow evaluation**

5. **Shadow evaluation (champion-challenger) closes the live-detection gap.**
   Sample live traffic, dual-route a slice to the chosen tier *and* one tier
   up, and compare via inter-tier agreement rate — there's no golden label
   in production, so this is a judge-style proxy, not exact-match. High
   disagreement recalibrates the escalation threshold.

**Cold-start → flywheel**

6. **Cold start resolves into a flywheel, it isn't a permanent limitation.**
   Day one: the golden set gives rough per-category quality-floor thresholds
   (pillar 2), not a trained classifier. Bootstrap phase: heuristic rules
   plus pillar 4's default-up caution cover the gap, accepting some
   overpaying. Flywheel: shadow eval (pillar 5) continuously generates
   real-distribution labeled data (agreement/disagreement per task), which
   periodically retrains the classifier — unlike the static golden set, this
   scales with actual usage and matches the real input distribution.

**Classifier input features**

7. **`category` is a free, already-labeled structural feature — reuse it.**
   `TestCase.expected_category` / `ModelOutput.predicted_category` already
   exist in `llm-eval-harness`'s schema. Rule #1 for the heuristic tier is
   mechanical: benchmark every tier against every golden case (pillar 2),
   group by category, and for each category pick the cheapest tier that
   clears the quality floor — no new feature engineering needed for the
   first heuristic. Richer features (e.g. "how many points need
   addressing") come later, once category-only routing proves too coarse
   for a category that's internally heterogeneous.

**Severity vs. difficulty**

8. **Difficulty is distributional rarity, not severity.** A common critical
   bug has a well-trodden solution path any tier nails; a rare, low-severity
   bug can be out-of-distribution and prone to confident-but-wrong reasoning.
   This is exactly the failure mode `judge_score` exists to catch
   (label-correct-but-unsupported-reasoning) — severity and difficulty are
   different axes and shouldn't be conflated when deciding escalation.

**Correlated-error blind spot**

9. **Shadow eval's agreement-rate check has a blind spot: correlated
   errors.** Same-model-family tiers can converge on the *same* wrong answer
   for a rare/OOD input — agreement rate reads that as "confident, no
   escalation needed" when it's actually the worst-case failure, silently
   passed through. Mitigations, cheapest first: (a) input-novelty detection
   (embedding distance from the known-resolved corpus, flags before either
   model even answers), (b) self-consistency sampling (same model, multiple
   samples, check reasoning stability), (c) cross-family verification (a
   genuinely independent model family as checker), (d) a human-audit
   backstop that feeds corrections back into the golden set. This is also
   *why* `llm-eval-harness`'s judge call stays hardcoded to Claude regardless
   of which provider is under test — grading with a different model lineage
   than the one being graded avoids exactly this correlated-error risk.

**Classifier mechanism (pillars 10-15)**

10. **Three-tier confidence cascade, not a single classifier mechanism.**
    Heuristic rule table (cheapest, fires on confident extremes, zero
    classification cost) → small trained model, cold-started per pillar 6
    (for cases the heuristic doesn't confidently cover) → cheap LLM call
    (remaining ambiguity band). Chosen because the classification step
    itself has a cost, and "cheap classification + right-sized model" only
    beats "always call one safe middle-tier model" if that breakeven math
    actually works out — folded into pillar 1's objective function, not
    treated as free.
11. **Escalation trigger = score-overlap between adjacent tiers, not a fixed
    confidence threshold.** This does double duty: it's both the signal that
    triggers escalation *and* where shadow eval should concentrate its
    dual-routing (active-learning style) — the highest-value boundary labels
    come from exactly where the classifier is least sure.
12. **Cascade-depth decay is emergent, not an explicit schedule.** As
    retraining tightens the decision boundary, the score-overlap band itself
    narrows, so escalation triggers less often over time on its own — no
    separate explore/exploit decay parameter needed on top of that.
13. **Live routing stays least-required; shadow eval is the separate
    exploration layer.** Live routing always stops at the first tier that
    clears the quality floor — anything else defeats the router's
    cost-saving purpose. The "holistic/explore more" behavior lives entirely
    in shadow eval's constant, deliberately non-decaying exploration budget,
    not as a live-routing default.
14. **Shadow eval also audits the confident heuristic tier, not just the
    ambiguous band.** If shadow eval only samples the score-overlap zone,
    the "golden rule" heuristic tier — the one making the most confident
    decisions — is also the one tier nothing ever double-checks. That's
    exactly where a silent, confident-but-wrong failure (pillar 8/9) would
    hide undetected.
15. *(Consolidation checkpoint — no new content: the classifier mechanism
    was confirmed fully specified by 10-14 combined.)*

**Category taxonomy (pillars 16-19)**

16. **Two-axis taxonomy: task type × domain.** Domain reuses the existing
    bug-triage taxonomy (frontend/backend/infra/data/other) for free per
    pillar 7; type is a new axis describing the shape of work being asked.
17. **Seven task types:** triage, code-gen, summarization, multi-step, code
    review, refactor, architecture/design reasoning. An eighth candidate —
    Socratic quiz/learning-reinforcement interactions — was explicitly
    scoped *out*: there's no single correct answer to grade against, so it
    doesn't fit a quality-floor benchmark the way the other seven do.
18. **Domain's effect on difficulty is type-dependent, not uniform.** The
    *strength* of the domain interaction varies by type — architecture×infra
    skews toward genuine tradeoffs/no-clean-answer, triage×frontend skews
    toward pattern-matching — rather than domain either mattering for every
    type or for none.
19. **The tradeoff-heavy vs. pattern-closed distinction is hand-authored
    now, learned later.** Written down today as an explicit tier-1 heuristic
    prior (sidesteps the cold-start problem from pillar 6 — no training data
    required), meant to be gradually superseded by the trained classifier's
    own learned interaction weights as shadow-eval data accumulates for
    those type×domain cells.

**The concrete grid**

22. **A confirmed 7×5 escalation-bias grid** (task type × domain, low/
    medium/high), grounded in pillars 8 and 18: summarization is uniformly
    low regardless of domain; architecture/design is uniformly high
    regardless of domain; the middle types (code-gen, multi-step, code
    review, refactor) are where domain actually does interaction work —
    infra and data consistently skew higher (blast-radius/silent-failure
    risk, per pillar 9), frontend consistently skews lower.

> **Numbering note:** the planning transcript's own running count jumps from
> "19 pillars" to "21 pillars" resolved right after pillar 1 got closed, and
> then to "22" for the grid above — without two more distinctly-stated
> pillars appearing in between. That's most likely a counting drift in the
> original session rather than two lost decisions; nothing has been invented
> here to fill pillars 20-21, and they're left out rather than guessed at.

**Implementation pillars (established during scaffolding, not planning)**

- **Subscription CLIs, not API keys, for every provider.** Every model call
  shells out to a headless CLI (`claude -p`, `codex exec`) instead of an
  API/SDK — no separate per-token billing, runs on existing subscriptions.
  Same rationale as `llm-eval-harness`, applied to a live-routing runtime
  instead of an offline benchmark runner.
- **Independent from `llm-eval-harness` at the code level.** The two repos
  interact conceptually — benchmark runs calibrate the tier map — but not
  through a shared package. A live-routing runtime and an offline benchmark
  runner are different enough consumption patterns that coupling them would
  be the wrong kind of DRY.
- **A tier mapping is only as good as the quality floor behind it.**
  `tiers.py` should never be hand-edited with a guessed (provider, model)
  pair. Every entry — Claude or otherwise — should trace back to a real,
  judged benchmark run in `llm-eval-harness/runs/`. As of 2026-07-23 that's
  true for the Claude entries (haiku/sonnet/opus, judged, monotonic) and
  explicitly *not* true for any Codex model yet — none of the four Codex
  models this account can reach clear the cheap tier's floor. See that
  repo's CLAUDE.md for the full comparison table.

## Setup

### 1. Prerequisites

- Python 3.14+
- [`uv`](https://docs.astral.sh/uv/) (recommended — all commands below use it)
- A local Postgres install with the [`pgvector`](https://github.com/pgvector/pgvector)
  extension available (`CREATE EXTENSION vector` must succeed) — required
  even if you only want tier-1 heuristic routing, since `router.py` imports
  `vector_store`/`embeddings` unconditionally
- An authenticated `claude` CLI (`claude auth login --claudeai`) for routed
  runs and `llm-chat`; an authenticated `codex` CLI too if you plan to add a
  Codex tier (see "Adding a provider" below) — no tier is calibrated to
  Codex yet, so it isn't required for normal use
- `tmux` on `PATH` — `llm-chat` delivers every message into a persistent
  `tmux`-backed session (`terminal.py`), checked via `terminal.tmux_available()`
  before it starts; without it, `llm-chat` refuses to start rather than
  failing mid-session
- macOS/Linux for `llm-chat` — `tmux` itself doesn't run natively on
  Windows (WSL/Cygwin only); `terminal.py` has a `_spawn_windows()` path for
  the visible-window attach step, but it's best-effort and unverified
  against a real Windows install (see `docs/rough-edges.md`). The `route`
  CLI itself is platform-independent

### 2. Install dependencies

```bash
uv sync --group dev
```

This pulls in `psycopg[binary]`, `pgvector`, and `sentence-transformers` —
real runtime dependencies of the tier-2 classifier, not optional extras.

### 3. Create the database and apply the schema

```bash
createdb llm_task_router
psql llm_task_router -f llm_task_router/db/schema.sql
```

`schema.sql` is applied manually and is **not idempotent** — it's a one-time
setup script (`CREATE TABLE` errors on a second run against a database that
already has the tables). It creates the `vector` extension plus two tables:
`routing_examples` (tier-2's labeled nearest-neighbor store) and
`routing_decisions` (the drift-audit / shadow-eval log). If you're
re-applying against an older database that predates shadow evaluation,
run the three commented `ALTER TABLE ... ADD COLUMN shadow_*` statements at
the bottom of that file instead of the `CREATE TABLE`.

Set `DATABASE_URL` so the router can reach it (every command below assumes
this is set). Copy `.env.example` to `.env` and fill it in, rather than
exporting it in your shell profile:

```bash
cp .env.example .env
```

`llm_task_router/__init__.py` loads `.env` automatically (see
`env_config.py`) - resolved next to the installed package, not your shell's
`cwd`, so it's picked up correctly whether you run `llm-route`/`llm-chat`
from inside this repo or, once installed via `uv tool install --editable .`
(see "Running `llm-chat` from anywhere" below), from anywhere else. A real
shell-exported `DATABASE_URL` still overrides whatever `.env` says, so
`DATABASE_URL=... uv run llm-route ...` one-off overrides keep working
unchanged. (A shell-profile export was tried first and reverted - it only
covers the shell it's sourced in, and doesn't help a caller invoking
`llm-chat` from a machine/shell where that profile line was never added;
see `docs/drift-and-shadow.md`'s "DATABASE_URL only in launchd plists"
writeup for the real incident that prompted this.)

### 4. Seed the tier-2 vector store

Before running this step (or any first call that reaches tier 2 — `embed()`
in `embeddings.py` lazy-loads `sentence-transformers/all-MiniLM-L6-v2` from
Hugging Face Hub the first time it's called), leave `HF_HUB_OFFLINE`
commented out in your `.env` (see step 3) — it ships commented out in
`.env.example` for exactly this reason. If it's already `1` with nothing
cached yet, the first embedding call fails outright: there's no local cache
for `sentence-transformers` to fall back to, so "offline mode" has nothing
to serve.

**Once the model is cached, uncomment `HF_HUB_OFFLINE=1` in `.env`.** This
isn't just an optional preference: without it, every fresh process that
imports `embeddings.py` (`llm-route`, `llm-chat`,
`llm-route-audit-tier2`, `scripts/seed_vector_store.py` — anything that
touches tier 2) makes a network round-trip to Hugging Face Hub on startup to
check for a newer model revision, *even though the file is already cached
locally and gets loaded from there either way*. That round-trip is what
prints:

```
Warning: You are sending unauthenticated requests to the HF Hub. Please set
a HF_TOKEN to enable higher rate limits and faster downloads.
```

This is a harmless, non-fatal warning — it comes from an `X-HF-Warning`
response header the Hub sends back on anonymous requests
(`huggingface_hub`'s own `utils/_http.py`), not an error, and the model
still loads correctly afterward. But since `llm-chat`/`llm-route` are each a
fresh process per run, `huggingface_hub`'s usual "only warn once" dedup
resets every time, so it reprints on every single invocation until that
revalidation call is skipped entirely by setting `HF_HUB_OFFLINE=1`.

The warning's suggested fix (create a real Hugging Face account token and
set `HF_TOKEN`) is the *other* valid way to quiet it, but is unnecessary
overhead here — this repo only ever requests one fixed, small, public model
name (`MODEL_NAME` in `embeddings.py`), so there's never a reason to check
for a newer revision, and going offline is simpler than managing a Hub
credential for a model this project doesn't need auth to use at all.

Tier 2 starts empty and needs the `llm-eval-harness` sibling repo checked
out alongside this one (default path `../llm-eval-harness`) to cold-start
from its 98 golden cases:

```bash
uv run python scripts/seed_vector_store.py
# or, if llm-eval-harness lives somewhere else:
uv run python scripts/seed_vector_store.py --cases-dir /path/to/llm-eval-harness/eval_harness/cases
```

This is a one-time operation — it only seeds `task_type` labels (no
`is_high_stakes` ground truth exists in the golden set, so that column
starts empty and fills in only from real traffic). Skipping this step
doesn't break routing: with too few neighbors, tier 2 falls through to a
cheap-LLM fallback (and still writes its own answer back), just without the
cold-start coverage.

### 5. Verify

```bash
uv run pytest -q

uv run python -m llm_task_router route \
  "Design a fault-tolerant deployment strategy for our Kubernetes cluster" \
  --dry-run
```

A working `--dry-run` call that prints a tier/provider/model/reason (rather
than a `DATABASE_URL`/connection traceback) confirms both the DB and the
package are wired up correctly.

## Usage

### Routing a single task

```bash
# Inspect a routing decision without calling a model
uv run llm-route route \
  "Design a fault-tolerant deployment strategy for our Kubernetes cluster" \
  --dry-run

# Route and execute a task with the currently calibrated provider/model tier
uv run llm-route route \
  "Summarize these release notes for a frontend team" \
  --type summarization --domain frontend
```

(`uv run llm-route ...` is the installed console script; `uv run python -m
llm_task_router route ...` is identical and still works.)

The tier-1 heuristic infers task type and domain from description keywords,
then selects a low/medium/high escalation bias; tier 2 (embeddings + the
`routing_examples` nearest-neighbor store, falling through to a cheap LLM
call) resolves whatever the heuristic can't confidently place. Supply
`--type` and/or `--domain` to skip inference for either. Ambiguous task types
fall back to the high-escalation architecture category rather than silently
routing to a cheap model. `--dry-run` shows the resulting tier, provider,
model, and rule without making a model call; without it, the router invokes
the calibrated Claude tier and prints the response, cost, and duration.

### Interactive chat (`llm-chat`)

```bash
uv run llm-chat
```

Authenticates each configured provider once at startup (refuses to start if
Codex is the only authenticated provider, since every tier currently maps to
Claude — see `CLAUDE.md`), then classifies every message you type
independently via `route()` and delivers it into a single persistent
`tmux`-backed terminal session (`terminal.py`) running the routed `claude`
CLI — not a new process per message. The first message creates that session
and opens a visible terminal window attached to it (`tmux attach`); every
message after, including a tier change, is injected into the same
already-running process via `tmux send-keys`, exactly as if typed by hand. A
tier change mid-conversation sends the provider CLI's own `/model <name>`
first. All messages in one session share a `session_id`, so Claude-side
history/tools continue across turns even as different messages land on
different tiers/models. Full tool use and your `CLAUDE.md`/hooks run for
real (not a stripped eval-harness call) at real per-call cost
(~$0.07-0.30/call). See `docs/llm-chat.md` for the full mechanics and known
limitations (e.g. switching *provider*, not just model, mid-session isn't
supported).

#### Running `llm-chat` from anywhere (not just inside this repo)

`uv run llm-chat` only resolves the entrypoint when your shell's `cwd` is
inside this repo (or `--project` points at it). To call `llm-chat`
regardless of which directory you're standing in — including from an
unrelated, non-Python project (a Go module, a Node app, your home
directory) — install it as a global `uv` tool instead:

```bash
cd /path/to/llm-task-router
uv tool install --editable .
```

This is `uv`'s equivalent of `pipx install` / `go install` / `npm install
-g`: a one-time install that drops thin shim executables (`llm-chat`,
`llm-route`, `llm-route-audit-tier2`, `llm-route-shadow-report`) into
`~/.local/bin` — `uv`'s own default shim directory, already on `$PATH` if
`uv` itself is runnable from your shell. `--editable` means source edits in
this repo take effect immediately, no reinstall step. Anyone cloning this
repo runs the same command from repo root to get the same result — no
machine-specific PATH edit to redo per clone.

**Worth knowing before you do this:** `terminal.create_session()` passes
`os.getcwd()` explicitly as the tmux pane's working directory, so whatever
directory you ran `llm-chat` from is what the routed `claude` session
operates on — a message you type while `cwd` is some other project operates
on *that* project's files, not this repo's. Unlike the old direct `claude
-p` calls, the tmux session runs `claude` with a real interactive pty and no
`--permission-mode` override, so it behaves exactly like running `claude`
yourself: the normal "do you trust this folder?" prompt shows the first time
you use it somewhere new, and tool-use approval prompts apply as usual
inside that session.

### Auditing tier 2 and shadow-eval divergence

Two more installed console scripts, both read-only (they print a report,
never write or auto-correct anything):

```bash
# Re-validates tier 2's AGREEMENT_THRESHOLD against the live routing_examples
# table (leave-one-out check), and flags any llm_fallback-sourced row whose
# neighbors now disagree with its stored label.
uv run llm-route-audit-tier2
uv run llm-route-audit-tier2 --label-column task_type
uv run llm-route-audit-tier2 --judge-flagged   # one real LLM call per flagged row

# Compares tier 2's real routing decisions against what tier 1 alone would
# have chosen on the same live traffic (routing_decisions table).
uv run llm-route-shadow-report
```

Both need `DATABASE_URL` set and enough logged/seeded rows to be meaningful
(`audit_tier2` reports "not enough rows yet" per label column below
`NEIGHBOR_K`; `shadow_report` reports "nothing to report" with zero
shadow-scored rows). Neither has a special exit-code contract for a "bad"
verdict — only a genuine crash (unset `DATABASE_URL`, connection failure)
exits nonzero, since a "consider raising the threshold" or high-divergence
reading is a judgment call for whoever reads the log, not something a
scheduler should page on.

### Scheduling the audits

Both scripts are meant to run daily via cron/launchd/Task Scheduler so
`routing_decisions` accumulates enough data to judge tier-1-vs-tier-2
divergence over time. See `docs/scheduling-audits.md` for ready-to-adapt
cron, launchd `.plist`, and Windows `schtasks` recipes.

### Adding a provider

See the `add-provider` skill (`.claude/skills/add-provider/SKILL.md`) for the
full workflow. Don't add a new entry to `tiers.TIER_MODELS` or
`classifier.TYPE_DOMAIN_GRID` without real calibration data from
`llm-eval-harness`'s `calibrate-tier` skill first — see
`docs/auth-and-providers.md`, "Adding a provider."

## Status

Implements a two-tier classifier cascade as of 2026-07-27 (the original
three-tier design was revised during review - see `docs/classifier.md`):
tier 1's heuristic grid, and tier 2, a
continuous-learning classifier (local embeddings + a pgvector-backed store
of labeled examples, with a cheap-LLM fallback that writes its own answer
back). Claude + Codex provider adapters and a Claude-only tier map are also
in place. Shadow evaluation and any non-Claude tier entries remain pillars
on paper, not code yet.

The tier-1 classifier's own accuracy is unmeasured: `classify_description()`
is a keyword-bag heuristic, not something benchmarked against a labeled set
the way `tiers.py`'s model quality floors are (see
[`llm-eval-harness`](../llm-eval-harness)). Treat task-type/domain inference
as a best-effort default, not a calibrated component, until it's evaluated
the same way. Combined with the small, hand-authored calibration set behind
`tiers.py` (15 `bug_triage` / 17 `code_gen` cases, no held-out split or
confidence intervals), this project is a prototype demonstrating the
architecture, not a production cost/quality system yet.
