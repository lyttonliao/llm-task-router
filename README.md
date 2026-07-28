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

## Quick start

Requirements:

- Python 3.14+
- [`uv`](https://docs.astral.sh/uv/) (recommended)
- An authenticated `claude` CLI for routed runs

```bash
uv sync --group dev

# Run the unit tests
uv run pytest -q

# Inspect a routing decision without calling a model
uv run python -m llm_task_router route \
  "Design a fault-tolerant deployment strategy for our Kubernetes cluster" \
  --dry-run

# Route and execute a task with the currently calibrated provider/model tier
uv run python -m llm_task_router route \
  "Summarize these release notes for a frontend team" \
  --type summarization --domain frontend
```

The current tier-1 router infers task type and domain from description
keywords, then selects a low, medium, or high escalation bias. Supply
`--type` and/or `--domain` to override either inference. Ambiguous task types
fall back to the high-escalation architecture category rather than silently
routing to a cheap model. `--dry-run` shows the resulting tier, provider,
model, and rule without making a model call. Without it, the router invokes
the calibrated Claude tier.

## Status

Implements a two-tier classifier cascade as of 2026-07-27 (the original
three-tier design was revised during review - see `CLAUDE.md`, "The
classifier is a two-tier cascade"): tier 1's heuristic grid, and tier 2, a
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
