# Drift auditing and shadow evaluation

## Drift auditing: logging every routing decision

Built 2026-07-28. Distinct from `llm-eval-harness`'s `calibrate-tier` skill
(offline calibration against a fixed 98-case golden set, never touches live
traffic) — before this, `route()` discarded every decision the instant it
was made, so there was no logged traffic to re-validate `AGREEMENT_THRESHOLD`
against or catch a bad `llm_fallback` write-back with.

**`decision_log.py`** (second SQL module, alongside `vector_store.py`) writes
one row per `route()` call to `routing_decisions`: resolved task_type/domain
and which mechanism produced each (`provided`/`inferred`/`tier2_nn`/
`tier2_llm_fallback`/`fallback` for task_type; `keyword`/`tier2_nn`/
`tier2_llm_fallback`/`unavailable` for high-stakes), plus the final
bias/tier/model/reason. `embedding` is null when the heuristic resolved
everything without ever calling `embeddings.embed()`. Wrapped in
`try/except Exception: pass` in `route()` — a logging failure degrades to
"no audit trail for this call," never a broken route. `tests/conftest.py`
autouse-patches `log_decision` to a no-op for the whole suite; a handful of
`test_router.py` tests override it explicitly to assert logged fields on
representative paths.

**`audit_tier2.py`** re-runs the same leave-one-out methodology against the
*live* `routing_examples` table instead of the frozen seed, printing the
same kind of agreement curve plus a "threshold holds / consider raising"
verdict, and flags `source='llm_fallback'` rows whose neighbors now
confidently disagree with the stored label — manual review only, never
auto-corrected. An opt-in `--judge-flagged` pass gets one cheap second LLM
opinion per flagged row. Verified against the real local store: 101
`task_type` rows produced a live curve (73.3%/79.0%/88.2%/100.0%) tracking
the seed-only numbers, verdict "holds," 0 suspect rows; `is_high_stakes` (3
rows) correctly reported "not enough rows yet." Now an installable console
script, cron/launchd-schedulable — see the `schedule-audits` skill.

**Gotcha worth restating**: `vector_store.all_labeled_examples()`'s first
draft called `list(row[2])` on the embedding column, assuming an iterable —
a `register_vector()`-registered connection actually returns a
`pgvector.Vector` object there, not iterable; failed with a real `TypeError`
against the real store. Fixed with `Vector.to_list()`. Any future module
reading embeddings back out of Postgres should expect the same and verify
against a real query first.

## Shadow evaluation: live dual-routing without doubling cost

Built 2026-07-28. Distinct from both drift auditing above (re-validates
tier 2's *own* NN/LLM resolutions against themselves) and
`llm-eval-harness`'s offline calibration (a fixed golden set) — this
compares tier 1 against tier 2 on live traffic, the actual open question
blocking tier 1's eventual deprecation.

**Not a real second model invocation.** Actually calling a second model on
every request to compare outputs would double real per-call cost, which
contradicts this repo's cost discipline everywhere else. Instead,
`router._shadow_tier1_only_decision()` computes a second **routing
decision**, not a second response: what tier the request would have gotten
if tier 2 had never been consulted, using only data already computed for the
real decision (`classification`, the grid, keyword high-stakes matching) —
zero extra embedding calls, zero extra LLM calls, zero extra Postgres round
trips.

**The no-signal case is a deliberate approximation, not a full replay.** The
real no-signal branch (tier 3) makes a genuine LLM call; re-invoking it a
second time just to populate a shadow column would reintroduce the same
double-cost problem. The shadow reading always hedges the no-signal case
straight to `mid` instead — the same static fallback tier 3 itself uses on
its own failure, a defensible stand-in for "would have escalated," not a
fabricated number.

**`route()` logs both, acts on only one.** Every call computes
`shadow_bias`/`shadow_tier`/`shadow_reason` alongside the real decision and
passes both to `decision_log.log_decision()` — the shadow fields never touch
`provider`/`model`/the returned `RouteDecision`, purely an audit-row column.
`decision_log.py` gained `fetch_decisions()` (a `LoggedDecision` NamedTuple
per row) to read the table back in batch, the same shape
`vector_store.all_labeled_examples()` already established.

**`report_shadow_divergence.py`** reads the live table and reports:
divergence rate (real tier vs. shadow tier), direction (`tiers.TIER_ORDER`
exists specifically for this — escalated = tier 2 routed more expensively
than tier 1 alone would have; de-escalated = the reverse, e.g. tier 2
correctly resolving a task_type the heuristic couldn't place), and which
`tier2_classifier` axis drove each divergence. Deliberately does not judge
which side was *right* — that needs a ground-truth label or a judge call
this script doesn't make; it only measures how often and in which direction
the two disagree, the input needed for an eventual tier-1 deprecation call.

**Migration note**: `db/schema.sql`'s `CREATE TABLE routing_decisions` now
includes `shadow_bias`/`shadow_tier`/`shadow_reason` (nullable, for
migration compatibility with rows logged before this existed). An existing
live database needs the three commented `ALTER TABLE` statements at the
bottom of that file run manually via `psql` — not applied automatically,
same one-time/not-idempotent discipline the rest of that file documents.
