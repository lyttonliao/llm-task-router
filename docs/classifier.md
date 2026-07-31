# The classifier: tier-1 heuristic + tier-2 continuous-learning cascade

Originally scoped as three tiers (heuristic → a trained model cold-started
from `llm-eval-harness` golden labels → cheap-LLM fallback). A 2026-07-27
design review collapsed the last two into one mechanism instead of building
the intermediate static model. `classifier.py`'s heuristic grid remains tier 1,
unchanged, the free/zero-latency first pass; expected to be deprecated later
(not yet) once tier 2 earns enough coverage to be trusted alone.

## Tier 1: Heuristic grid

**Escalation to flagship is precision-first, not recall-first** (reversed
2026-07-27 from an original "escalate under uncertainty" default).
`router.route()` requires both a grid cell mapped to `H` and — unless the
caller explicitly provided `task_type` (a trusted override) — a genuine
high-stakes signal in the description itself
(`classifier.has_high_stakes_signal()`, matching a deliberately narrow
production/security/compliance/irreversibility/scale vocabulary). A
shape-only match (bare "design"/"strategy" firing on a trivial UI question
as readily as a real one) no longer earns flagship alone. An inferred `H`
without keyword corroboration falls through to tier 2's
`resolve_high_stakes()` rather than capping unconditionally at `mid` — a
real escalation path for hard tasks that don't use recognized vocabulary
(e.g. "migrate the k8s cluster to a new region"), confirmed against a real
run.

**No-signal fallback**: a fully-unresolved description used to inherit
`classify_description()`'s `task_type="architecture"` safety placeholder and
silently escalate via architecture's uniform-`H` row — a real incident:
"reply with exactly the word: pong" routed to opus, ~$0.18 to repeat one
word. Both axes fully unresolved now hedges to `mid` instead of consulting
the grid (pinned by
`tests/test_router.py::test_route_hedges_to_mid_when_no_keyword_signal_and_tier3_is_unavailable`).

**Tier 3 (`_classify_via_llm()`)** handles that no-signal band: one stripped
haiku call judges difficulty/consequence/ambiguity directly, rather than a
keyword list (tried, rejected — doesn't generalize) or embeddings (rejected
at the time on `llm-eval-harness`'s prior finding that they don't cleanly
separate this kind of distinction at small scale — tier 2 below revisited
that finding for a different question with a different result). Reuses
`claude_cli.invoke(disable_tools=True, system_prompt=...)`, landing in
`eval_harness`'s ~$0.003-0.005/call bracket instead of `llm-chat`'s
~$0.07-0.30/call. Returns `None` on any failure; `route()` falls back to the
`mid` hedge.

## Tier 2: Continuous-learning classifier

Built 2026-07-27. Not a static trained model — embeddings + a pgvector-backed
store of labeled examples (`routing_examples`), falling through to one cheap
LLM call for whatever the store can't confidently answer, which then writes
its own answer back as a new labeled row. `router.route()` calls the same
primitive (`tier2_classifier.py`) against two different questions — task_type
resolution and high-stakes corroboration — not two separate mechanisms.

**Modules**: `embeddings.py` (`embed(text) -> list[float]`, lazy-singleton
`sentence-transformers all-MiniLM-L6-v2`, 384-dim) · `vector_store.py`
(Postgres/pgvector client, `DATABASE_URL`-configured) · `tier2_classifier.py`
(`resolve_task_type()` / `resolve_high_stakes()`, the orchestration
`router.py` calls). `scripts/seed_vector_store.py` cold-started the store
from `llm-eval-harness`'s 98 golden cases (task_type only — no
`is_high_stakes` ground truth exists in that data, so that column starts
empty and fills in only as real `needs_corroboration` requests resolve
through tier 2 over time).

**Write-back is LLM-fallback-only, never on an NN-confident hit** — a
matching neighbor already covers that region of the embedding space;
re-inserting it would add a near-duplicate with no new information. This is
what makes it "continuous learning" rather than a one-time cold start.

**`AGREEMENT_THRESHOLD = 0.8` (≥4/5 of `NEIGHBOR_K=5` nearest neighbors must
share a label) came from a real leave-one-out check against the seeded
store, not a guess, and overturned the original design en route:**
1. Raw cosine similarity does **not** cleanly separate same-label from
   different-label neighbors at this scale (same-label median ~0.32,
   cross-label median ~0.29 — heavily overlapping). A hard similarity floor
   (the first draft) would have rejected most true matches — the same shape
   of finding `llm-eval-harness`'s CLAUDE.md documents for a structurally
   similar problem.
2. Relative *agreement* among the k neighbors carries real signal despite
   (1): 72.4% accuracy unconditionally (98 samples, 7 classes, chance ~14%),
   rising monotonically — 78.2% at ≥3/5 (80% coverage), 87.1% at ≥4/5 (32%
   coverage), 100% at 5/5 (8% coverage, too rare alone). Fewer than
   `NEIGHBOR_K` neighbors never counts as confident regardless of agreement
   fraction (true of `is_high_stakes` early on, since it starts empty).

**`needs_corroboration` no longer caps to `mid` unconditionally on a
keyword-negative miss** — it falls through to `resolve_high_stakes()`
(reusing the embedding already computed for task_type when there is one).
`True` escalates to flagship for real; `False` or `None` (tier 2
unavailable) both still cap to `mid` — tier 2 only ever adds a path to a
*correct* escalation, never removes the safety cap. Both resolver functions
return `None` on any failure — `router.py` treats tier-2-unavailable as
"fall back to heuristic-only," never a crash.

**`resolve_high_stakes()` gained a `source` field, 2026-07-28**, symmetric
with `resolve_task_type()`'s `Tier2Resolution(task_type, source)` — needed
by drift auditing to tell whether a resolution came from the NN vote
or the LLM fallback.

**Both resolvers now wrap their `vector_store.nearest_neighbors()`/
`insert_example()` calls in `try/except Exception`** — a transient Postgres
error previously propagated straight up through `route()` (no
`try/except Exception: pass` at that layer the way `decision_log.py` has
for logging, see `drift-and-shadow.md`), turning a store hiccup into a
crashed route instead of a graceful "tier 2 unavailable" fallback. A failed
NN lookup now falls through to the LLM fallback exactly as if there simply
weren't enough neighbors yet; a failed write-back is treated as
best-effort — the resolution the caller already has in hand is still
returned, it just doesn't get persisted for next time.
