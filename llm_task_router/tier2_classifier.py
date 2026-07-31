"""Tier 2 of the router's classifier: a continuous-learning primitive, not a
static model. Both public functions below apply the same shape to two
different questions ("what task_type is this?", "is this genuinely
high-stakes?"): check labeled neighbors in `routing_examples` first; if
they don't confidently agree, ask one cheap LLM call and write its answer
back as a new labeled row, so the same question never needs an LLM call
again. See /Users/lliao/.claude/plans/i-believe-for-now-smooth-cook.md for
the full design this implements.

Only LLM-fallback resolutions get written back, not NN-confident ones - a
matching neighbor already covers that region of the embedding space, so
writing it again would just add near-duplicate rows without adding new
information (see the plan's "Open judgment calls" section; revisit only if
coverage growth turns out too slow in practice).

The LLM-call discipline mirrors router.py's `_classify_via_llm` exactly:
haiku, `disable_tools=True` + an explicit system_prompt (stripped-cost
bracket, not llm-chat's full-functionality one - see claude_cli.invoke's own
docstring), no session_id/on_event (internal, stateless, one-shot), and
`None` on ANY failure - provider error, unparseable text, or a raised
exception (claude_cli.check_auth's subprocess.run has no FileNotFoundError
guard, so invoke() is not exception-safe). Callers treat `None` the same way
router.py already does elsewhere: "tier 2 unavailable," not a crash.

NN confidence thresholds below are placeholders pending real usage data -
see the module-level constants' comments for how they were chosen and what
would justify changing them.
"""

from collections import Counter
from typing import NamedTuple

from llm_task_router import vector_store
from llm_task_router.providers import claude_cli
from llm_task_router.schema import TASK_TYPES

NEIGHBOR_K = 5

# Tuned from a real leave-one-out check against the actual seeded store (98
# llm-eval-harness cases, all-MiniLM-L6-v2 embeddings) - not guessed. Two
# findings from that check, both worth recording since they overturned the
# first (unvalidated) design:
#
# 1. Raw cosine similarity does NOT cleanly separate same-task_type from
#    different-task_type neighbors at this data/model scale - same-label
#    pairs measured median ~0.32 similarity, cross-label median ~0.29,
#    heavily overlapping. A hard similarity floor (the original design here)
#    would reject most true matches. This echoes llm-eval-harness's own
#    documented finding that off-the-shelf embeddings don't cleanly separate
#    certain problems at small data scale - don't re-add a similarity floor
#    without new evidence it helps.
# 2. Relative agreement among the k nearest neighbors DOES carry real signal,
#    despite (1): leave-one-out majority-of-5 vote scored 72.4% accuracy
#    unconditionally (98 samples, 7 classes, chance ~14%), and accuracy rises
#    monotonically with how many of the 5 agree - 78.2% accuracy at >=3/5
#    agreement (80% coverage), 87.1% at >=4/5 (32% coverage), 100% at 5/5
#    (8% coverage, too rare to rely on alone). AGREEMENT_THRESHOLD=0.8 (i.e.
#    >=4/5) was chosen from that curve: real coverage now, but erring toward
#    the LLM fallback (proven far more accurate) rather than trusting a
#    borderline NN vote - consistent with this project's asymmetric-risk
#    default (escalate under uncertainty, not default-to-cheap-and-wrong).
#    Revisit as real (non-seed) query volume accumulates and this can be
#    re-measured on live traffic instead of the seed set alone.
AGREEMENT_THRESHOLD = 0.8  # fraction of the k nearest neighbors that must share the majority label

_TASK_TYPE_LLM_PROMPT = """Classify this task request into exactly one category, choosing the single best fit:

TRIAGE - diagnosing/investigating a bug, error, or regression.
CODE_GEN - writing new code: a function, class, endpoint, or similar.
SUMMARIZATION - condensing existing text/notes into a shorter form.
MULTI_STEP - planning a multi-step process: a migration, rollout, or workflow.
CODE_REVIEW - reviewing existing code for issues.
REFACTOR - restructuring existing code without changing its behavior.
ARCHITECTURE - designing a system or making an architectural/strategic decision.

Task request:
\"\"\"{description}\"\"\"

Respond with exactly one word, uppercase, from the list above (TRIAGE, CODE_GEN, SUMMARIZATION, MULTI_STEP, CODE_REVIEW, REFACTOR, or ARCHITECTURE). No explanation, no punctuation, nothing else."""

_TASK_TYPE_LLM_SYSTEM_PROMPT = "You are a task-category classifier. Respond with exactly one word and nothing else."

_HIGH_STAKES_LLM_PROMPT = """A task-difficulty heuristic flagged this request as potentially
high-stakes (production impact, security/compliance exposure, irreversible
consequences, or meaningful scale) but found no confirming keyword. Judge
for yourself whether it genuinely involves real production impact,
security/compliance/regulatory exposure, irreversible consequences, or
operation at meaningful scale - not just topical similarity to those
concerns.

Task request:
\"\"\"{description}\"\"\"

Respond with exactly one word: YES or NO. No explanation, no punctuation, nothing else."""

_HIGH_STAKES_LLM_SYSTEM_PROMPT = "You are a high-stakes-consequence classifier. Respond with exactly one word and nothing else."


class Tier2Resolution(NamedTuple):
    task_type: str
    source: str  # "nn" | "llm_fallback"


class HighStakesResolution(NamedTuple):
    is_high_stakes: bool
    source: str  # "nn" | "llm_fallback"


def _nn_majority(matches: list[vector_store.NeighborMatch]) -> str | bool | None:
    """Returns the majority label among the given neighbors if enough of them
    agree, else None (not confident). No similarity floor - see
    AGREEMENT_THRESHOLD's comment above for why raw similarity isn't a
    useful gate at this data/model scale; agreement fraction is.

    Requires a full NEIGHBOR_K matches before trusting a majority at all -
    fewer than that means the store barely has labeled examples yet for this
    axis (e.g. is_high_stakes starts completely empty at seed time), and a
    majority among 1-2 points isn't a real signal. Always falls through to
    the LLM during that early regime, which is the correct behavior, not a
    gap - it's exactly how the store gains its first labels for that axis."""
    if len(matches) < NEIGHBOR_K:
        return None
    label, count = Counter(m.label for m in matches).most_common(1)[0]
    if count / len(matches) >= AGREEMENT_THRESHOLD:
        return label
    return None


def _classify_task_type_via_llm(description: str) -> str | None:
    try:
        result = claude_cli.invoke(
            _TASK_TYPE_LLM_PROMPT.format(description=description),
            "haiku",
            system_prompt=_TASK_TYPE_LLM_SYSTEM_PROMPT,
            disable_tools=True,
        )
    except Exception:
        return None
    if result.error:
        return None
    text = result.text.strip().upper()
    for task_type in TASK_TYPES:
        if task_type.upper() in text:
            return task_type
    return None


def _corroborate_high_stakes_via_llm(description: str) -> bool | None:
    try:
        result = claude_cli.invoke(
            _HIGH_STAKES_LLM_PROMPT.format(description=description),
            "haiku",
            system_prompt=_HIGH_STAKES_LLM_SYSTEM_PROMPT,
            disable_tools=True,
        )
    except Exception:
        return None
    if result.error:
        return None
    text = result.text.strip().upper()
    if "YES" in text:
        return True
    if "NO" in text:
        return False
    return None


def resolve_task_type(description: str, embedding: list[float]) -> Tier2Resolution | None:
    """NN lookup against task_type-labeled neighbors first; falls through to
    one cheap LLM call, writing its answer back, only when neighbors don't
    confidently agree. Returns None only when both the NN lookup is
    unconfident AND the LLM fallback is unavailable/unparseable - callers
    treat that the same way as any other tier-2-unavailable case."""
    try:
        matches = vector_store.nearest_neighbors(embedding, "task_type", k=NEIGHBOR_K)
    except Exception:
        matches = []
    majority = _nn_majority(matches)
    if majority is not None:
        return Tier2Resolution(task_type=majority, source="nn")

    label = _classify_task_type_via_llm(description)
    if label is None:
        return None
    try:
        vector_store.insert_example(description, embedding, task_type=label, is_high_stakes=None, source="llm_fallback")
    except Exception:
        pass  # write-back is best-effort - a store failure shouldn't invalidate a resolution we already have
    return Tier2Resolution(task_type=label, source="llm_fallback")


def resolve_high_stakes(
    description: str, embedding: list[float], *, task_type: str | None = None
) -> HighStakesResolution | None:
    """Same shape as resolve_task_type, against is_high_stakes-labeled
    neighbors. `task_type` is optional context to include in the write-back
    row when the caller already resolved it earlier in the same request -
    a richer row (both labels from one description) is strictly more useful
    than a high-stakes-only one, and costs nothing extra to include. Returns
    None on NN-unconfident + LLM-unavailable, same discipline as
    resolve_task_type. Carries a `source` ("nn" | "llm_fallback"), the same
    symmetry resolve_task_type already had - a drift-auditing consumer needs
    to know which mechanism produced a live resolution, not just the
    resolution itself."""
    try:
        matches = vector_store.nearest_neighbors(embedding, "is_high_stakes", k=NEIGHBOR_K)
    except Exception:
        matches = []
    majority = _nn_majority(matches)
    if majority is not None:
        return HighStakesResolution(is_high_stakes=majority, source="nn")

    is_high_stakes = _corroborate_high_stakes_via_llm(description)
    if is_high_stakes is None:
        return None
    try:
        vector_store.insert_example(
            description, embedding, task_type=task_type, is_high_stakes=is_high_stakes, source="llm_fallback"
        )
    except Exception:
        pass  # write-back is best-effort - a store failure shouldn't invalidate a resolution we already have
    return HighStakesResolution(is_high_stakes=is_high_stakes, source="llm_fallback")
