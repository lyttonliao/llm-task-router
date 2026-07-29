"""Orchestrates classify -> resolve tier -> invoke provider.

The tier-1 heuristic grid (classifier.py) drives routing for every case with
real keyword signal. Tier 2 - a continuous-learning classifier (embeddings +
a pgvector-backed store of labeled examples, falling through to a cheap-LLM
call that writes its own answer back) - was added 2026-07-27; see
tier2_classifier.py's module docstring for the full design. This module's
job is only to call it at the right point and use its answer, not to know
how it works internally (see route()'s docstring for exactly where it's
called). Originally this project was scoped as a three-tier cascade
(heuristic -> trained model -> cheap-LLM fallback); that direction changed
during design review to two tiers, with tier 2 absorbing what would have
been both the "trained model" and "cheap-LLM fallback" tiers into one
lookup-then-ask-and-remember mechanism.
"""

from collections.abc import Callable

from llm_task_router import decision_log, embeddings, tier2_classifier
from llm_task_router.classifier import (
    classify,
    classify_description,
    has_high_stakes_signal,
)
from llm_task_router.providers import claude_cli, codex_cli
from llm_task_router.schema import ProviderResult, RouteDecision, TaskRequest
from llm_task_router.tiers import TIER_BIAS, TIER_MODELS

PROVIDERS = {
    "claude": claude_cli,
    "codex": codex_cli,
}

_LLM_CLASSIFY_PROMPT = """You are classifying a task request by its actual difficulty, consequence, and ambiguity - not by surface keywords or phrasing style. Choose exactly one tier:

CHEAP - low difficulty, low consequence, low ambiguity. Mechanical, trivially verified, nothing goes wrong if it's slightly off. Example: "repeat this word back to me", "what's 2+2", basic formatting.
MID - real reasoning or judgment required, but moderate stakes and recoverable mistakes. Example: writing a typical function, summarizing a document, debugging an ordinary bug.
FLAGSHIP - deep reasoning required, or genuine real-world stakes (production, security, compliance, irreversibility, scale), or wide-open tradeoffs with no single right answer. Example: designing a multi-region disaster-recovery architecture for a payments system.

Task request:
\"\"\"{description}\"\"\"

Respond with exactly one word: CHEAP, MID, or FLAGSHIP. No explanation, no punctuation, nothing else."""

_LLM_CLASSIFY_SYSTEM_PROMPT = "You are a task-difficulty classifier. Respond with exactly one word and nothing else."


def _classify_via_llm(description: str) -> str | None:
    """Tier 3 of the cascade (see module docstring): one cheap, stripped
    haiku call for the narrow no-signal band where tier-1's grid has
    nothing to go on (classifier.py's keyword tables matched zero keywords
    on either axis - not "unclear which of several types," genuinely no
    signal at all). Internal/stateless/one-shot - no session_id (unrelated
    to the caller's own conversation), no on_event (nothing to stream),
    disable_tools=True + an explicit system_prompt (see claude_cli.py) so
    this rides the same stripped cost profile eval_harness's calls get
    (~$0.003-0.005/call there), not llm-chat's full-functionality one
    (~$0.07-0.30/call) - reusing the adapter unchanged here would have made
    this "cheap" fallback cost about as much as the misroute it prevents.

    Returns a bias letter ("L"/"M"/"H") or None on ANY failure - provider
    error, unparseable text, or an unanticipated exception (invoke() is not
    guaranteed exception-free: check_auth()'s subprocess.run has no
    FileNotFoundError guard, so a missing `claude` binary would otherwise
    propagate here uncaught). Callers must treat None as "fall back to the
    tier-1 hedge" - same never-let-one-fault-kill-the-flow discipline
    repl.py's chat_loop() already applies to route_and_run().
    """
    try:
        result = claude_cli.invoke(
            _LLM_CLASSIFY_PROMPT.format(description=description),
            "haiku",
            system_prompt=_LLM_CLASSIFY_SYSTEM_PROMPT,
            disable_tools=True,
        )
    except Exception:
        return None

    if result.error:
        return None

    text = result.text.strip().upper()
    if "FLAGSHIP" in text:
        return "H"
    if "MID" in text:
        return "M"
    if "CHEAP" in text:
        return "L"
    return None


def _shadow_tier1_only_decision(
    classification, description: str
) -> tuple[str, str, str]:
    """What route() would decide for this exact request if tier 2 had never
    been built - the pure heuristic-grid router, including the no-signal and
    corroboration safety nets (both added independently of tier 2 - see
    route()'s docstring - so they belong in this counterfactual too), but
    with zero tier2_classifier calls. Logged alongside the real decision (see
    decision_log.py) for shadow evaluation: measuring how often, and in which
    direction, tier 2 actually changes the outcome on live traffic - the data
    this repo's CLAUDE.md says is needed before tier 1 can ever be
    deprecated, not a guess.

    Deliberately does NOT re-invoke tier 3's no-signal LLM call to populate
    this - that would mean a second real LLM call solely for a shadow log
    column, which contradicts every other cost discipline in this module
    (see _classify_via_llm's docstring). The no-signal case here always
    hedges to the static "M" bias instead, the same fallback tier 3 itself
    uses on failure - a defensible approximation of "would have escalated"
    that costs nothing to compute, since every input (task_type_source,
    domain_source, the grid, keyword high-stakes matching) is already
    available on `classification` for free."""
    if (
        classification.task_type_source == "fallback"
        and classification.domain_source == "fallback"
    ):
        return (
            "M",
            TIER_BIAS["M"],
            "shadow (no tier 2): no keyword signal on either axis, hedged to mid",
        )

    grid_bias = classify(classification.task_type, classification.domain)
    needs_corroboration = (
        grid_bias == "H" and classification.task_type_source != "provided"
    )
    if needs_corroboration and not has_high_stakes_signal(description):
        reason = (
            f"shadow (no tier 2): heuristic grid said H for {classification.task_type} x "
            f"{classification.domain} (type {classification.task_type_source}), no keyword corroboration - "
            "capped at mid"
        )
        return "M", TIER_BIAS["M"], reason

    reason = (
        f"shadow (no tier 2): heuristic grid {classification.task_type} x {classification.domain} -> "
        f"{grid_bias} (type {classification.task_type_source}, domain {classification.domain_source})"
    )
    return grid_bias, TIER_BIAS[grid_bias], reason


def route(request: TaskRequest) -> RouteDecision:
    """No-signal fallback, added 2026-07-27: classify_description() labels a
    fully-unrecognized description task_type="architecture" (see its own
    docstring) purely so classify() always has a valid grid key - that label
    is NOT evidence the task is actually architecture-shaped. Feeding it
    straight into the grid used to mean "we have zero idea what this is"
    silently became "escalate to flagship", because architecture's row is
    uniform H - confirmed in practice by a trivial toy prompt ("reply with
    exactly the word: pong", no keyword overlap on either axis) routing to
    opus at real cost. So: when BOTH axes are fully unresolved
    (task_type_source and domain_source both "fallback"), this hedges at mid
    instead of consulting the grid.

    High-stakes gate on flagship, corrected 2026-07-27 (superseding the
    "escalate under uncertainty" framing above - see classifier.py's
    IMPACT_KEYWORDS docstring): the no-signal fix alone was still too broad.
    An *inferred* type/domain match reaching the grid's H cells doesn't mean
    the request is actually difficult, consequential, or ambiguous - it
    means a shape keyword happened to appear (architecture's "design"
    matches a trivial UI question exactly as readily as a real
    multi-region-failover one). Flagship must now be earned twice: a
    type/domain combo that maps to H, AND (unless the caller explicitly
    provided that task_type - an explicit override is trusted as a
    deliberate human/caller judgment, not second-guessed) a genuine
    high-stakes signal in the description itself
    (classifier.has_high_stakes_signal() - production/security/compliance/
    irreversibility/scale vocabulary). An inferred H without that keyword
    corroboration used to be capped straight to mid; as of the tier-2
    extension below, it instead falls through to tier 2's own
    resolve_high_stakes() - a genuinely hard task that doesn't use recognized
    high-stakes vocabulary is no longer permanently underrouted to mid on
    that basis alone, closing the gap this paragraph originally accepted as
    a bet on tier-1 heuristics' limits.

    Tier 2 extension of the corroboration gate, added 2026-07-27: when the
    keyword gate above doesn't corroborate an inferred H, this now calls
    tier2_classifier.resolve_high_stakes() (same NN-lookup-then-cheap-LLM
    primitive as task_type resolution, reusing this request's embedding
    rather than computing a second one) instead of capping to mid
    unconditionally. True -> escalate to flagship for real; False or
    None (tier 2 unavailable) both still cap to mid, same as the old
    unconditional behavior - tier 2 only ever adds a path to a *correct*
    escalation, it never removes the existing safety cap.

    Tier 3 for the no-signal band, added 2026-07-27: a keyword-list approach
    to distinguishing "repeat this word: hello" (trivial, should be cheap)
    from "help with this request" (genuinely ambiguous, the mid hedge above
    is correct) was tried and rejected - it doesn't generalize. This repo's
    zero-third-party-dependency rule and llm-eval-harness's finding that
    embeddings don't cleanly separate soft semantic distinctions both applied
    at the time - see tier2_classifier.py's module docstring for how tier 2
    later revisited both of those with real data instead of settling for the
    no-signal band alone. This branch makes one cheap, stripped haiku call
    (_classify_via_llm()) asking the model to judge the task's actual
    difficulty/consequence/ambiguity directly. Falls back to the mid hedge on
    any failure (see _classify_via_llm()'s docstring).

    Tier 2, added 2026-07-27 (before this no-signal branch runs): when the
    heuristic grid found zero task_type keyword signal at all
    (classification.task_type_source == "fallback"), tier 2 gets a chance to
    resolve it before falling all the way through to the no-signal
    difficulty-judgment call above - a description that trips a domain
    keyword ("kubernetes") but zero TYPE_KEYWORDS used to be silently
    mislabeled task_type="architecture" (classify_description()'s safety
    placeholder) and pushed straight into architecture's uniform-H grid row;
    tier 2 can now correct that placeholder first. Only task_type is in
    scope for tier 2 - domain stays on the keyword heuristic indefinitely
    (see tier2_classifier.py's docstring for why). A successful resolution
    marks resolved_task_type_source="tier2", which the no_signal check and
    needs_corroboration gate below both treat as "we have a real signal now,"
    same as "inferred" - not as another flavor of "fallback".

    Decision logging, added for drift auditing: every call writes one row to
    `routing_decisions` via decision_log.log_decision() - which mechanism
    resolved task_type/high-stakes (heuristic/tier2_nn/tier2_llm_fallback/
    keyword/unavailable), not just the final tier. This exists because
    nothing previously recorded what route() actually decided on live
    traffic - `tier2_classifier.AGREEMENT_THRESHOLD` was tuned only against
    the 98-row seed set, with no way to re-validate it against real query
    behavior (see decision_log.py and scripts/audit_tier2.py). Wrapped in
    try/except: a logging failure must never break routing, same discipline
    tier2_classifier's own Postgres/LLM calls already follow.

    Shadow evaluation, added for live dual-routing (closing this repo's last
    documented rough edge - "no live dual-routing, a request never goes to
    two tiers to compare"): every call also computes, and logs alongside the
    real decision, what tier-1-alone (_shadow_tier1_only_decision(), no tier
    2 consulted at all) would have decided for the same request - never
    acted on, never affecting `provider`/`model`/the returned RouteDecision,
    purely a second column on the same audit row. This is genuinely free:
    the shadow function only reads `classification`, which is computed once
    above and never mutated by the tier-2 branches below (they reassign
    local `resolved_*` variables, not `classification` itself), so there is
    no second embedding call, no second LLM call, no second Postgres
    round-trip - just a second reading of data already in hand. See
    scripts/report_shadow_divergence.py for what this data is for: measuring
    how often and in which direction tier 2 changes the outcome, ahead of
    ever using that to decide whether tier 1 can be deprecated.
    """
    classification = classify_description(
        request.description, request.task_type, request.domain
    )
    shadow_bias, shadow_tier, shadow_reason = _shadow_tier1_only_decision(
        classification, request.description
    )

    resolved_task_type = classification.task_type
    resolved_task_type_source = classification.task_type_source
    embedding: list[float] | None = None
    tier2_task_type_resolution: tier2_classifier.Tier2Resolution | None = None
    if classification.task_type_source == "fallback":
        embedding = embeddings.embed(request.description)
        tier2_task_type_resolution = tier2_classifier.resolve_task_type(
            request.description, embedding
        )
        if tier2_task_type_resolution is not None:
            resolved_task_type = tier2_task_type_resolution.task_type
            resolved_task_type_source = "tier2"

    resolved_is_high_stakes: bool | None = None
    high_stakes_source: str | None = None
    no_signal_llm_used = False

    no_signal = (
        resolved_task_type_source == "fallback"
        and classification.domain_source == "fallback"
    )
    if no_signal:
        no_signal_llm_used = True
        llm_bias = _classify_via_llm(request.description)
        if llm_bias is not None:
            bias = llm_bias
            reason = (
                "no keyword signal on either axis - tier-3 cheap-LLM fallback classified this as "
                f"{TIER_BIAS[bias]} (see router.route()'s no-signal fallback note)"
            )
        else:
            bias = "M"
            reason = (
                "no keyword signal on either axis - tier-3 cheap-LLM fallback was unavailable or "
                "returned an unparseable response, hedged to mid rather than escalating to flagship "
                "by default (see router.route()'s no-signal fallback note)"
            )
    else:
        grid_bias = classify(resolved_task_type, classification.domain)
        needs_corroboration = (
            grid_bias == "H" and resolved_task_type_source != "provided"
        )
        if needs_corroboration and not has_high_stakes_signal(request.description):
            if embedding is None:
                embedding = embeddings.embed(request.description)
            tier2_high_stakes_resolution = tier2_classifier.resolve_high_stakes(
                request.description, embedding, task_type=resolved_task_type
            )
            if tier2_high_stakes_resolution is not None:
                resolved_is_high_stakes = tier2_high_stakes_resolution.is_high_stakes
                high_stakes_source = f"tier2_{tier2_high_stakes_resolution.source}"
            else:
                high_stakes_source = "unavailable"
            if (
                tier2_high_stakes_resolution is not None
                and tier2_high_stakes_resolution.is_high_stakes
            ):
                bias = grid_bias
                reason = (
                    f"heuristic grid said H for {resolved_task_type} x {classification.domain} "
                    f"(type {resolved_task_type_source}); no keyword corroborated it, but tier 2's "
                    "continuous-learning classifier confirmed genuine high stakes - escalated to flagship"
                )
            else:
                bias = "M"
                reason = (
                    f"heuristic grid said H for {resolved_task_type} x {classification.domain} "
                    f"(type {resolved_task_type_source}), but no high-stakes signal (production/"
                    "security/compliance/irreversibility/scale) was found in the description, and tier "
                    "2's continuous-learning classifier "
                    + (
                        "confirmed no genuine high stakes"
                        if tier2_high_stakes_resolution is not None
                        else "was unavailable"
                    )
                    + " - capped at mid; flagship requires confirmed difficulty/consequence/ambiguity, "
                    "not just an inferred shape/domain match"
                )
        else:
            bias = grid_bias
            if needs_corroboration:
                # needs_corroboration True but we didn't take the tier-2 branch above means
                # has_high_stakes_signal() short-circuited it - free keyword corroboration.
                resolved_is_high_stakes = True
                high_stakes_source = "keyword"
            reason = (
                f"heuristic grid: {resolved_task_type} x {classification.domain} -> {bias} "
                f"(type {resolved_task_type_source}, domain {classification.domain_source})"
            )

    tier = TIER_BIAS[bias]
    provider, model = TIER_MODELS[tier]

    logged_task_type_source = resolved_task_type_source
    if tier2_task_type_resolution is not None:
        logged_task_type_source = f"tier2_{tier2_task_type_resolution.source}"

    try:
        decision_log.log_decision(
            request.description,
            embedding,
            resolved_task_type=resolved_task_type,
            task_type_source=logged_task_type_source,
            domain=classification.domain,
            domain_source=classification.domain_source,
            resolved_is_high_stakes=resolved_is_high_stakes,
            high_stakes_source=high_stakes_source,
            no_signal_llm_used=no_signal_llm_used,
            bias=bias,
            tier=tier,
            provider=provider,
            model=model,
            reason=reason,
            shadow_bias=shadow_bias,
            shadow_tier=shadow_tier,
            shadow_reason=shadow_reason,
        )
    except Exception:
        pass  # a logging failure must never break routing - see decision_log.py's module docstring

    return RouteDecision(tier=tier, provider=provider, model=model, reason=reason)


def route_and_run(
    request: TaskRequest,
    *,
    on_event: Callable[[dict], None] | None = None,
    on_decision: Callable[[RouteDecision], None] | None = None,
) -> tuple[RouteDecision, ProviderResult]:
    """on_decision, if given, fires the moment routing resolves - before the
    (potentially long) provider call starts - so a caller like repl.py can
    print "routed to X" immediately instead of only after the full response
    lands. on_event is forwarded straight into provider.invoke() for
    per-event streaming (see claude_cli.py); providers that can't stream yet
    (codex_cli.py) accept and ignore it."""
    decision = route(request)
    if on_decision:
        on_decision(decision)
    provider = PROVIDERS[decision.provider]
    result = provider.invoke(
        request.description,
        decision.model,
        session_id=request.session_id,
        on_event=on_event,
    )
    return decision, result
