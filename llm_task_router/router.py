"""Orchestrates classify -> resolve tier -> invoke provider.

Only the tier-1 heuristic grid (classifier.py) drives routing right now. The
full cascade planned in the router design thread (pillar 10) also has:
  - a trained model, cold-started from llm-eval-harness golden-set labels,
    for cases the heuristic doesn't cover confidently
  - a cheap-LLM-call fallback for the remaining ambiguity band
Neither exists yet - route() always resolves via the heuristic grid. Adding
the next cascade tier means giving classify() a confidence signal and falling
through to it here, not replacing this function's shape.
"""

from collections.abc import Callable

from llm_task_router.classifier import classify, classify_description, has_high_stakes_signal
from llm_task_router.providers import claude_cli, codex_cli
from llm_task_router.schema import ProviderResult, RouteDecision, TaskRequest
from llm_task_router.tiers import TIER_BIAS, TIER_MODELS

PROVIDERS = {
    "claude": claude_cli,
    "codex": codex_cli,
}


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
    irreversibility/scale vocabulary). An inferred H without that
    corroboration is capped at mid. This is a real, accepted precision/
    recall tradeoff, not a free lunch: a genuinely hard task that doesn't
    happen to use recognized high-stakes vocabulary will now be underrouted
    to mid rather than reaching flagship - see IMPACT_KEYWORDS' docstring for
    why that's an intentional bet on tier-1 heuristics' limits, not an
    oversight, pending tier 2/3 of the confidence cascade.
    """
    classification = classify_description(request.description, request.task_type, request.domain)

    no_signal = classification.task_type_source == "fallback" and classification.domain_source == "fallback"
    if no_signal:
        bias = "M"
        reason = (
            "no keyword signal on either axis - hedged to mid rather than escalating to "
            "flagship by default (see router.route()'s no-signal fallback note)"
        )
    else:
        grid_bias = classify(classification.task_type, classification.domain)
        needs_corroboration = grid_bias == "H" and classification.task_type_source != "provided"
        if needs_corroboration and not has_high_stakes_signal(request.description):
            bias = "M"
            reason = (
                f"heuristic grid said H for {classification.task_type} x {classification.domain} "
                f"(type {classification.task_type_source}), but no high-stakes signal (production/"
                "security/compliance/irreversibility/scale) was found in the description - capped at "
                "mid; flagship requires confirmed difficulty/consequence/ambiguity, not just an "
                "inferred shape/domain match"
            )
        else:
            bias = grid_bias
            reason = (
                f"heuristic grid: {classification.task_type} x {classification.domain} -> {bias} "
                f"(type {classification.task_type_source}, domain {classification.domain_source})"
            )

    tier = TIER_BIAS[bias]
    provider, model = TIER_MODELS[tier]
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
    result = provider.invoke(request.description, decision.model, session_id=request.session_id, on_event=on_event)
    return decision, result
