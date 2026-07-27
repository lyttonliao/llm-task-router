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

from llm_task_router.classifier import classify, classify_description
from llm_task_router.providers import claude_cli, codex_cli
from llm_task_router.schema import ProviderResult, RouteDecision, TaskRequest
from llm_task_router.tiers import TIER_BIAS, TIER_MODELS

PROVIDERS = {
    "claude": claude_cli,
    "codex": codex_cli,
}


def route(request: TaskRequest) -> RouteDecision:
    classification = classify_description(request.description, request.task_type, request.domain)
    bias = classify(classification.task_type, classification.domain)
    tier = TIER_BIAS[bias]
    provider, model = TIER_MODELS[tier]
    return RouteDecision(
        tier=tier,
        provider=provider,
        model=model,
        reason=(
            f"heuristic grid: {classification.task_type} x {classification.domain} -> {bias} "
            f"(type {classification.task_type_source}, domain {classification.domain_source})"
        ),
    )


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
