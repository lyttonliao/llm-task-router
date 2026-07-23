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

from llm_task_router.classifier import classify
from llm_task_router.providers import claude_cli, codex_cli
from llm_task_router.schema import ProviderResult, RouteDecision, TaskRequest
from llm_task_router.tiers import TIER_BIAS, TIER_MODELS

PROVIDERS = {
    "claude": claude_cli,
    "codex": codex_cli,
}


def route(request: TaskRequest) -> RouteDecision:
    bias = classify(request.task_type, request.domain)
    tier = TIER_BIAS[bias]
    provider, model = TIER_MODELS[tier]
    return RouteDecision(
        tier=tier,
        provider=provider,
        model=model,
        reason=f"heuristic grid: {request.task_type} x {request.domain} -> {bias}",
    )


def route_and_run(request: TaskRequest) -> tuple[RouteDecision, ProviderResult]:
    decision = route(request)
    provider = PROVIDERS[decision.provider]
    result = provider.invoke(request.description, decision.model)
    return decision, result
