"""Orchestrates classify -> resolve tier -> invoke provider.

The tier-1 heuristic grid (classifier.py) drives routing for every case with
real keyword signal. Tier 3 - a cheap-LLM-call fallback - was added
2026-07-27, scoped only to the no-signal band (see route()'s docstring); a
trained model, cold-started from llm-eval-harness golden-set labels, for
cases the heuristic doesn't cover confidently, is still not built (tier 2).
Adding that tier means giving classify() a confidence signal and falling
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

    Tier 3 for the no-signal band, added 2026-07-27: a keyword-list approach
    to distinguishing "repeat this word: hello" (trivial, should be cheap)
    from "help with this request" (genuinely ambiguous, the mid hedge above
    is correct) was tried and rejected - it doesn't generalize, and a
    vector-DB/embeddings alternative was evaluated and rejected too (breaks
    this repo's zero-third-party-dependency rule, and llm-eval-harness's own
    CLAUDE.md already found embeddings don't cleanly separate this kind of
    soft semantic distinction - see that repo's "regex, not AST or
    embeddings" section). Instead, the no-signal branch now makes one cheap,
    stripped haiku call (_classify_via_llm()) asking the model to judge the
    task's actual difficulty/consequence/ambiguity directly - genuinely a
    judgment call, which is what a small model is good at and keyword
    matching structurally isn't. Falls back to the mid hedge on any failure
    (see _classify_via_llm()'s docstring). Scoped to the no-signal branch
    only - the needs_corroboration branch below has a different failure
    mode (underrouting a possibly-genuinely-hard task, not overrouting a
    trivial one) and was deliberately left alone here.
    """
    classification = classify_description(request.description, request.task_type, request.domain)

    no_signal = classification.task_type_source == "fallback" and classification.domain_source == "fallback"
    if no_signal:
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
