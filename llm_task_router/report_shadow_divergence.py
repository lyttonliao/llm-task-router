"""Shadow-evaluation report for the router's live dual-routing (see
router.py's `_shadow_tier1_only_decision()` for how the shadow reading is
computed - free, no extra LLM/embedding/Postgres calls beyond what route()
already makes for the real decision).

Every routing_decisions row (see db/schema.sql) carries both the real
decision (bias/tier/provider/model - what actually served the request) and a
shadow decision (shadow_bias/shadow_tier/shadow_reason - what tier 1 alone,
with tier 2 never consulted, would have decided for the same request). This
script reads the live table back and reports how often, and in which
direction, tier 2 changes the outcome - the data this repo's CLAUDE.md says
is needed before tier 1's heuristic grid can ever be deprecated (see "The
classifier" section).

Does NOT judge which of the two decisions was "right" - that needs either a
future ground-truth label or a judge call, neither of which this script
makes. It only measures divergence and direction (escalated = tier 2 routed
more expensively than tier 1 alone would have; de-escalated = the reverse),
plus which tier2_classifier axis (task_type resolution vs high-stakes
corroboration) drove each divergence.

Requires DATABASE_URL to already be set, same as audit_tier2.py -
decision_log.py raises RuntimeError on its own the first time it needs a
connection, so that check isn't duplicated here.

Usage:
    DATABASE_URL=postgresql:///llm_task_router uv run llm-route-shadow-report
"""

from collections import Counter

from llm_task_router import decision_log
from llm_task_router.decision_log import LoggedDecision
from llm_task_router.tiers import TIER_ORDER


def _direction(real_tier: str, shadow_tier: str) -> str:
    if TIER_ORDER[real_tier] > TIER_ORDER[shadow_tier]:
        return "escalated"
    if TIER_ORDER[real_tier] < TIER_ORDER[shadow_tier]:
        return "de-escalated"
    return "unchanged"


def _driving_source(decision: LoggedDecision) -> str:
    """Which tier2_classifier axis explains a divergence, if either does -
    both a task_type and a high-stakes tier2 resolution can fire on the same
    request (see router.route()), so this can report either, both, or
    neither (a divergence can also come purely from the no-signal mid-hedge
    interacting differently with a tier2-rescued task_type - see
    router._shadow_tier1_only_decision()'s docstring)."""
    task_type_tier2 = decision.task_type_source.startswith("tier2_")
    high_stakes_tier2 = bool(
        decision.high_stakes_source
    ) and decision.high_stakes_source.startswith("tier2_")
    if task_type_tier2 and high_stakes_tier2:
        return "task_type + high_stakes"
    if task_type_tier2:
        return "task_type"
    if high_stakes_tier2:
        return "high_stakes"
    return "neither (no-signal hedge divergence)"


def main() -> None:
    decisions = decision_log.fetch_decisions()
    print(f"{len(decisions)} logged decision(s).")

    scored = [d for d in decisions if d.shadow_tier is not None]
    skipped = len(decisions) - len(scored)
    if skipped:
        print(
            f"{skipped} row(s) predate shadow evaluation (null shadow_tier) - excluded."
        )
    if not scored:
        print("No rows with shadow data yet - nothing to report.")
        return

    divergent = [d for d in scored if d.shadow_tier != d.tier]
    rate = len(divergent) / len(scored)
    print(
        f"\n{len(divergent)}/{len(scored)} ({rate:.1%}) diverge from the tier-1-only shadow."
    )

    direction_counts = Counter(_direction(d.tier, d.shadow_tier) for d in divergent)
    print("\nDirection (tier 2 real vs. tier-1-only shadow):")
    for direction in ("escalated", "de-escalated"):
        print(f"  {direction:<14}{direction_counts.get(direction, 0)}")

    source_counts = Counter(_driving_source(d) for d in divergent)
    print("\nWhich tier2_classifier axis drove the divergence:")
    for source, count in source_counts.most_common():
        print(f"  {source:<28}{count}")

    print("\nSample divergent rows (up to 10):")
    for d in divergent[:10]:
        direction = _direction(d.tier, d.shadow_tier)
        print(
            f"  [{direction}] real={d.tier} shadow={d.shadow_tier} source={_driving_source(d)}"
        )
        print(f"    {d.description!r}")


if __name__ == "__main__":
    main()
