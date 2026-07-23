"""Tier-1 of the eventual three-tier confidence cascade (router planning
thread, pillar 10): a hand-authored heuristic rule table. Cheapest to run,
fires first, and is meant to be refined over time by shadow-eval data rather
than replaced outright (pillars 19, 22). The trained-model tier and the
cheap-LLM-fallback tier for ambiguous cases aren't built yet - see
router.route()'s docstring for where they'd plug in.

Grid below is pillar 22's confirmed type x domain escalation-bias table:
L = low/pattern-closed (cheap tier default), M = medium, H = high/tradeoff-open
(default escalate).
"""

from llm_task_router.schema import DOMAINS, TASK_TYPES

TYPE_DOMAIN_GRID: dict[str, dict[str, str]] = {
    "triage": {"frontend": "L", "backend": "L", "infra": "M", "data": "M", "other": "M"},
    "code_gen": {"frontend": "L", "backend": "L", "infra": "H", "data": "M", "other": "M"},
    "summarization": {"frontend": "L", "backend": "L", "infra": "L", "data": "L", "other": "L"},
    "multi_step": {"frontend": "M", "backend": "M", "infra": "H", "data": "H", "other": "M"},
    "code_review": {"frontend": "L", "backend": "M", "infra": "H", "data": "H", "other": "M"},
    "refactor": {"frontend": "L", "backend": "M", "infra": "H", "data": "M", "other": "M"},
    "architecture": {"frontend": "H", "backend": "H", "infra": "H", "data": "H", "other": "H"},
}


def classify(task_type: str, domain: str) -> str:
    if task_type not in TASK_TYPES:
        raise ValueError(f"unknown task_type: {task_type!r}, expected one of {TASK_TYPES}")
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain: {domain!r}, expected one of {DOMAINS}")
    return TYPE_DOMAIN_GRID[task_type][domain]
