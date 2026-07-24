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

from llm_task_router.schema import DOMAINS, TASK_TYPES, TaskClassification

TYPE_DOMAIN_GRID: dict[str, dict[str, str]] = {
    "triage": {"frontend": "L", "backend": "L", "infra": "M", "data": "M", "other": "M"},
    "code_gen": {"frontend": "L", "backend": "L", "infra": "H", "data": "M", "other": "M"},
    "summarization": {"frontend": "L", "backend": "L", "infra": "L", "data": "L", "other": "L"},
    "multi_step": {"frontend": "M", "backend": "M", "infra": "H", "data": "H", "other": "M"},
    "code_review": {"frontend": "L", "backend": "M", "infra": "H", "data": "H", "other": "M"},
    "refactor": {"frontend": "L", "backend": "M", "infra": "H", "data": "M", "other": "M"},
    "architecture": {"frontend": "H", "backend": "H", "infra": "H", "data": "H", "other": "H"},
}

TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "triage": ("bug", "error", "exception", "regression", "debug", "diagnose", "investigate", "fix"),
    "code_gen": ("implement", "write", "create", "generate", "function", "class", "endpoint"),
    "summarization": ("summarize", "summary", "tl;dr", "recap"),
    "multi_step": ("migrate", "migration", "rollout", "workflow", "orchestrate", "integrate"),
    "code_review": ("code review", "review", "audit", "pull request"),
    "refactor": ("refactor", "restructure", "cleanup", "rename"),
    "architecture": ("architecture", "system design", "design", "strategy", "tradeoff", "scalable", "fault-tolerant"),
}

DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "frontend": ("frontend", "react", "css", "html", "browser", "ui", "ux"),
    "backend": ("backend", "api", "server", "authentication", "auth", "service"),
    "infra": ("infrastructure", "kubernetes", "k8s", "terraform", "deployment", "docker", "cloud", "ci/cd"),
    "data": ("data", "analytics", "pipeline", "warehouse", "sql", "etl"),
    "other": (),
}


def classify(task_type: str, domain: str) -> str:
    if task_type not in TASK_TYPES:
        raise ValueError(f"unknown task_type: {task_type!r}, expected one of {TASK_TYPES}")
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain: {domain!r}, expected one of {DOMAINS}")
    return TYPE_DOMAIN_GRID[task_type][domain]


def _infer_from_keywords(description: str, keywords: dict[str, tuple[str, ...]]) -> str | None:
    normalized = description.lower()
    matches = {
        label: sum(keyword in normalized for keyword in label_keywords)
        for label, label_keywords in keywords.items()
        if label_keywords
    }
    best_label, best_score = max(matches.items(), key=lambda item: item[1])
    return best_label if best_score else None


def classify_description(
    description: str, task_type: str | None = None, domain: str | None = None
) -> TaskClassification:
    """Resolve optional caller labels, inferring missing metadata conservatively.

    Unknown task shape falls back to architecture rather than the cheap tier:
    an expensive false escalation is preferable to silently underrouting an
    ambiguous task. Domain uncertainty is represented as ``other``.
    """
    inferred_type = _infer_from_keywords(description, TYPE_KEYWORDS)
    inferred_domain = _infer_from_keywords(description, DOMAIN_KEYWORDS)
    return TaskClassification(
        task_type=task_type if task_type is not None else inferred_type or "architecture",
        domain=domain if domain is not None else inferred_domain or "other",
        task_type_source="provided" if task_type is not None else ("inferred" if inferred_type else "fallback"),
        domain_source="provided" if domain is not None else ("inferred" if inferred_domain else "fallback"),
    )
