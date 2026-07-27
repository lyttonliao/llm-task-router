"""Tier-1 of the eventual three-tier confidence cascade (router planning
thread, pillar 10): a hand-authored heuristic rule table. Cheapest to run,
fires first, and is meant to be refined over time by shadow-eval data rather
than replaced outright (pillars 19, 22). The trained-model tier and the
cheap-LLM-fallback tier for ambiguous cases aren't built yet - see
router.route()'s docstring for where they'd plug in.

Grid below started as pillar 22's hand-authored type x domain escalation-bias
table: L = low/pattern-closed (cheap tier default), M = medium, H = high/
tradeoff-open (default escalate). As of 2026-07-27, the code_gen and refactor
rows have been superseded by real calibration data from llm-eval-harness (see
that repo's CLAUDE.md, "Router tier synthesis across all 7 suites") - both
suites showed zero measured discrimination across haiku/sonnet/opus on
adversarial-hardened cases, so those two rows are now uniform L rather than
pillar 22's original guess. The remaining rows are still the original
hand-authored heuristic and have not yet been calibration-confirmed or
-contradicted (triage and code_review have real but N=1 signal questioning
their M cells; architecture and multi_step have no comparative tier data or
a confirmed scorer bug blocking one - see that same CLAUDE.md section).
"""

from llm_task_router.schema import DOMAINS, TASK_TYPES, TaskClassification

TYPE_DOMAIN_GRID: dict[str, dict[str, str]] = {
    "triage": {"frontend": "L", "backend": "L", "infra": "M", "data": "M", "other": "M"},
    "code_gen": {"frontend": "L", "backend": "L", "infra": "L", "data": "L", "other": "L"},
    "summarization": {"frontend": "L", "backend": "L", "infra": "L", "data": "L", "other": "L"},
    "multi_step": {"frontend": "M", "backend": "M", "infra": "H", "data": "H", "other": "M"},
    "code_review": {"frontend": "L", "backend": "M", "infra": "H", "data": "H", "other": "M"},
    "refactor": {"frontend": "L", "backend": "L", "infra": "L", "data": "L", "other": "L"},
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

# Signals that a request is genuinely high-stakes - production impact,
# security/compliance exposure, irreversibility, meaningful scale - as
# opposed to merely matching an H-mapped type/domain's *shape* keywords.
# Added 2026-07-27 because TYPE_KEYWORDS' architecture row ("design",
# "strategy") is loose enough to fire on a trivial question ("what's a good
# design for this button?") with the same weight as a real one ("design the
# multi-region disaster recovery strategy for the payment system") - shape
# alone isn't evidence of difficulty, consequence, or ambiguity. See
# router.route()'s docstring for how this gates flagship.
#
# Deliberately narrow and NOT exhaustive - this is a coarse keyword net, the
# same kind of imprecise instrument TYPE_KEYWORDS/DOMAIN_KEYWORDS already
# are, and will have false negatives on real high-stakes asks that don't
# happen to use one of these exact phrases (e.g. "migrate the k8s cluster to
# a new region" with no "multi-region"/"disaster recovery" wording). That is
# an accepted tradeoff, not an oversight: closing this gap for real needs
# tier 2/3 of the confidence cascade (a trained model or a cheap-LLM
# judgment call, see router.py's module docstring), not a longer guessed
# keyword list - widen this against real routing decisions that turned out
# wrong, not speculation, same discipline llm-eval-harness's CLAUDE.md
# documents for phrase-group widening generally.
IMPACT_KEYWORDS: tuple[str, ...] = (
    "production",
    "customer data",
    "customers'",
    "compliance",
    "regulatory",
    "regulation",
    "gdpr",
    "hipaa",
    "pci",
    "security",
    "vulnerability",
    "breach",
    "irreversible",
    "data loss",
    "outage",
    "downtime",
    "disaster recovery",
    "high availability",
    "multi-region",
    "distributed system",
    "at scale",
    "millions of users",
    "financial",
    "payment",
    "pii",
    "breaking change",
    "sla",
    "mission-critical",
    "safety-critical",
)


def has_high_stakes_signal(description: str) -> bool:
    normalized = description.lower()
    return any(keyword in normalized for keyword in IMPACT_KEYWORDS)


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
