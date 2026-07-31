from dataclasses import dataclass

# The two routing axes from the router planning thread (pillar 16): what shape
# of work this is, and what part of the stack it touches.
TASK_TYPES = (
    "triage",
    "code_gen",
    "summarization",
    "multi_step",
    "code_review",
    "refactor",
    "architecture",
)
DOMAINS = ("frontend", "backend", "infra", "data", "other")


@dataclass
class TaskRequest:
    description: str
    task_type: str | None = None  # one of TASK_TYPES; inferred when omitted
    domain: str | None = None  # one of DOMAINS; inferred when omitted
    session_id: str | None = None  # pins/continues a claude -p conversation across
    # messages; None means no continuity. See docs/llm-chat.md.


@dataclass(frozen=True)
class TaskClassification:
    task_type: str
    domain: str
    task_type_source: str
    domain_source: str


@dataclass
class RouteDecision:
    tier: str  # cheap | mid | flagship
    provider: str
    model: str
    reason: str
    # id of the row decision_log.log_decision() wrote for this decision, or
    # None if logging was unavailable/failed (see router.route()) - lets
    # route_and_run() attach the real ProviderResult's cost/duration back
    # onto that same row via decision_log.log_result() once the provider
    # call completes. Not meaningful outside router.py/decision_log.py.
    decision_log_id: int | None = None


@dataclass
class ProviderResult:
    text: str
    cost_usd: float
    duration_ms: int
    error: str = ""
