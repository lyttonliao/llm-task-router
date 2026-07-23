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
    task_type: str  # one of TASK_TYPES
    domain: str = "other"  # one of DOMAINS


@dataclass
class RouteDecision:
    tier: str  # cheap | mid | flagship
    provider: str
    model: str
    reason: str


@dataclass
class ProviderResult:
    text: str
    cost_usd: float
    duration_ms: int
    error: str = ""
