"""Maps abstract cost tiers to concrete (provider, model) pairs.

All tiers point at Claude for now. Populating real Codex/OpenAI entries here
needs calibration data - per-category quality floors benchmarked the same way
llm-eval-harness benchmarks Claude prompt versions (see that repo's CLAUDE.md,
"Why two scorers") - which doesn't exist yet for non-Claude models. Don't add
a provider here without that data; a routing decision is only as good as the
quality floor behind it.
"""

TIER_BIAS = {"L": "cheap", "M": "mid", "H": "flagship"}

TIER_MODELS: dict[str, tuple[str, str]] = {
    "cheap": ("claude", "haiku"),
    "mid": ("claude", "sonnet"),
    "flagship": ("claude", "opus"),
}
