"""Static, hand-maintained table of model slugs confirmed reachable on this
account. Used ONLY for repl.py's startup summary so the user knows what's
roughly usable given who they authenticated with - NOT consulted by
router.py or tiers.py, and NOT a source of routing truth (tiers.TIER_MODELS
owns that exclusively). Can go stale - a slug listed here may stop working
or a new one may become available - re-verify with a real call before
trusting it for anything beyond display, the same discipline the
calibrate-tier skill already follows for tiers.py itself.
"""

KNOWN_MODELS: dict[str, tuple[str, ...]] = {
    "claude": ("haiku", "sonnet", "opus"),  # confirmed via judged eval runs, see CLAUDE.md
    "codex": ("gpt-5.4-mini", "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.5"),  # reachable as of 2026-07-23, this account
}


def known_models_for(provider: str) -> tuple[str, ...]:
    return KNOWN_MODELS.get(provider, ())
