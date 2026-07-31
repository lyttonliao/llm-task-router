# Auth pre-flight check

Both provider adapters export `check_auth() -> tuple[bool, str]` that
`invoke()` calls first — `claude auth status --json` (parses `loggedIn`) and
`codex login status` (text-matches "logged in"/"not logged in"). An
unauthenticated call short-circuits to `ProviderResult(error="auth check
failed: ...")` before reaching the real model subprocess, instead of falling
through to whatever failure shape the underlying CLI produces on its own.

Both adapters' logged-out shapes are confirmed against real output
(2026-07-26), without actually logging this dev account out:
- **Claude**: `env -u ANTHROPIC_API_KEY claude --bare auth status` →
  `{"loggedIn": false, ...}` at exit 1. `--bare` skips keychain/OAuth reads
  entirely.
- **Codex**: `CODEX_HOME=<empty dir> codex login status` → "Not logged in"
  at exit 1.

Reuse these two techniques for testing this gate instead of a real logout,
which needs an interactive re-auth flow to undo. A new provider's
`check_auth()` should follow this same shape — don't skip it just because
the provider's own nonzero-exit path eventually surfaces an auth error; the
point is failing fast and consistently.

## Adding a provider

See the `add-provider` skill for the full real-verification workflow. Key
points:

1. Add `llm_task_router/providers/<name>.py` with `invoke(prompt, model) ->
   ProviderResult` following `claude_cli.py`'s shape, including a
   `check_auth() -> tuple[bool, str]` that `invoke()` calls first.

2. Register it in `router.PROVIDERS` as a **module**, not a pre-grabbed
   function — `{"name": module}` then `.invoke(...)` at call time, not
   `{"name": module.invoke}`. The latter early-binds at import time and
   silently defeats `patch("...module.invoke")` in tests (already happened
   once in `router.py`'s own first draft).

3. Only add entries to `tiers.TIER_MODELS` once you have real quality-floor
   calibration data (`llm-eval-harness`'s `calibrate-tier` skill). As of
   2026-07-23, no Codex model clears haiku's cheap-tier floor — see
   `rough-edges.md` for the current reachable-model list; don't add a
   Codex entry off stale data. Same discipline applies to
   `classifier.TYPE_DOMAIN_GRID` — see `llm-eval-harness/CLAUDE.md`'s
   "Router tier synthesis across all 7 suites".
