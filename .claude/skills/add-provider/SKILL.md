---
name: add-provider
description: Add and verify a new provider CLI adapter for llm-task-router (e.g. a new coding-agent CLI), following the real-verification workflow used for claude_cli.py and codex_cli.py. Use when asked to add support for a new model provider or CLI tool.
---

The codex_cli.py adapter shipped with several wrong guesses (an invalid flag
that only surfaced on a real authenticated call, a JSON-shape assumption that
was wrong) that a real call caught and a placeholder-only build would have
shipped silently. Don't skip the real-verification steps below just because
the code "looks done" after matching `--help` output alone.

## Steps

1. **Check if the CLI is installed** (`which <cli>`). If not, ask the user
   before installing anything globally (e.g. `npm install -g ...`) - don't
   assume a package manager or install method.

2. **Read the actual subcommand's `--help`, not just the top-level one.**
   A flag valid at the top level (`codex --ask-for-approval`) was NOT valid
   on the subcommand actually used (`codex exec`) - a real call failed with
   "unexpected argument" until this was caught. Don't carry a flag from one
   `--help` output to a different subcommand without checking that
   subcommand's own `--help`.

3. **Check auth/login status** (e.g. `<cli> doctor` or equivalent). If not
   authenticated, tell the user to run the login command themselves
   interactively (suggest `! <cli> login`) - never attempt to complete an
   OAuth/browser login flow on their behalf.

4. **Write the adapter** in `llm_task_router/providers/<name>.py`, matching
   `providers/base.py`'s `Provider` protocol: `invoke(prompt, model) ->
   ProviderResult`. Prefer whatever mechanism gets you the final answer most
   directly (codex_cli.py uses `--output-last-message <tempfile>` rather than
   parsing a streamed JSONL event log) - simpler parsing, fewer ways to get
   the shape wrong.

5. **Write mocked-subprocess tests** in `tests/test_<name>.py`, mirroring
   `tests/test_claude_cli.py`/`tests/test_codex_cli.py`: assert the exact
   command list (not just that `subprocess.run` was called), plus timeout,
   nonzero-exit, and missing-output-file paths.

6. **Do one real authenticated call** (confirm with the user first if it
   might cost money/quota) to verify the adapter actually works, not just
   that it matches documented flags. Fix whatever a real call catches that
   `--help` alone didn't - document the fix and what was wrong in the
   adapter's docstring, the same way codex_cli.py's docstring records what
   was confirmed vs. guessed.

7. **Register in `router.PROVIDERS`** - store the provider *module*, not a
   pre-grabbed function reference (`router.PROVIDERS = {"name": module}`,
   then call `.invoke(...)` on it). A dict built with the bound function
   directly (`{"name": module.invoke}`) early-binds the reference at import
   time and silently breaks `patch("...module.invoke")` in tests - this bit
   `router.py` once already, see its comments.

8. **Do NOT add an entry to `tiers.TIER_MODELS`** for the new provider yet -
   that needs real quality-floor data from `llm-eval-harness`'s
   `calibrate-tier` skill first. A tier mapping is only as good as the
   quality floor behind it.

9. **Update this repo's `CLAUDE.md`** - architecture tree, and a "known rough
   edges" entry for anything still unverified (cost/usage reporting, error
   edge cases you didn't get to exercise for real).
