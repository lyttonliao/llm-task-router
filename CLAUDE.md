# llm-task-router

Given a task description, classify it and route it to the cheapest model tier
that clears a quality floor for that task category, then actually run it.
The live-routing counterpart to `llm-eval-harness`, which benchmarks
prompt/model quality offline to calibrate the tiers this router picks from.

## Why it's built this way

- **Subscription CLIs, not API keys.** Every provider call shells out to a
  headless CLI (`claude -p`, `codex exec`) instead of an API/SDK, same
  cost-avoidance rationale as `llm-eval-harness`'s `claude_cli.py`: no
  separate per-token billing, runs on existing subscriptions.
- **Zero third-party dependencies** — stdlib only (`dataclasses`, `argparse`,
  `subprocess`, `json`), matching `llm-eval-harness`.
- **Independent from `llm-eval-harness`.** This repo has its own dataclasses
  and its own provider adapters rather than importing the eval harness as a
  package. The two projects interact conceptually (the harness's benchmark
  runs are what should calibrate the tier map in `tiers.py`), not through
  shared code — a live-routing runtime and an offline benchmark runner are
  different enough consumption patterns that coupling them would be the wrong
  kind of DRY.

## Architecture

```
llm_task_router/
  schema.py       - TaskRequest, RouteDecision, ProviderResult dataclasses
  providers/
    base.py       - Provider protocol (invoke(prompt, model) -> ProviderResult)
    claude_cli.py - subprocess wrapper around `claude -p`
    codex_cli.py  - subprocess wrapper around `codex exec`, verified end to
                    end against a real install and a real authenticated call
                    (see rough edges below for what's still unverified)
  classifier.py   - tier-1 heuristic rule table (type x domain grid)
  tiers.py        - tier name -> concrete (provider, model) mapping
  router.py       - route() classifies + resolves a tier; route_and_run()
                    also invokes the provider
  cli.py          - `python -m llm_task_router route <description> --type ... --domain ...`
```

## The classifier is a three-tier cascade; only tier 1 exists

The full design (from the router planning thread) is a confidence cascade:
1. **Heuristic rule table** (`classifier.py`) - cheapest, fires first. This is
   the only tier implemented so far.
2. A **trained model**, cold-started from `llm-eval-harness` golden-set
   labels, for cases the heuristic doesn't confidently cover. Not built.
3. A **cheap LLM call** for the remaining ambiguity band. Not built.

Default-to-escalate under uncertainty, not default-to-cheap - an
underrouted, silently-wrong answer is worse than an overrouted, correct-but-
pricier one, because the former is undetectable without dedicated auditing.

Adding tier 2 or 3 means giving `classifier.classify()` a confidence signal
and having `router.route()` fall through to the next tier when it's low -
don't restructure `route()`'s shape to do this, extend it.

## Adding a provider

1. Add `llm_task_router/providers/<name>.py` with an `invoke(prompt, model)
   -> ProviderResult` function, following `claude_cli.py`'s shape.
2. Register it in `router.PROVIDERS`.
3. Only add entries to `tiers.TIER_MODELS` pointing at it once you have real
   quality-floor calibration data for that provider's models (the same kind
   of benchmark `llm-eval-harness` runs for Claude models) - a tier mapping is
   only as good as the quality floor behind it.

## Known rough edges

- `providers/codex_cli.py` is verified against a real `codex-cli 0.145.0`
  install and a real authenticated (ChatGPT-account) call - not a placeholder
  anymore. Two things still open: (1) there's no dollar-cost field anywhere
  in `codex exec`'s output (a real run's stderr does print an unstructured
  "tokens used" line followed by a token count, but no dollar figure, and
  converting tokens to cost needs per-model pricing this CLI doesn't expose),
  so `cost_usd`/`duration_ms` stay 0.0/0 placeholders; (2) `--output-last-message`'s
  behavior on a genuine
  content refusal (as opposed to a hard API error, which is covered) is
  unconfirmed. It's currently unreachable through the router in practice
  anyway since no tier in `tiers.py` routes to it yet.
- Codex has no flag equivalent to claude_cli.py's `--disallowed-tools "*"`
  that fully disables tool/shell use - `--sandbox read-only` is the closest
  analog (can't write files), but the model can still choose to run
  read-only shell commands to gather context before answering. Don't assume
  cost/latency parity between the two provider adapters. Also: which model
  names are valid depends on the auth mode - a ChatGPT-account login rejects
  some names outright (confirmed via a real 400 invalid_request_error), so
  don't hardcode a guessed model name into `tiers.py` without checking it
  against the account that will actually run it.
- `tiers.TIER_MODELS` only has Claude entries, all guessed at (haiku/sonnet/
  opus for cheap/mid/flagship) rather than derived from benchmark data. Don't
  treat these as validated quality-floor tiers yet.
- Whether task types like `code_gen`/`refactor` should eventually get real
  tool access (vs. the current single-shot, tools-disabled completion every
  provider call makes) is an open, deliberately deferred question - see the
  cost-guardrail flags in `claude_cli.py`.
- No shadow evaluation, no live dual-routing, no drift auditing - this
  scaffold only exploits the current heuristic grid, it doesn't explore or
  self-correct yet.
