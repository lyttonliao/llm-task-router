# Known rough edges

- `providers/codex_cli.py` is verified against a real `codex-cli 0.145.0`
  install and a real authenticated call. Two things still open: (1) no
  dollar-cost field anywhere in `codex exec`'s output (stderr prints an
  unstructured token count, no per-model pricing to convert it), so
  `cost_usd`/`duration_ms` stay 0.0/0 placeholders; (2)
  `--output-last-message`'s behavior on a genuine content refusal (vs. a
  hard API error, which is covered) is unconfirmed. Currently unreachable
  through the router anyway — no tier routes to it yet.
- Codex has no flag equivalent to `--disallowed-tools "*"` — `--sandbox
  read-only` is the closest analog (can't write files) but can still run
  read-only shell commands. Don't assume cost/latency parity between the two
  adapters. Valid model names depend on auth mode (a ChatGPT-account login
  rejects some outright, confirmed via a real 400). As of 2026-07-23,
  reachable on the dev account: `gpt-5.4-mini`, `gpt-5.6-luna`,
  `gpt-5.6-terra`, `gpt-5.5`; not reachable: `gpt-5.6-sol`, `gpt-5.3-codex`,
  `gpt-5.1-codex-mini`, `gpt-5.4-nano`, `gpt-5.4`, `gpt-5.2` (all 400).
  Re-probe with a single cheap `codex exec` call before trusting either list
  on a different account.
- `tiers.TIER_MODELS`'s Claude entries are backed by real judged data, not a
  guess — as of 2026-07-23 a clean, monotonic ladder on `bug_triage`/
  `v1_naive` (haiku 60% fully-correct → sonnet 66.7% → opus 73.3%, judge
  coherence flat ~0.84-0.85). No Codex model clears haiku's floor yet, so the
  map stays Claude-only — see `llm-eval-harness/CLAUDE.md`'s calibration
  status section for the full table.
- Tool access is resolved for every Claude call, not just `llm-chat`'s —
  `llm-route`'s one-shot path goes through the same `claude_cli.invoke()`,
  sharing the same full-functionality/`bypassPermissions` behavior.
  Deliberate: one adapter, one behavior, rather than threading a cost-mode
  flag through both call paths.
- Nothing auto-adjusts `AGREEMENT_THRESHOLD` or deprecates tier 1 based on
  what the drift/shadow reports find — scheduling only automates the
  *running* of those reports, not the judgment; a human still reads the log
  and decides.
- Windows `select()` support is still unverified for the streaming transport.
- Cross-provider session continuity is deferred (see `llm-chat` docs).
- `terminal.attach_terminal()`'s macOS path (`open <script>.command`) can
  lose the race against the spawned window's own interactive shell
  startup: confirmed live (2026-07-31, first real test of the persistent
  tmux redesign) — `open` launches Terminal.app, which starts a normal
  interactive login shell (zsh + oh-my-zsh here) *before* typing the
  script's path into it as if typed by hand. If that shell's own startup
  has a pending interactive prompt (oh-my-zsh's periodic "Would you like
  to update? [Y/n]" hit this time), the injected script path gets
  consumed by that prompt's read instead of executing, so `tmux attach`
  never runs and the window just shows a shell error. This is a
  pre-existing risk in the `open`+`.command` mechanism itself
  (`_spawn_macos()`'s code is unchanged from before the tmux redesign),
  not something new the redesign introduced — it just hadn't been
  triggered live before. **Confirmed NOT to affect the actual session**:
  `create_session()` and `send_message()` ran and worked correctly despite
  this (verified via `tmux capture-pane` against the live detached
  session — a real message got a real response) — only the *viewing*
  window is at risk, not message delivery. The standard fix (oh-my-zsh's
  own `DISABLE_UPDATE_PROMPT="true"` in `~/.zshrc`) was offered and
  declined for now (2026-07-31) — revisit if this bites again, since any
  interactive shell-startup hook (not just oh-my-zsh's update check) could
  trigger the same race, not just this specific one.
