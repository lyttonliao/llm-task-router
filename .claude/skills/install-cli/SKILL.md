---
name: install-cli
description: Install llm-chat and routing scripts globally via `uv tool install` so they're callable from any directory, independent of this repo's location.
---

## Within the repo: `uv run`

The basic form works from inside the repo:

```bash
uv run llm-chat
uv run llm-route route "<description>" --type task --domain code
uv run llm-route-audit-tier2
uv run llm-route-shadow-report
```

`uv run` is repo-scoped — it only resolves the entrypoint when `cwd` is
inside this repo (or `--project` points at it).

## Global installation: `uv tool install --editable`

For a globally callable `llm-chat` (or any of the other scripts) regardless
of `cwd` — including from a non-Python project (Go, Node, whatever) — install
this package as a `uv` tool instead of running it in place:

```bash
cd /path/to/llm-task-router
uv tool install --editable .
```

This is `uv`'s equivalent of `pipx install` / `go install` / `npm install -g`:
it builds one isolated venv for the package under
`~/.local/share/uv/tools/llm-task-router/` and drops thin shim executables
(`llm-chat`, `llm-route`, `llm-route-audit-tier2`, `llm-route-shadow-report`)
into `~/.local/bin`.

`--editable` means source edits in this repo take effect immediately, no
reinstall needed after a change. Confirmed end-to-end 2026-07-30: `which
llm-chat` from `/tmp` resolves to the `~/.local/bin` shim, whose shebang
points straight at the isolated venv's own interpreter — `uv` is only the
installer here, never a runtime dependency, and `cwd`/`pyproject.toml` are
irrelevant at call time.

`~/.local/bin` is `uv`'s own default shim directory (same place `uv` itself
often lives), so on a machine that already has `uv` on `$PATH` this needs no
further shell-rc edit. If it isn't yet, `uv tool install` prints a warning
and `uv tool update-shell` fixes it.

Anyone cloning this repo runs the exact same `uv tool install --editable .`
from repo root — no machine-specific path to hand-edit, no per-clone divergence.

## Verifying the installation

```bash
which llm-chat
# Output: /Users/you/.local/bin/llm-chat

# Works from anywhere:
cd /tmp
llm-chat
```

`/tmp` here is just a stand-in for "some directory that obviously isn't
`llm-task-router`" — any Unix scratch directory works; the point is proving
behavior doesn't depend on `cwd` (see `env_config.py`'s `.env` loading,
which has to hold under exactly this test).

## Debugging: running the installed venv's Python directly

The `llm-chat`/`llm-route` shims only accept their own subcommands (`llm-route
route "..."`), not an arbitrary inline script — useful for real usage, not
for probing internals (e.g. confirming what `os.environ` looks like inside
the installed package, independent of the caller's shell). For that, skip
the shim and call the isolated venv's own interpreter directly:

```bash
~/.local/share/uv/tools/llm-task-router/bin/python -c "
import os
import llm_task_router
print(os.environ.get('DATABASE_URL'))
"
```

This is the exact same venv `uv tool install --editable .` already built —
not a second install, not a different mechanism, just reaching past the
`llm-chat`/`llm-route` wrappers to run one-off Python against the installed
package. Combine with the `/tmp`-style cwd trick above and `env -u
DATABASE_URL` to test what a completely clean shell actually sees (used to
verify `env_config.py`'s `.env` loader works regardless of both `cwd` and
which shell-profile lines happen to be sourced).
