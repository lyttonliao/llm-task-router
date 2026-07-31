# Installed CLI entrypoints

`pyproject.toml` declares `[project.scripts] llm-route =
"llm_task_router.cli:main"` (plus `llm-chat`, `llm-route-audit-tier2`,
`llm-route-shadow-report`) and `uv`-package build config (package lives at
repo root, not `src/`, hence `[tool.uv.build-backend] module-root = ""`).
`uv run llm-route route "<description>" --dry-run` works as a real command;
`python -m llm_task_router route ...` still works identically side by side.
`uv run` is repo-scoped, though — it only resolves the entrypoint when `cwd`
is inside this repo (or `--project` points at it). See below for calling
these from anywhere, independent of the caller's language/project type.

## Calling these from any directory, on any machine

`uv run` needs to be invoked from inside this repo. For a globally callable
`llm-chat` (or any of the other three scripts) regardless of `cwd` —
including from a non-Python project (Go, Node, whatever) — install this
package as a `uv` tool instead of running it in place:

```bash
cd /path/to/llm-task-router
uv tool install --editable .
```

This is `uv`'s equivalent of `pipx install` / `go install` / `npm install
-g`: it builds one isolated venv for the package under
`~/.local/share/uv/tools/llm-task-router/` and drops thin shim executables
(`llm-chat`, `llm-route`, `llm-route-audit-tier2`, `llm-route-shadow-report`)
into `~/.local/bin`. `--editable` means source edits in this repo take
effect immediately, no reinstall needed after a change. Confirmed
end-to-end 2026-07-30: `which llm-chat` from `/tmp` resolves to the
`~/.local/bin` shim, whose shebang points straight at the isolated venv's
own interpreter — `uv` is only the installer here, never a runtime
dependency, and `cwd`/`pyproject.toml` are irrelevant at call time.

`~/.local/bin` is `uv`'s own default shim directory (same place `uv` itself
often lives), so on a machine that already has `uv` on `$PATH` this needs no
further shell-rc edit. If it isn't yet, `uv tool install` prints a warning
and `uv tool update-shell` fixes it. Anyone cloning this repo runs the exact
same `uv tool install --editable .` from repo root — no machine-specific
path to hand-edit, no per-clone divergence.

See the `install-cli` skill for the automated version of this.
