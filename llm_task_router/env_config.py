"""Loads .env (DATABASE_URL, HF_HUB_OFFLINE, ...) from next to the
installed package, not the caller's cwd - see docs/drift-and-shadow.md's
"DATABASE_URL only in launchd plists" gotcha: DATABASE_URL had been set only
in the two launchd plists (never in a shell profile), so every interactive
llm-chat/llm-route session was silently failing to log to routing_decisions
(swallowed by router.route()'s own try/except Exception: pass). A shell
profile export fixes that but only for the shell it's sourced in and doesn't
help a caller invoking llm-chat/llm-route from a different machine's shell
setup - and a directory-scoped tool like direnv actively breaks the "callable
from any cwd" property `uv tool install --editable .` exists to provide (see
CLAUDE.md, "Calling these from any directory, on any machine") - direnv only
activates on cd, and llm-chat is designed to be invoked from anywhere without
ever cd-ing through this repo first.

Hand-rolled instead of python-dotenv: the zero-third-party-dependency rule
for the CLI/provider-adapter layer (see CLAUDE.md, "Why it's built this
way") covers this too, and a KEY=VALUE parser doesn't need a real library.

Resolved via Path(__file__) - one directory up from this package, i.e. the
repo root - not Path.cwd(): an editable `uv tool install` still points this
file at the real repo checkout regardless of the caller's cwd, so this
works identically whether llm-chat is invoked from inside this repo or from
/tmp.
"""

import os
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_dotenv_if_present(path: Path = _ENV_FILE) -> None:
    """Existing os.environ values always win (setdefault, not []=) - .env
    only fills in what the caller's real environment didn't already set, so
    an explicit `DATABASE_URL=... llm-chat` override, or the launchd
    plists' own EnvironmentVariables block, still take priority unchanged.
    Silently does nothing if the file doesn't exist - .env is optional
    (e.g. CI, or a machine that already exports these some other way), not
    a hard requirement to import this package."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
