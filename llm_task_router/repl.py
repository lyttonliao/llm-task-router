"""Interactive terminal chat client, strictly a routing layer: authenticates
once per provider at startup (offering to run each provider's own login
command interactively), then for every message classifies via
router.route() and delegates to a single persistent tmux-backed terminal
session (terminal.py) running the routed claude/codex CLI - that session is
what actually shows the response and handles all interactive work (tool
use, follow-up turns, plan mode, etc.), natively, not this module. See
CLAUDE.md's "Next step" section and docs/llm-chat.md for the full pivot
away from rendering provider output in-process (the old route_and_run() +
tui.StreamRenderer streaming path).

Persistent tmux session, not spawn-per-message (revised 2026-07-31, fourth
same-day revision - see docs/llm-chat.md for the full history): earlier
revisions spawned a brand new terminal window and provider CLI process for
every message (using --session-id on the first spawn, --resume on every
spawn after). Live use showed this was the wrong model - a new window per
message instead of one continuing place, and a real, previously-untested
--resume race whenever a second message arrived before the first spawn's
session had actually finished being established. chat_loop() now creates
the provider CLI session once (terminal.create_session()), attaches ONE
terminal window to it (terminal.attach_terminal()), and delivers every
message - the first one included - via terminal.send_message(), which
injects the text into that same still-running process exactly as a human
typing would. There is only ever one provider process per run, so the
--resume race is gone structurally, not just mitigated. active_model
tracks which model that live session is currently running as; a tier
change mid-run sends the provider CLI's own /model command
(terminal.switch_model()) before the next message, rather than starting a
new process. See terminal.py's own docstring for the tmux mechanics.
"""

import os
import sys
import uuid

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings

from llm_task_router import terminal, tui
from llm_task_router.known_models import known_models_for
from llm_task_router.router import PROVIDERS, route
from llm_task_router.schema import TaskRequest
from llm_task_router.tiers import TIER_MODELS

_MULTILINE_KEY_BINDINGS = KeyBindings()


@_MULTILINE_KEY_BINDINGS.add("escape", "enter")
def _insert_newline(event) -> None:
    """Alt+Enter (Meta+Enter) inserts a literal newline instead of
    submitting - the deliberate-multiline complement to bracketed paste
    below: paste never needs this (pasted text lands as literal buffer
    content on its own), this is for typing/adding a second line by hand,
    e.g. appending context above or below pasted text before sending."""
    event.current_buffer.insert_text("\n")


def build_input_fn():
    """Returns an input_fn (str -> str, same call contract as builtin
    input()) backed by prompt_toolkit instead of readline. Fixes three
    real UX gaps in plain input(), all coming from the same root cause -
    input() has no real line editor, just raw readline passthrough over
    the terminal's own pty: (1) pasted text containing newlines used to
    submit each line separately the moment the first embedded '\\n' hit
    the pty, since a paste is indistinguishable from fast typing without a
    real line editor - prompt_toolkit's bracketed-paste support (automatic
    on any vt100-compatible terminal, no config needed here) inserts a
    paste as one literal buffer edit instead, so multi-line pasted content
    (e.g. resume bullets, code) lands intact and only sends on a real
    Enter keypress; (2) arrow-key/Home/End/Ctrl-A-style in-line editing now
    works properly instead of however the raw terminal happens to handle
    escape sequences; (3) up/down now recalls this run's previous messages
    via InMemoryHistory (session-scoped only, by design - no persistence
    across llm-chat runs needed for this).

    tui.prompt() returns raw ANSI escape codes for the "you>" label -
    wrapped in ANSI() so prompt_toolkit parses/renders those codes instead
    of treating them as literal printable characters (its own screen
    buffer manages the terminal, unlike input() which just writes straight
    through)."""
    session = PromptSession(history=InMemoryHistory(), key_bindings=_MULTILINE_KEY_BINDINGS)

    def input_fn(prompt_text: str = "") -> str:
        return session.prompt(ANSI(prompt_text))

    return input_fn


def check_provider_auth(name: str, module) -> tuple[bool, str]:
    """Wraps module.check_auth() so a provider CLI simply not being
    installed on PATH surfaces as a clean (False, reason) instead of an
    uncaught traceback - the first place in this repo that matters, since a
    one-shot llm-route failing this way is tolerable but an interactive
    client crashing immediately on startup is not."""
    try:
        return module.check_auth()
    except (FileNotFoundError, OSError) as exc:
        return False, f"could not run {name} CLI: {exc}"


def ensure_provider_authenticated(name: str, module, *, input_fn=input, print_fn=print) -> bool:
    authenticated, reason = check_provider_auth(name, module)
    if authenticated:
        print_fn(f"[{name}] authenticated")
        return True

    print_fn(f"[{name}] not authenticated ({reason})")
    try:
        answer = input_fn(f"Log in to {name} now? [Y/n] ").strip().lower()
    except EOFError:
        answer = "n"

    if answer not in ("", "y", "yes"):
        print_fn(f"[{name}] skipping login; {name} will be unavailable this session")
        return False

    try:
        module.login()
    except (FileNotFoundError, OSError) as exc:
        print_fn(f"[{name}] could not launch login flow: {exc}")
        return False

    # login()'s own return code isn't trusted (see its docstring) - re-check
    # via check_auth() as the real source of truth.
    authenticated, reason = check_provider_auth(name, module)
    if authenticated:
        print_fn(f"[{name}] authenticated")
    else:
        print_fn(f"[{name}] still not authenticated after login attempt ({reason})")
    return authenticated


def startup_auth_check(providers: dict = PROVIDERS, *, input_fn=input, print_fn=print) -> set[str]:
    """Iterates providers in router.PROVIDERS's declared order (claude, then
    codex) for deterministic output. providers is parameterized purely for
    testability - pass a small fake dict instead of exercising real CLIs."""
    authenticated = set()
    for name, module in providers.items():
        if ensure_provider_authenticated(name, module, input_fn=input_fn, print_fn=print_fn):
            authenticated.add(name)
    return authenticated


def routable_tiers(authenticated_providers: set[str], tier_models: dict = TIER_MODELS) -> tuple[dict, dict]:
    """Splits tier_models into (routable, unroutable) by whether each tier's
    provider is in authenticated_providers. Display/startup-decision helper
    ONLY - never filters PROVIDERS/TIER_MODELS, route() is called completely
    unmodified for every real message (see chat_loop).
    Because TIER_MODELS maps every tier to "claude" today, authenticating
    Codex only yields an empty routable dict - a real, documented
    consequence of tiers.py's current calibration state (see
    docs/rough-edges.md), not something this feature works around."""
    routable = {}
    unroutable = {}
    for tier, (provider, model) in tier_models.items():
        if provider in authenticated_providers:
            routable[tier] = (provider, model)
        else:
            unroutable[tier] = (provider, model)
    return routable, unroutable


def print_startup_summary(authenticated_providers: set[str], routable: dict, unroutable: dict, *, print_fn=print) -> None:
    print_fn(f"\n{tui.style(tui.DIM)}Authenticated providers:{tui.style(tui.RESET)}")
    for name in sorted(authenticated_providers):
        color = tui.provider_color(name)
        print_fn(f"  {color}{name}{tui.style(tui.RESET)}: known models (reference only) = {known_models_for(name)}")

    print_fn(f"\n{tui.style(tui.DIM)}Routable tiers:{tui.style(tui.RESET)}")
    for tier, (provider, model) in routable.items():
        color = tui.provider_color(provider)
        print_fn(f"  {tier}: {color}{provider}/{model}{tui.style(tui.RESET)}")

    for tier, (provider, model) in unroutable.items():
        print_fn(
            f"{tui.style(tui.ERROR_COLOR)}[warning]{tui.style(tui.RESET)} tier '{tier}' maps to '{provider}/{model}' but you're not "
            f"authenticated with {provider} - messages classified into this tier "
            f"will fail with an auth error until you log in."
        )
    print_fn("")


def format_response(decision, result) -> str:
    """Non-streaming full-message formatter - kept as the pinned text
    contract (see tests) and available to any future caller with no live
    on_event wiring. chat_loop's success path does not call this (see module
    docstring)."""
    header = tui.header(decision)
    if result.error:
        return f"{header} {tui.error_line(result.error)}"
    wrapped = tui.wrap_text(result.text, tui.get_terminal_width())
    return f"{header}\n{wrapped}\n{tui.footer(result)}"


def chat_loop(*, input_fn=input, print_fn=print) -> None:
    """One session_id, generated once here (not per message), is threaded
    into every route() call and into the single tmux session created for
    this run (see terminal.create_session()). session_created starts False
    and flips to True right after create_session() itself succeeds - not
    after attach_terminal() too, deliberately: a real live failure
    (2026-07-31, see docs/rough-edges.md) showed attach_terminal() can fail
    to actually display anything (a race against the spawned shell's own
    interactive startup) while create_session() genuinely succeeded, so a
    retry must fall through to send_message() rather than attempt a second
    tmux new-session with a name that already exists (tmux rejects that as
    a duplicate). Only a create_session() failure itself - unknown
    provider, tmux missing, unsupported platform - means the session
    doesn't exist and the next message must retry creation.

    active_model tracks which model the live tmux session is currently
    running as. Every message after the session is created either sends
    straight through (terminal.send_message()) when the routed model
    matches active_model, or first sends a /model switch
    (terminal.switch_model()) when the tier changed - both go through the
    same synchronous send-keys injection, so there's no spawn/resume race
    to worry about: chat_loop()'s own while loop already serializes these
    calls one message at a time.

    A boxed input frame (top/bottom border around each prompt) was tried and
    removed the same day: it can't wrap around input that line-wraps in the
    terminal without raw terminal mode and a hand-rolled line editor (the
    same bigger build declined for live-redraw streaming) - see tui.py's
    module docstring for the full rationale. Plain styled prompt only.

    A divider + blank line print before every prompt after the first (so
    turns don't run together) and a blank line prints after the user's line
    for messages that actually route (before the header) - the visible gap
    the user asked around each "you>" prompt. Skipped for blank input and
    slash commands, which stay tight to the prompt they answer."""
    session_id = str(uuid.uuid4())
    session_created = False
    active_model = None
    first_turn = True
    while True:
        if not first_turn:
            print_fn(tui.divider())
            print_fn()
        first_turn = False

        try:
            line = input_fn(tui.prompt())
        except EOFError:
            print_fn()
            return

        line = line.strip()
        if not line:
            continue

        if line in ("/exit", "/quit"):
            return
        if line.startswith("/"):
            print_fn(f"unknown command: {line} (only /exit and /quit are supported)")
            continue

        print_fn()

        request = TaskRequest(description=line, session_id=session_id)
        try:
            decision = route(request)
        except Exception as exc:
            # Deliberate broad catch, unlike every other layer of this repo:
            # route() "shouldn't" raise since its own logging is already
            # wrapped in try/except - but an interactive session dying
            # entirely over one unanticipated exception is worse UX than a
            # one-shot process exiting on the same fault (which just gets
            # re-run). Always printed in full, never swallowed silently.
            print_fn(f"[internal error] {exc!r} - message not sent, try again")
            continue

        print_fn(tui.header(decision))

        try:
            if not session_created:
                terminal.create_session(decision.provider, decision.model, session_id, os.getcwd())
                # Flip these as soon as create_session() succeeds, not after
                # attach_terminal() too - a retry after an attach failure
                # must fall through to send_message() below rather than
                # attempting a second tmux new-session with the same name
                # (which tmux rejects as a duplicate).
                session_created = True
                active_model = decision.model
                print_fn(
                    f"{tui.style(tui.DIM)}tmux session: {session_id} "
                    f"(if no window opens, attach manually: tmux attach -t {session_id}){tui.style(tui.RESET)}"
                )
                terminal.attach_terminal(session_id, os.getcwd())
            elif decision.model != active_model:
                terminal.switch_model(session_id, decision.model)
                active_model = decision.model

            terminal.send_message(session_id, line)
        except Exception as exc:
            print_fn(f"[internal error] {exc!r} - could not deliver this message, try again")
            continue

        print_fn(f"{tui.style(tui.DIM)}sent to session{tui.style(tui.RESET)}")


def main() -> None:
    print("llm-chat - interactive router client\n")

    if not terminal.tmux_available():
        print(
            "tmux not found on PATH - llm-chat needs it to keep a persistent "
            "provider session alive across messages (see docs/llm-chat.md). "
            "Install it (e.g. `brew install tmux`) and try again. Exiting."
        )
        sys.exit(1)

    authenticated = startup_auth_check()
    if not authenticated:
        print("No providers authenticated. Nothing to route to - exiting.")
        sys.exit(1)

    routable, unroutable = routable_tiers(authenticated)
    print_startup_summary(authenticated, routable, unroutable)
    if not routable:
        print(
            "No routing tier currently points at a provider you're authenticated "
            "with - see docs/rough-edges.md's tier calibration status. Exiting."
        )
        sys.exit(1)

    chat_loop(input_fn=build_input_fn())


if __name__ == "__main__":
    main()
