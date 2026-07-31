"""Interactive terminal chat client, strictly a routing layer: authenticates
once per provider at startup (offering to run each provider's own login
command interactively), then for every message classifies via
router.route() and spawns a real native terminal (terminal.py) running the
routed claude/codex CLI call - the spawned session is what actually shows
the response and handles all interactive work (tool use, follow-up turns,
plan mode, etc.), natively, not this module. See CLAUDE.md's "Next step"
section and docs/llm-chat.md for the full pivot away from rendering
provider output in-process (the old route_and_run() + tui.StreamRenderer
streaming path).

Every message in a chat_loop() run shares one session_id, threaded into
each spawn call, so conversation history continues across messages even as
the classifier routes different messages to different Claude tiers/models -
see docs/llm-chat.md for why cross-provider continuity (Codex) isn't part
of this and for the --session-id/--resume distinction this rests on.
"""

import sys
import uuid

from llm_task_router import terminal, tui
from llm_task_router.known_models import known_models_for
from llm_task_router.router import PROVIDERS, route
from llm_task_router.schema import TaskRequest
from llm_task_router.tiers import TIER_MODELS


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
    into every route() call and spawn_provider_session() call - the latter
    turns that into --session-id on the first spawn and --resume on every
    spawn after (see session_established below), so conversation history
    continues in the spawned claude/codex session even as the classifier
    sends different messages to different tiers. There is no way to reset
    session_id mid-run (no /clear - see module docstring); a fresh
    conversation means restarting llm-chat.

    session_established starts False and flips to True only after a spawn
    call that didn't raise - every raise path inside
    terminal.spawn_provider_session() (unknown provider, unsupported
    platform, no terminal emulator found, the launcher command itself
    failing) happens before the wrapped claude/codex command ever runs, so
    the session was never actually established on any of those paths (see
    terminal.py's own docstring). A spawn that succeeds but whose inner
    claude/codex process exits nonzero still counts as established -
    unlike providers/claude_cli.py's stricter per-call success gating (it
    inspects the NDJSON result event), there's nothing to inspect here: the
    terminal genuinely opened under that session id, which is what
    --session-id vs --resume correctness actually depends on.

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
    session_established = False
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
            break

        line = line.strip()
        if not line:
            continue

        if line in ("/exit", "/quit"):
            break
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
            print_fn(f"[internal error] {exc!r} - message not sent, session continues")
            continue

        print_fn(tui.header(decision))

        try:
            terminal.spawn_provider_session(
                decision.provider, decision.model, session_id, line, resume=session_established
            )
            session_established = True
        except Exception as exc:
            print_fn(f"[internal error] {exc!r} - could not open a terminal for this session")


def main() -> None:
    print("llm-chat - interactive router client\n")
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

    chat_loop()


if __name__ == "__main__":
    main()
