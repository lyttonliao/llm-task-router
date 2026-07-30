"""Interactive terminal chat client. Authenticates once per provider at
startup (offering to run each provider's own login command interactively),
then routes each message through router.route_and_run(). Every message in a
chat_loop() run shares one session_id, so conversation history continues
across messages even as the classifier routes different messages to
different Claude tiers/models - see CLAUDE.md, "llm-chat: interactive
terminal client" for the real-CLI verification this rests on and for why
cross-provider continuity (Codex) isn't part of this.

Streaming + styling, added 2026-07-27: chat_loop() prints the routing header
as soon as route_and_run() resolves it (via on_decision), then streams the
model's answer live token-by-token as it arrives (via on_event ->
tui.StreamRenderer) instead of waiting for the full response and printing it
all at once - see tui.py for the ANSI styling and claude_cli.py for the
stream-json transport this rides on. format_response() is kept as the
non-streaming full-message formatter (still exercised by tests as the text
contract, and available to any future caller without live on_event wiring),
but chat_loop's success path no longer calls it directly - doing so would
print the answer a second time after it was already streamed.
"""

import sys
import uuid

from llm_task_router import tui
from llm_task_router.known_models import known_models_for
from llm_task_router.router import PROVIDERS, route_and_run
from llm_task_router.schema import TaskRequest
from llm_task_router.tiers import TIER_MODELS

HELP_TEXT = """Commands:
  /help          show this message
  /exit, /quit   leave the chat (Ctrl+D also works)
"""


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
    ONLY - never filters PROVIDERS/TIER_MODELS, route()/route_and_run() are
    called completely unmodified for every real message (see chat_loop).
    Because TIER_MODELS maps every tier to "claude" today, authenticating
    Codex only yields an empty routable dict - a real, documented
    consequence of tiers.py's current calibration state (see CLAUDE.md),
    not something this feature works around."""
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
    return f"{header}\n{result.text}\n{tui.footer(result)}"


def chat_loop(*, input_fn=input, print_fn=print, write_fn=tui.default_write) -> None:
    """One session_id, generated once here (not per message), is attached to
    every TaskRequest built in this loop - route_and_run() -> claude_cli.invoke()
    turns that into --session-id on the first call and --resume on every call
    after, so conversation history continues even as the classifier sends
    different messages to different Claude tiers. A message that routes to a
    provider the user skipped at login needs no special handling here:
    invoke() already runs its own check_auth() first, so it naturally comes
    back as ProviderResult(error="auth check failed: ..."), which flows
    through the same error-print path below.

    write_fn is separate from print_fn: print_fn emits discrete, newline-
    terminated lines (header, footer, errors - what tests assert on),
    write_fn emits raw, unterminated chunks for live token streaming
    (defaults to real stdout, see tui.default_write) - collapsing these into
    one parameter would make either the streamed text or the line-based
    assertions awkward to test, not both.

    A boxed input frame (top/bottom border around each prompt) was tried and
    removed the same day: it can't wrap around input that line-wraps in the
    terminal without raw terminal mode and a hand-rolled line editor (the
    same bigger build declined for live-redraw streaming) - see tui.py's
    module docstring for the full rationale. Plain styled prompt only."""
    session_id = str(uuid.uuid4())
    while True:
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
        if line == "/help":
            print_fn(HELP_TEXT)
            continue
        if line.startswith("/"):
            print_fn(f"unknown command: {line} (try /help)")
            continue

        request = TaskRequest(description=line, session_id=session_id)
        renderer = tui.StreamRenderer(write_fn=write_fn)
        try:
            decision, result = route_and_run(
                request,
                on_event=renderer.handle,
                on_decision=lambda d: print_fn(tui.header(d)),
            )
        except Exception as exc:
            # Deliberate broad catch, unlike every other layer of this repo:
            # provider adapters already convert every anticipated failure
            # into a ProviderResult(error=...), so route_and_run "shouldn't"
            # raise - but an interactive session dying entirely over one
            # unanticipated exception is worse UX than a one-shot process
            # exiting on the same fault (which just gets re-run). Always
            # printed in full, never swallowed silently.
            print_fn(f"[internal error] {exc!r} - message not sent, session continues")
            continue

        renderer.finish()
        if result.error:
            print_fn(tui.error_line(result.error))
        else:
            print_fn(tui.footer(result))


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
            "with - see CLAUDE.md's tier calibration status. Exiting."
        )
        sys.exit(1)

    chat_loop()


if __name__ == "__main__":
    main()
