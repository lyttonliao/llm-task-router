"""Interactive terminal chat client. Authenticates once per provider at
startup (offering to run each provider's own login command interactively),
then routes each message independently through router.route_and_run() -
stateless, single-shot, no conversation history sent to the model. See
CLAUDE.md, "llm-chat: interactive terminal client" for the v2 multi-turn
seam (claude -p --session-id/--continue/--resume, codex exec resume) this
deliberately does not wire up.
"""

import sys

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
    print_fn("\nAuthenticated providers:")
    for name in sorted(authenticated_providers):
        print_fn(f"  {name}: known models (reference only) = {known_models_for(name)}")

    print_fn("\nRoutable tiers:")
    for tier, (provider, model) in routable.items():
        print_fn(f"  {tier}: {provider}/{model}")

    for tier, (provider, model) in unroutable.items():
        print_fn(
            f"[warning] tier '{tier}' maps to '{provider}/{model}' but you're not "
            f"authenticated with {provider} - messages classified into this tier "
            f"will fail with an auth error until you log in."
        )
    print_fn("")


def format_response(decision, result) -> str:
    header = f"[{decision.provider}/{decision.model}, tier={decision.tier}]"
    if result.error:
        return f"{header} error: {result.error}"
    return f"{header}\n{result.text}\n(cost ${result.cost_usd:.4f}, {result.duration_ms}ms)"


def chat_loop(*, input_fn=input, print_fn=print) -> None:
    """Stateless, single-shot per message - every line is routed
    independently via route_and_run() with no conversation history. A
    message that routes to a provider the user skipped at login needs no
    special handling here: invoke() already runs its own check_auth() first,
    so it naturally comes back as ProviderResult(error="auth check failed:
    ..."), which flows through format_response()'s normal error path."""
    while True:
        try:
            line = input_fn("you> ")
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

        request = TaskRequest(description=line)
        try:
            decision, result = route_and_run(request)
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

        print_fn(format_response(decision, result))


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
