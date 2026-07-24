import argparse

from llm_task_router.router import route, route_and_run
from llm_task_router.schema import DOMAINS, TASK_TYPES, TaskRequest


def run_route(description: str, task_type: str | None, domain: str | None, dry_run: bool) -> None:
    request = TaskRequest(description=description, task_type=task_type, domain=domain)

    if dry_run:
        decision = route(request)
        print(f"tier:     {decision.tier}")
        print(f"provider: {decision.provider}")
        print(f"model:    {decision.model}")
        print(f"reason:   {decision.reason}")
        return

    decision, result = route_and_run(request)
    print(f"routed to {decision.provider}/{decision.model} ({decision.reason})")
    if result.error:
        print(f"error: {result.error}")
    else:
        print(result.text)
        print(f"\n[cost: ${result.cost_usd:.4f}, {result.duration_ms}ms]")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m llm_task_router")
    sub = parser.add_subparsers(dest="command", required=True)

    p_route = sub.add_parser("route", help="classify and route a task description to a model")
    p_route.add_argument("description", help="the task to run")
    p_route.add_argument("--type", dest="task_type", choices=TASK_TYPES, help="override inferred task type")
    p_route.add_argument("--domain", choices=DOMAINS, help="override inferred domain")
    p_route.add_argument("--dry-run", action="store_true", help="print the routing decision without calling a model")

    args = parser.parse_args()

    if args.command == "route":
        run_route(args.description, args.task_type, args.domain, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
