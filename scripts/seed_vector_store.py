"""One-time dev tool: seeds `routing_examples` from llm-eval-harness's golden
test cases (see /Users/lliao/.claude/plans/i-believe-for-now-smooth-cook.md,
commit 2 of the tier-2 classifier plan).

Lives in scripts/, not the llm_task_router package, because it isn't shipped:
it reads a sibling repo's checkout, the one place in this codebase allowed to
do that, and only offline/one-time - never at runtime.

Requires DATABASE_URL to already be set in the environment. This script does
NOT default or validate it itself; vector_store.py raises RuntimeError on its
own the first time it needs a connection, so that check isn't duplicated
here.

Usage:
    DATABASE_URL=postgresql:///llm_task_router uv run python scripts/seed_vector_store.py
    DATABASE_URL=... uv run python scripts/seed_vector_store.py --cases-dir /path/to/cases
"""

import argparse
import json
import sys
from pathlib import Path

from llm_task_router import embeddings, vector_store
from llm_task_router.schema import TASK_TYPES

DEFAULT_CASES_DIR = "../llm-eval-harness/eval_harness/cases"

# eval_harness/cases/*.jsonl filename stem -> llm_task_router.schema.TASK_TYPES
# label. Every stem not listed here is assumed to match a TASK_TYPES value
# directly (code_gen, summarization, multi_step, code_review, refactor,
# architecture) - validated in _task_type_for_stem, not just assumed.
STEM_TO_TASK_TYPE = {"bug_triage": "triage"}


def _task_type_for_stem(stem: str) -> str:
    task_type = STEM_TO_TASK_TYPE.get(stem, stem)
    if task_type not in TASK_TYPES:
        raise ValueError(
            f"jsonl file stem {stem!r} maps to {task_type!r}, which is not a "
            f"member of TASK_TYPES {TASK_TYPES}. Add an entry to "
            "STEM_TO_TASK_TYPE, or confirm this file belongs in the seed set "
            "at all."
        )
    return task_type


def _load_cases(path: Path) -> list[dict]:
    """Reads one *.jsonl file. Doesn't assume any field beyond 'input' exists
    or matters, but does require 'input' to be present and a non-empty
    string - fails loudly on a line that doesn't satisfy that rather than
    silently embedding an empty/garbage string."""
    cases = []
    with path.open() as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            input_text = obj.get("input")
            if not isinstance(input_text, str) or not input_text.strip():
                raise ValueError(
                    f"{path}:{line_num} is missing a non-empty string "
                    f"'input' field: {obj!r}"
                )
            cases.append(obj)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed routing_examples from llm-eval-harness's golden test cases."
    )
    parser.add_argument(
        "--cases-dir",
        default=DEFAULT_CASES_DIR,
        help="Directory of *.jsonl case files to seed from (default: "
        f"{DEFAULT_CASES_DIR}, resolved relative to the current working "
        "directory - run this from the repo root).",
    )
    args = parser.parse_args()

    cases_dir = Path(args.cases_dir)
    if not cases_dir.is_dir():
        sys.exit(f"--cases-dir {cases_dir} is not a directory")

    jsonl_paths = sorted(cases_dir.glob("*.jsonl"))
    if not jsonl_paths:
        sys.exit(f"no *.jsonl files found in {cases_dir}")

    print("Fetching existing descriptions from routing_examples ...")
    already_seen = vector_store.existing_descriptions()
    print(f"  {len(already_seen)} description(s) already present.\n")

    inserted_by_file: dict[str, int] = {}
    skipped_by_file: dict[str, int] = {}
    total_inserted = 0
    total_skipped = 0

    for path in jsonl_paths:
        task_type = _task_type_for_stem(path.stem)
        cases = _load_cases(path)

        inserted = 0
        skipped = 0
        for case in cases:
            input_text = case["input"]
            if input_text in already_seen:
                skipped += 1
                continue
            embedding = embeddings.embed(input_text)
            vector_store.insert_example(
                input_text,
                embedding,
                task_type=task_type,
                is_high_stakes=None,
                source="seed",
            )
            already_seen.add(input_text)  # guards against a within-run dupe too
            inserted += 1

        inserted_by_file[path.name] = inserted
        skipped_by_file[path.name] = skipped
        total_inserted += inserted
        total_skipped += skipped
        print(f"{path.name} ({task_type}): inserted {inserted}, skipped {skipped}")

    print()
    print(f"Total: inserted {total_inserted}, skipped {total_skipped} (already present)")


if __name__ == "__main__":
    main()
