"""Drift audit for tier 2's continuous-learning classifier (see
/Users/lliao/.claude/plans/harmonic-greeting-squid.md for the full design
this implements).

`tier2_classifier.AGREEMENT_THRESHOLD` was tuned via leave-one-out against
only the 98-row seed set - its own comment says to revisit that once real
(non-seed) query volume accumulates. This script re-runs the same
leave-one-out methodology against the *live* `routing_examples` table
instead, and separately flags `source='llm_fallback'` rows whose own label
now disagrees with what their neighbors would confidently vote - a cheap
signal that an earlier LLM-fallback write-back was probably wrong. It does
NOT auto-correct anything; flagged rows are printed for manual review only,
same "review, don't auto-fix" discipline used everywhere else tier 2 defers
to a human/judge rather than trusting a heuristic to self-correct.

Requires DATABASE_URL to already be set in the environment, same as
seed_vector_store.py - vector_store.py raises RuntimeError on its own the
first time it needs a connection, so that check isn't duplicated here.

Usage:
    DATABASE_URL=postgresql:///llm_task_router uv run python scripts/audit_tier2.py
    DATABASE_URL=... uv run python scripts/audit_tier2.py --label-column task_type
    DATABASE_URL=... uv run python scripts/audit_tier2.py --judge-flagged
"""

import argparse
from collections import Counter

import numpy as np

from llm_task_router import tier2_classifier, vector_store
from llm_task_router.providers import claude_cli
from llm_task_router.vector_store import LabeledExample

# Accuracy a bucket must clear to be considered "trustworthy enough to skip
# the LLM fallback" - close to the 87.1% originally observed at >=4/5 on the
# 98-row seed set (see tier2_classifier.py's AGREEMENT_THRESHOLD comment).
ACCEPTABLE_ACCURACY_BAR = 0.85

_AGREEMENT_BUCKETS = [
    ("unconditional (>=1/5)", 0.0),
    (">=3/5", 0.6),
    (">=4/5", 0.8),
    ("5/5", 1.0),
]

_JUDGE_PROMPTS = {
    "task_type": (
        "A stored example labels this task request's category as {label}. "
        "Judging independently, is {label} the correct category for this "
        "request?\n\nTask request:\n\"\"\"{description}\"\"\"\n\n"
        "Respond with exactly one word: YES or NO. No explanation, no "
        "punctuation, nothing else."
    ),
    "is_high_stakes": (
        "A stored example labels this task request as {label_word} genuinely "
        "high-stakes (real production impact, security/compliance/regulatory "
        "exposure, irreversible consequences, or meaningful scale). Judging "
        "independently, is that label correct?\n\nTask request:\n"
        "\"\"\"{description}\"\"\"\n\nRespond with exactly one word: YES or NO. "
        "No explanation, no punctuation, nothing else."
    ),
}
_JUDGE_SYSTEM_PROMPT = "You are a second-opinion label auditor. Respond with exactly one word and nothing else."


def _leave_one_out_neighbors(examples: list[LabeledExample], k: int) -> list[list[LabeledExample]]:
    """For each example, its k nearest *other* examples by cosine similarity
    - the same neighbor set tier 2's own nearest_neighbors() would have
    returned had this row not existed in the store yet."""
    matrix = np.array([ex.embedding for ex in examples])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / norms
    similarity = normalized @ normalized.T

    neighbors = []
    for i in range(len(examples)):
        row_similarity = similarity[i].copy()
        row_similarity[i] = -np.inf  # exclude self
        top_k_idx = np.argsort(row_similarity)[::-1][:k]
        neighbors.append([examples[j] for j in top_k_idx])
    return neighbors


def _majority(neighbors: list[LabeledExample]) -> tuple[str | bool, float] | None:
    """Majority label and agreement fraction among the given neighbors, or
    None if there aren't enough of them (mirrors tier2_classifier._nn_majority's
    NEIGHBOR_K floor - fewer neighbors than that isn't a real signal)."""
    if not neighbors:
        return None
    label, count = Counter(n.label for n in neighbors).most_common(1)[0]
    return label, count / len(neighbors)


def _judge_label(description: str, label_column: str, label: str | bool) -> bool | None:
    label_word = "YES" if label is True else "NO" if label is False else label
    prompt = _JUDGE_PROMPTS[label_column].format(description=description, label=label, label_word=label_word)
    try:
        result = claude_cli.invoke(prompt, "haiku", system_prompt=_JUDGE_SYSTEM_PROMPT, disable_tools=True)
    except Exception:
        return None
    if result.error:
        return None
    text = result.text.strip().upper()
    if "YES" in text:
        return True
    if "NO" in text:
        return False
    return None


def _print_agreement_curve(examples: list[LabeledExample], neighbors: list[list[LabeledExample]], k: int) -> None:
    print(f"  {'bucket':<24}{'accuracy':>10}{'coverage':>10}{'n':>6}")
    for bucket_label, min_fraction in _AGREEMENT_BUCKETS:
        rows_in_bucket = []
        for example, example_neighbors in zip(examples, neighbors):
            result = _majority(example_neighbors)
            if result is None or len(example_neighbors) < k:
                continue
            majority_label, agreement_fraction = result
            if agreement_fraction >= min_fraction:
                rows_in_bucket.append(majority_label == example.label)
        n = len(rows_in_bucket)
        accuracy = sum(rows_in_bucket) / n if n else float("nan")
        coverage = n / len(examples) if examples else float("nan")
        print(f"  {bucket_label:<24}{accuracy:>9.1%}{coverage:>10.1%}{n:>6}")


def _current_threshold_verdict(
    examples: list[LabeledExample], neighbors: list[list[LabeledExample]], k: int, current_threshold: float
) -> str:
    at_threshold = []
    for example, example_neighbors in zip(examples, neighbors):
        if len(example_neighbors) < k:
            continue
        result = _majority(example_neighbors)
        if result is None:
            continue
        majority_label, agreement_fraction = result
        if agreement_fraction >= current_threshold:
            at_threshold.append(majority_label == example.label)
    if not at_threshold:
        return f"AGREEMENT_THRESHOLD={current_threshold}: no rows reach this bucket yet on live data - can't verify."
    accuracy = sum(at_threshold) / len(at_threshold)
    verdict = "holds" if accuracy >= ACCEPTABLE_ACCURACY_BAR else "consider raising - accuracy has drifted below the bar"
    return (
        f"AGREEMENT_THRESHOLD={current_threshold}: {accuracy:.1%} accuracy over {len(at_threshold)} live rows "
        f"(bar: {ACCEPTABLE_ACCURACY_BAR:.0%}) - {verdict}"
    )


def _flag_suspect_llm_fallback_rows(
    examples: list[LabeledExample], neighbors: list[list[LabeledExample]], k: int, threshold: float
) -> list[tuple[LabeledExample, str | bool, float]]:
    """Rows written back by tier 2's own LLM fallback whose neighbors now
    confidently disagree with the stored label - a signal the original
    fallback resolution was probably wrong, not proof of it."""
    flagged = []
    for example, example_neighbors in zip(examples, neighbors):
        if example.source != "llm_fallback" or len(example_neighbors) < k:
            continue
        result = _majority(example_neighbors)
        if result is None:
            continue
        majority_label, agreement_fraction = result
        if agreement_fraction >= threshold and majority_label != example.label:
            flagged.append((example, majority_label, agreement_fraction))
    return flagged


def _audit_label_column(label_column: str, k: int, threshold: float, judge_flagged: bool) -> None:
    print(f"\n=== {label_column} ===")
    examples = vector_store.all_labeled_examples(label_column)
    print(f"{len(examples)} labeled row(s).")
    if len(examples) <= k:
        print(f"Fewer than {k + 1} rows - not enough to compute leave-one-out neighbors yet.")
        return

    neighbors = _leave_one_out_neighbors(examples, k)

    print("\nAgreement curve (leave-one-out, live data):")
    _print_agreement_curve(examples, neighbors, k)
    print()
    print(_current_threshold_verdict(examples, neighbors, k, threshold))

    flagged = _flag_suspect_llm_fallback_rows(examples, neighbors, k, threshold)
    print(f"\n{len(flagged)} suspect llm_fallback row(s) (neighbors confidently disagree with the stored label):")
    for example, majority_label, agreement_fraction in flagged:
        print(f"  id={example.id} label={example.label!r} neighbors_say={majority_label!r} "
              f"agreement={agreement_fraction:.0%}")
        print(f"    {example.description!r}")
        if judge_flagged:
            verdict = _judge_label(example.description, label_column, example.label)
            print(f"    judge second opinion: label correct = {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-validate tier 2's AGREEMENT_THRESHOLD against live data.")
    parser.add_argument(
        "--label-column",
        choices=["task_type", "is_high_stakes"],
        help="Restrict to one label column (default: audit both).",
    )
    parser.add_argument("--neighbor-k", type=int, default=tier2_classifier.NEIGHBOR_K)
    parser.add_argument("--agreement-threshold", type=float, default=tier2_classifier.AGREEMENT_THRESHOLD)
    parser.add_argument(
        "--judge-flagged",
        action="store_true",
        help="For each flagged row, make one cheap LLM call for a second opinion on whether the "
        "stored label is correct (real cost per flagged row - off by default).",
    )
    args = parser.parse_args()

    label_columns = [args.label_column] if args.label_column else list(vector_store.VALID_LABEL_COLUMNS)
    for label_column in label_columns:
        _audit_label_column(label_column, args.neighbor_k, args.agreement_threshold, args.judge_flagged)


if __name__ == "__main__":
    main()
