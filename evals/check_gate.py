"""Fail the CI if eval metrics are below threshold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--min-faithfulness", type=float, default=0.85)
    parser.add_argument("--min-answer-relevance", type=float, default=0.80)
    parser.add_argument("--min-context-precision", type=float, default=None)
    parser.add_argument("--min-context-recall", type=float, default=None)
    args = parser.parse_args()

    if not args.results.exists():
        print(f"❌ Results file not found: {args.results}")
        return 1

    results = json.loads(args.results.read_text())

    checks = [
        ("faithfulness", args.min_faithfulness),
        ("answer_relevancy", args.min_answer_relevance),
        ("context_precision", args.min_context_precision),
        ("context_recall", args.min_context_recall),
    ]

    failed = []
    for name, threshold in checks:
        if threshold is None:
            continue
        actual = results.get(name)
        if actual is None:
            failed.append(f"{name}: missing from results")
            continue
        if actual < threshold:
            failed.append(f"{name} {actual:.3f} < {threshold:.3f}")

    print("=" * 60)
    print("EVAL GATE")
    print("=" * 60)
    for name, _ in checks:
        val = results.get(name)
        if val is not None:
            print(f"  {name:22s}  {val:.3f}")
    print("=" * 60)

    if failed:
        print("\n❌ EVAL GATE FAILED:")
        for f in failed:
            print(f"  · {f}")
        return 1

    print("\n✅ EVAL GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())