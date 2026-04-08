"""
CI threshold checker — reads eval_results.json and exits 1 if any threshold failed.
Usage: python scripts/check_threshold.py [--file eval_results.json]
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="eval_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.file)

    if not path.exists():
        print(f"[ERROR] {args.file} not found — run eval/run_eval.py first")
        sys.exit(1)

    results = json.loads(path.read_text(encoding="utf-8"))
    thresholds = results.get("thresholds_passed", {})
    aggregate  = results.get("aggregate", {})

    print(f"Eval run: {results.get('run_id', 'unknown')}")
    print(f"Questions evaluated: {results.get('num_questions', '?')}\n")

    all_passed = True
    for key, passed in thresholds.items():
        score = aggregate.get(f"mean_{key}", 0.0)
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {key:<20} score={score:.3f}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All thresholds passed.")
        sys.exit(0)
    else:
        print("One or more thresholds FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
