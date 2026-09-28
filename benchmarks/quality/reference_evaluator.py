from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

EVALUATOR_ID = "reference-token-agreement-v1"


def _edit_distance(left: Sequence[int], right: Sequence[int]) -> int:
    """Return Levenshtein distance with O(min(n, m)) memory."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_token in enumerate(left, start=1):
        current = [row]
        for column, right_token in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def score_token_agreement(reference: Sequence[int], candidate: Sequence[int]) -> float:
    """Bounded deterministic sequence agreement; 1 means exact token equality."""
    if not reference and not candidate:
        return 1.0
    denominator = max(len(reference), len(candidate))
    return 1.0 - (_edit_distance(reference, candidate) / denominator)


def score_request(reference: dict[str, Any], candidate: dict[str, Any]) -> float:
    left = reference.get("output_token_ids")
    right = candidate.get("output_token_ids")
    if not isinstance(left, list) or not isinstance(right, list):
        raise ValueError("reference-token-agreement-v1 requires output_token_ids on both requests")
    if not all(isinstance(value, int) and value >= 0 for value in left + right):
        raise ValueError("output_token_ids must contain non-negative integers")
    return score_token_agreement(left, right)


def main() -> None:
    parser = argparse.ArgumentParser(description=EVALUATOR_ID)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = {
        "schema": "beglin-quality-external-evaluation/1",
        "evaluator_id": EVALUATOR_ID,
        "score": score_request(reference, candidate),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
