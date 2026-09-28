from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def inspect(summary: dict[str, Any]) -> dict[str, Any]:
    runs = summary.get("runs") if isinstance(summary, dict) else None
    if not isinstance(runs, list):
        raise ValueError("summary.runs must be an array")

    activation_cells = sum(isinstance(row.get("activation"), dict) for row in runs)
    performance_cells = sum(isinstance(row.get("performance"), dict) for row in runs)
    corrected_counts = sum(
        isinstance(row.get("counts"), dict) and row["counts"].get("corrected_hits") is not None
        for row in runs
    )
    validation_cells = sum(isinstance(row.get("validation"), dict) for row in runs)

    capabilities = {
        "perplexity": False,
        "quality_score": False,
        "task_score": False,
        "finite_logits": validation_cells == len(runs) and len(runs) > 0,
        "deterministic_output": False,
        "correction_count": corrected_counts == len(runs) and len(runs) > 0,
        "promotion_count": False,
        "activation_mean_abs_rms": activation_cells > 0,
        "router_near_tie": False,
        "long_context_stability": False,
        "exact_output_tokens": False,
    }
    missing = sorted(key for key, value in capabilities.items() if not value)
    return {
        "schema": "beglin-p9-readiness-probe/1",
        "status": "P7_INSUFFICIENT_FOR_P9" if missing else "P7_P9_INPUT_COMPLETE",
        "source_schema": summary.get("schema"),
        "physical_run_id": summary.get("run_id"),
        "runner_id": summary.get("runner_id"),
        "matrix_cells": len(runs),
        "activation_cells": activation_cells,
        "performance_cells": performance_cells,
        "capabilities": capabilities,
        "missing_for_full_p9": missing,
        "note": (
            "P7 is valid physical instrumentation evidence, but it does not contain the "
            "token-level NLL/output/task/router/long-context evidence required for P9 quality PASS."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = inspect(json.loads(args.summary.read_text(encoding="utf-8")))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
