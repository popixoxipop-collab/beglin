from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .execution_adapter import parse_engine_log
from .reference_evaluator import score_request


PLAN_SCHEMA = "beglin-p9-single-target-ablation-plan/1"
ARTIFACT_SCHEMA = "beglin-p9-single-target-artifact/1"
ANALYSIS_SCHEMA = "beglin-p9-single-target-boundary-analysis/1"
IDENTITY_FIELDS = (
    "source_commit",
    "source_tree",
    "binary_sha256",
    "checkpoint_sha256",
    "model_id",
    "hardware_id",
    "backend",
    "seed",
    "diagnostic_corpus_sha256",
    "runtime_config_hash",
)


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(stable_json(value).encode("utf-8"))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top level must be an object")
    return value


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"plan.schema must be {PLAN_SCHEMA}")
    if plan.get("diagnostic_prompts") != ["constraint_words", "ppl_short_1", "short_reasoning"]:
        raise ValueError("diagnostic prompt set or order changed")
    if plan.get("repetitions") != 2 or plan.get("expected_requests_per_cell") != 6:
        raise ValueError("diagnostic request shape changed")
    cells = plan.get("cells")
    if not isinstance(cells, list) or len(cells) != 6:
        raise ValueError("plan must contain six ablation cells")
    ids = []
    pairs = set()
    for cell in cells:
        cell_id = cell.get("cell_id")
        target = cell.get("target") or {}
        role = target.get("role")
        layer = target.get("layer")
        n = cell.get("n")
        if not isinstance(cell_id, str) or not re.fullmatch(r"[a-z0-9_]+", cell_id):
            raise ValueError("invalid cell_id")
        if (role, layer) not in {("kv_a_proj_with_mqa", 11), ("shared_up_proj", 3)}:
            raise ValueError(f"{cell_id}: unexpected target")
        if n not in {5, 6, 7}:
            raise ValueError(f"{cell_id}: n must be 5, 6, or 7")
        if cell.get("promotion_lines") != [f"{role} {layer} {n}"]:
            raise ValueError(f"{cell_id}: promotion must contain exactly its target")
        ids.append(cell_id)
        pairs.add((role, layer, n))
    expected_pairs = {
        (role, layer, n)
        for role, layer in (("kv_a_proj_with_mqa", 11), ("shared_up_proj", 3))
        for n in (5, 6, 7)
    }
    if len(set(ids)) != 6 or pairs != expected_pairs:
        raise ValueError("ablation cells are incomplete or duplicated")
    if plan.get("reference", {}).get("promotion_lines") != []:
        raise ValueError("reference promotion_lines must be empty")
    if plan.get("production_write_allowed") is not False:
        raise ValueError("plan must deny production writes")
    return plan


def cell_policy(plan: dict[str, Any], cell_id: str) -> dict[str, Any]:
    if cell_id == "reference":
        return plan["reference"]
    for cell in plan["cells"]:
        if cell["cell_id"] == cell_id:
            return cell
    raise ValueError(f"unknown ablation cell: {cell_id}")


def tokenizer_decoder(tokenizer_json: Path | None):
    if tokenizer_json is None:
        return lambda _: None
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    return lambda values: tokenizer.decode(values, skip_special_tokens=True)


def normalize(
    *,
    plan_path: Path,
    token_fixture_path: Path,
    log_path: Path,
    identity_path: Path,
    cell_id: str,
    run_id: str,
    tokenizer_json: Path | None,
    reference_path: Path | None,
) -> dict[str, Any]:
    plan = validate_plan(load_json(plan_path))
    policy = cell_policy(plan, cell_id)
    token_fixture = load_json(token_fixture_path)
    token_entries = {row["prompt_id"]: row for row in token_fixture["entries"]}
    entries = [token_entries[prompt_id] for prompt_id in plan["diagnostic_prompts"]]
    rows = parse_engine_log(log_path.read_text(encoding="utf-8"))
    if len(rows) != plan["expected_requests_per_cell"]:
        raise ValueError(
            f"expected {plan['expected_requests_per_cell']} request rows, got {len(rows)}"
        )
    identity = load_json(identity_path)
    decode = tokenizer_decoder(tokenizer_json)
    reference = load_artifact(reference_path) if reference_path else None
    if cell_id == "reference" and reference is not None:
        raise ValueError("reference cell must not receive --reference")
    if cell_id != "reference" and reference is None:
        raise ValueError("candidate cell requires --reference")
    reference_map = {}
    if reference:
        reference_map = {
            (row["prompt_id"], row["repetition"]): row
            for row in reference["requests"]
        }

    requests = []
    for row in rows:
        prompt_index = row["prompt"]
        expected_prompt_index = row["req"] % len(entries)
        if prompt_index != expected_prompt_index:
            raise ValueError(
                f"request {row['req']} prompt={prompt_index}, expected {expected_prompt_index}"
            )
        entry = entries[prompt_index]
        if row["context_tokens"] != entry["context_tokens"]:
            raise ValueError(f"request {row['req']} context token mismatch")
        repetition = row["req"] // len(entries)
        request = {
            "request_id": f"{run_id}:{entry['prompt_id']}:{repetition}",
            "prompt_id": entry["prompt_id"],
            "repetition": repetition,
            "context_tokens": row["context_tokens"],
            "output_token_ids": row["tokens"],
            "output_text": decode(row["tokens"]),
            "finite_logits": row["finite_logits"],
            "nll_sum": row["nll_sum"],
            "nll_token_count": row["nll_count"],
            "router_near_tie_count": row["router_near"],
            "router_decision_count": row["router_decisions"],
            "activation": {
                "kv_a_proj_with_mqa_l11": {
                    "mean_abs": row["kva_mean_abs"],
                    "rms": row["kva_rms"],
                },
                "shared_up_proj_l3": {
                    "mean_abs": row["shared_up_mean_abs"],
                    "rms": row["shared_up_rms"],
                },
            },
            "reference_agreement": 1.0,
        }
        if reference:
            key = (entry["prompt_id"], repetition)
            if key not in reference_map:
                raise ValueError(f"reference missing request {key}")
            request["reference_agreement"] = score_request(reference_map[key], request)
        requests.append(request)

    cell = {
        "cell_id": cell_id,
        "kind": "reference" if cell_id == "reference" else "single_target_candidate",
        "n": policy.get("n"),
        "target": policy.get("target"),
        "promotion_lines": policy["promotion_lines"],
        "policy_hash": sha256_json(policy),
    }
    payload = {
        "schema": ARTIFACT_SCHEMA,
        "run_id": run_id,
        "cell": cell,
        "identity": identity,
        "plan_sha256": sha256_bytes(plan_path.read_bytes()),
        "raw_log_sha256": sha256_bytes(log_path.read_bytes()),
        "requests": requests,
    }
    payload["artifact_hash"] = sha256_json(payload)
    return payload


def load_artifact(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise ValueError("artifact path is required")
    artifact = load_json(path)
    claimed = artifact.get("artifact_hash")
    core = dict(artifact)
    core.pop("artifact_hash", None)
    if artifact.get("schema") != ARTIFACT_SCHEMA or claimed != sha256_json(core):
        raise ValueError(f"{path}: invalid ablation artifact")
    if len(artifact.get("requests", [])) != 6:
        raise ValueError(f"{path}: expected six requests")
    return artifact


def request_map(artifact: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (row["prompt_id"], row["repetition"]): row
        for row in artifact["requests"]
    }


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def causal_class(
    reference: list[int],
    kv_only: list[int],
    shared_only: list[int],
    combined: list[int],
) -> str:
    kv_changed = kv_only != reference
    shared_changed = shared_only != reference
    combined_changed = combined != reference
    if not combined_changed:
        if not kv_changed and not shared_changed:
            return "stable_all"
        return "isolated_change_cancelled_in_combination"
    if kv_only == combined and not shared_changed:
        return "kv_a_proj_with_mqa_l11"
    if shared_only == combined and not kv_changed:
        return "shared_up_proj_l3"
    if kv_only == combined and shared_only == combined:
        return "both_independently_same_output"
    if not kv_changed and not shared_changed:
        return "joint_interaction_only"
    if kv_changed and not shared_changed:
        return "kv_a_proj_with_mqa_l11_with_interaction"
    if shared_changed and not kv_changed:
        return "shared_up_proj_l3_with_interaction"
    return "both_or_interaction"


def analyze(
    *,
    plan_path: Path,
    reference_path: Path,
    candidate_paths: list[Path],
    combined_dir: Path,
) -> dict[str, Any]:
    plan = validate_plan(load_json(plan_path))
    reference = load_artifact(reference_path)
    candidates = [load_artifact(path) for path in candidate_paths]
    by_id = {row["cell"]["cell_id"]: row for row in candidates}
    expected_ids = {row["cell_id"] for row in plan["cells"]}
    if set(by_id) != expected_ids or len(candidates) != 6:
        raise ValueError("candidate artifact set does not match the six-cell plan")
    reference_identity = reference["identity"]
    for cell_id, candidate in by_id.items():
        differences = [
            field
            for field in IDENTITY_FIELDS
            if candidate["identity"].get(field) != reference_identity.get(field)
        ]
        if differences:
            raise ValueError(f"{cell_id}: control identity mismatch: {differences}")

    reference_requests = request_map(reference)
    cell_summaries = []
    for cell in plan["cells"]:
        artifact = by_id[cell["cell_id"]]
        requests = request_map(artifact)
        changed = []
        exact_matches = 0
        deterministic = True
        finite = True
        for prompt_id in plan["diagnostic_prompts"]:
            repetitions = []
            for repetition in range(plan["repetitions"]):
                key = (prompt_id, repetition)
                left = reference_requests[key]["output_token_ids"]
                right = requests[key]["output_token_ids"]
                repetitions.append(right)
                finite = finite and requests[key]["finite_logits"]
                difference = first_difference(left, right)
                if difference is None:
                    exact_matches += 1
                else:
                    changed.append(
                        {
                            "prompt_id": prompt_id,
                            "repetition": repetition,
                            "first_difference_token_index": difference,
                            "reference_token_id": left[difference] if difference < len(left) else None,
                            "candidate_token_id": right[difference] if difference < len(right) else None,
                            "reference_output": reference_requests[key]["output_text"],
                            "candidate_output": requests[key]["output_text"],
                        }
                    )
            deterministic = deterministic and repetitions[0] == repetitions[1]
        cell_summaries.append(
            {
                "cell_id": cell["cell_id"],
                "target": cell["target"],
                "n": cell["n"],
                "artifact_hash": artifact["artifact_hash"],
                "finite_logits": finite,
                "deterministic": deterministic,
                "exact_matches": exact_matches,
                "total_requests": plan["expected_requests_per_cell"],
                "exact_match_rate": exact_matches / plan["expected_requests_per_cell"],
                "changed": changed,
            }
        )

    combined = {
        n: load_json(combined_dir / f"n{n}.json")
        for n in (5, 6, 7)
    }
    combined_maps = {
        n: {
            (row["prompt_id"], row["repetition"]): row
            for row in artifact["requests"]
        }
        for n, artifact in combined.items()
    }
    attribution = []
    for n in (5, 6, 7):
        kv = request_map(by_id[f"kv_l11_n{n}"])
        shared = request_map(by_id[f"shared_l3_n{n}"])
        for prompt_id in plan["diagnostic_prompts"]:
            for repetition in range(plan["repetitions"]):
                key = (prompt_id, repetition)
                ref_tokens = reference_requests[key]["output_token_ids"]
                combined_tokens = combined_maps[n][key]["output_token_ids"]
                attribution.append(
                    {
                        "n": n,
                        "prompt_id": prompt_id,
                        "repetition": repetition,
                        "combined_changed": combined_tokens != ref_tokens,
                        "kv_only_changed": kv[key]["output_token_ids"] != ref_tokens,
                        "shared_only_changed": shared[key]["output_token_ids"] != ref_tokens,
                        "causal_class": causal_class(
                            ref_tokens,
                            kv[key]["output_token_ids"],
                            shared[key]["output_token_ids"],
                            combined_tokens,
                        ),
                        "combined_first_difference_token_index": first_difference(ref_tokens, combined_tokens),
                        "kv_first_difference_token_index": first_difference(ref_tokens, kv[key]["output_token_ids"]),
                        "shared_first_difference_token_index": first_difference(ref_tokens, shared[key]["output_token_ids"]),
                    }
                )

    target_boundaries = []
    for role, layer, prefix in (
        ("kv_a_proj_with_mqa", 11, "kv_l11"),
        ("shared_up_proj", 3, "shared_l3"),
    ):
        rows = [next(row for row in cell_summaries if row["cell_id"] == f"{prefix}_n{n}") for n in (5, 6, 7)]
        target_boundaries.append(
            {
                "target": {"role": role, "layer": layer},
                "matching_ns": [row["n"] for row in rows if not row["changed"]],
                "divergent_ns": [row["n"] for row in rows if row["changed"]],
                "changed_prompt_ids_by_n": {
                    str(row["n"]): sorted({item["prompt_id"] for item in row["changed"]})
                    for row in rows
                },
                "monotonic_threshold_claimed": False,
            }
        )

    all_finite = all(row["finite_logits"] for row in cell_summaries)
    all_deterministic = all(row["deterministic"] for row in cell_summaries)
    combined_changes = [row for row in attribution if row["combined_changed"]]
    unresolved = [
        row
        for row in combined_changes
        if row["causal_class"] in {"both_or_interaction", "joint_interaction_only"}
    ]
    status = (
        "P9_SINGLE_TARGET_BOUNDARY_RESOLVED"
        if all_finite and all_deterministic and not unresolved
        else "P9_SINGLE_TARGET_BOUNDARY_INTERACTION_FOUND"
        if all_finite and all_deterministic
        else "P9_SINGLE_TARGET_ABLATION_INVALID"
    )
    payload = {
        "schema": ANALYSIS_SCHEMA,
        "status": status,
        "plan_sha256": sha256_bytes(plan_path.read_bytes()),
        "reference_artifact_hash": reference["artifact_hash"],
        "combined_artifact_hashes": {
            f"n{n}": combined[n].get("artifact_hash") for n in (5, 6, 7)
        },
        "control_identity": reference_identity,
        "cells": cell_summaries,
        "target_boundaries": target_boundaries,
        "combined_attribution": attribution,
        "unresolved_combined_changes": unresolved,
        "all_finite_logits": all_finite,
        "all_deterministic": all_deterministic,
        "production_write_allowed": False,
    }
    payload["analysis_hash"] = sha256_json(payload)
    return payload


def write_or_print(value: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def self_test(plan_path: Path) -> dict[str, Any]:
    plan = validate_plan(load_json(plan_path))
    ref = [1, 2, 3]
    assert causal_class(ref, [1, 2, 4], ref, [1, 2, 4]) == "kv_a_proj_with_mqa_l11"
    assert causal_class(ref, ref, [1, 2, 4], [1, 2, 4]) == "shared_up_proj_l3"
    assert causal_class(ref, ref, ref, [1, 2, 4]) == "joint_interaction_only"
    assert causal_class(ref, [1, 2, 4], [1, 5, 3], [1, 9, 3]) == "both_or_interaction"
    return {
        "schema": "beglin-p9-single-target-ablation-self-test/1",
        "status": "PASS",
        "plan_sha256": sha256_bytes(plan_path.read_bytes()),
        "cell_count": len(plan["cells"]),
        "causal_classes_tested": 4,
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    default_plan = repo / "configs/quality/p9_single_target_ablation_v1.json"
    default_fixture = repo / "configs/quality/p9_token_fixture_v1.json"
    parser = argparse.ArgumentParser(description="Normalize and analyze P9 single-target ablations.")
    sub = parser.add_subparsers(dest="command", required=True)

    test_parser = sub.add_parser("self-test")
    test_parser.add_argument("--plan", type=Path, default=default_plan)

    normalize_parser = sub.add_parser("normalize")
    normalize_parser.add_argument("--plan", type=Path, default=default_plan)
    normalize_parser.add_argument("--token-fixture", type=Path, default=default_fixture)
    normalize_parser.add_argument("--log", type=Path, required=True)
    normalize_parser.add_argument("--identity", type=Path, required=True)
    normalize_parser.add_argument("--cell", required=True)
    normalize_parser.add_argument("--run-id", required=True)
    normalize_parser.add_argument("--tokenizer-json", type=Path)
    normalize_parser.add_argument("--reference", type=Path)
    normalize_parser.add_argument("--output", type=Path)

    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--plan", type=Path, default=default_plan)
    analyze_parser.add_argument("--reference", type=Path, required=True)
    analyze_parser.add_argument("--candidate", type=Path, action="append", required=True)
    analyze_parser.add_argument("--combined-dir", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path)

    args = parser.parse_args()
    if args.command == "self-test":
        write_or_print(self_test(args.plan), None)
        return
    if args.command == "normalize":
        result = normalize(
            plan_path=args.plan,
            token_fixture_path=args.token_fixture,
            log_path=args.log,
            identity_path=args.identity,
            cell_id=args.cell,
            run_id=args.run_id,
            tokenizer_json=args.tokenizer_json,
            reference_path=args.reference,
        )
        write_or_print(result, args.output)
        return
    result = analyze(
        plan_path=args.plan,
        reference_path=args.reference,
        candidate_paths=args.candidate,
        combined_dir=args.combined_dir,
    )
    write_or_print(result, args.output)


if __name__ == "__main__":
    main()
