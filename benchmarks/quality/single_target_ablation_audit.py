from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


AUDIT_SCHEMA = "beglin-p9-single-target-independent-audit/1"
RAW_SCHEMA = "beglin-p9-single-target-physical-run/1"
ARTIFACT_SCHEMA = "beglin-p9-single-target-artifact/1"
ANALYSIS_SCHEMA = "beglin-p9-single-target-boundary-analysis/1"
PROMPTS = ("constraint_words", "ppl_short_1", "short_reasoning")
CELLS = {
    "reference": [],
    "kv_l11_n5": ["kv_a_proj_with_mqa 11 5"],
    "kv_l11_n6": ["kv_a_proj_with_mqa 11 6"],
    "kv_l11_n7": ["kv_a_proj_with_mqa 11 7"],
    "shared_l3_n5": ["shared_up_proj 3 5"],
    "shared_l3_n6": ["shared_up_proj 3 6"],
    "shared_l3_n7": ["shared_up_proj 3 7"],
}
REQUEST_RE = re.compile(
    r"^P9_QUALITY_REQUEST_V1 req=(\d+) prompt=(\d+) context_tokens=(\d+) "
    r"finite_logits=(\d).* nout=(\d+) tokens:(.*)$"
)


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_json(value: Any) -> str:
    return sha256_bytes(stable_json(value).encode("utf-8"))


def hash_without(value: dict[str, Any], field: str) -> str:
    core = dict(value)
    core.pop(field, None)
    return sha256_json(core)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top level must be an object")
    return value


def request_map(artifact: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(row["prompt_id"], row["repetition"]): row for row in artifact["requests"]}


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def parse_raw_requests(path: Path) -> dict[int, dict[str, Any]]:
    parsed: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = REQUEST_RE.match(line)
        if not match:
            continue
        request_id, prompt, context, finite, nout, token_text = match.groups()
        tokens = [int(value) for value in token_text.split()]
        if len(tokens) != int(nout):
            raise ValueError(f"{path}: request {request_id} nout mismatch")
        parsed[int(request_id)] = {
            "prompt": int(prompt),
            "context_tokens": int(context),
            "finite_logits": finite == "1",
            "output_token_ids": tokens,
        }
    return parsed


def effect_class(reference: list[int], kv: list[int], shared: list[int], combined: list[int]) -> str:
    kv_changed = kv != reference
    shared_changed = shared != reference
    combined_changed = combined != reference
    if not combined_changed:
        if not kv_changed and not shared_changed:
            return "stable_all"
        if kv_changed and not shared_changed:
            return "kv_change_cancelled_in_combination"
        if shared_changed and not kv_changed:
            return "shared_change_cancelled_in_combination"
        return "both_changes_cancelled_in_combination"
    if kv_changed and shared_changed:
        return "both_independently_causal"
    if kv_changed:
        return "kv_a_proj_with_mqa_l11_causal"
    if shared_changed:
        return "shared_up_proj_l3_causal"
    return "joint_interaction_only"


def exact_source(reference: list[int], kv: list[int], shared: list[int], combined: list[int]) -> str:
    matches = []
    if combined == reference:
        matches.append("reference")
    if combined == kv:
        matches.append("kv_only")
    if combined == shared:
        matches.append("shared_only")
    return "+".join(matches) if matches else "interaction_distinct"


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently audit a P9 single-target ablation.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--combined-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--observed-at", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    def require(check_id: str, passed: bool, detail: Any = None) -> None:
        checks.append({"id": check_id, "status": "PASS" if passed else "FAIL", "detail": detail})
        if not passed:
            failures.append(check_id)

    raw_dir = args.run_dir / "raw"
    raw_path = raw_dir / "result.json"
    identity_path = raw_dir / "identity.json"
    metadata_path = args.run_dir / "job_metadata.json"
    analysis_path = args.run_dir / "boundary_analysis.json"
    raw = load_json(raw_path)
    identity = load_json(identity_path)
    metadata = load_json(metadata_path)
    analysis = load_json(analysis_path)
    plan = load_json(args.plan)
    source_audit = load_json(args.source_audit)
    registration = load_json(args.registration)

    require("raw_schema", raw.get("schema") == RAW_SCHEMA, raw.get("schema"))
    require("raw_status", raw.get("status") == "P9_SINGLE_TARGET_RAW_PASS", raw.get("status"))
    require("raw_result_hash", raw.get("result_sha256") == hash_without(raw, "result_sha256"))
    require("raw_production_denied", raw.get("production_write_allowed") is False)
    require("runner_hash", raw.get("runner_sha256") == sha256_file(args.runner), raw.get("runner_sha256"))
    require("plan_hash", raw.get("ablation_plan_sha256") == sha256_file(args.plan))
    require("plan_embedded", raw.get("ablation_plan") == plan)
    require("identity_file", identity == raw.get("identity"))
    require("metal_backend", identity.get("backend") == "mlx_metal", identity.get("backend"))
    require("job_succeeded", metadata.get("state") == "SUCCEEDED" and metadata.get("exit_code") == 0)
    require("job_exact_argv", metadata.get("argv") == ["python3", args.runner.name])
    require("job_confirmed", metadata.get("reconciliation_status") == "confirmed" and metadata.get("observation_stale") is False)
    require("registration_status", registration.get("status") == "P9_SINGLE_TARGET_FIXTURE_REGISTERED")
    require("registration_fixture_hash", registration.get("admission", {}).get("expected_fixture_sha256") == sha256_file(args.runner))
    require("registration_job", registration.get("tailnet_job", {}).get("job_id") == metadata.get("job_id"))
    negative_probe = registration.get("live_negative_probe", {})
    require(
        "registration_live_negative_probe",
        negative_probe.get("ok") is False
        and negative_probe.get("error_code") == "INVALID_ARGUMENT"
        and negative_probe.get("argv") == ["python3", args.runner.name, "--self-test"],
    )

    require("source_audit_hash", source_audit.get("audit_sha256") == hash_without(source_audit, "audit_sha256"))
    require("source_audit_status", source_audit.get("status") == "P9_EVIDENCE_INDEPENDENT_PASS")
    source_link = plan.get("source_evidence", {})
    require("source_audit_link", source_link.get("audit_hash") == source_audit.get("audit_sha256"))
    require(
        "source_evaluation_link",
        source_link.get("evaluation_hash") == source_audit.get("inputs", {}).get("evaluation_claimed_hash"),
    )
    require("source_p9_failed", source_audit.get("conclusion", {}).get("p9_quality") == "P9_QUALITY_FAIL")
    require("source_production_denied", source_audit.get("conclusion", {}).get("production_release_authorized") is False)

    raw_rows = {row.get("cell_id"): row for row in raw.get("runs", [])}
    require("raw_cell_set", set(raw_rows) == set(CELLS), sorted(raw_rows))
    normalized: dict[str, dict[str, Any]] = {}
    normalized_rows = []
    all_finite = True
    all_deterministic = True
    for cell_id, promotion_lines in CELLS.items():
        raw_row = raw_rows.get(cell_id, {})
        log_path = raw_dir / f"{cell_id}.log"
        artifact_path = args.run_dir / f"{cell_id}.json"
        require(f"{cell_id}_raw_pass", raw_row.get("status") == "PASS")
        require(f"{cell_id}_raw_log_hash", raw_row.get("log_sha256") == sha256_file(log_path))
        require(
            f"{cell_id}_raw_shape",
            raw_row.get("request_count") == 6
            and raw_row.get("parsed_request_count") == 6
            and raw_row.get("finite_logits") is True
            and raw_row.get("metrics_complete") is True,
        )
        require(f"{cell_id}_policy", raw_row.get("policy", {}).get("promotion_lines") == promotion_lines)
        if promotion_lines:
            promotion_path = raw_dir / f"{cell_id}.promotion.txt"
            require(f"{cell_id}_promotion_file", promotion_path.read_text(encoding="utf-8") == promotion_lines[0] + "\n")
        else:
            require("reference_no_promotion", raw_row.get("promotion_path") is None)

        artifact = load_json(artifact_path)
        normalized[cell_id] = artifact
        require(f"{cell_id}_artifact_schema", artifact.get("schema") == ARTIFACT_SCHEMA)
        require(f"{cell_id}_artifact_hash", artifact.get("artifact_hash") == hash_without(artifact, "artifact_hash"))
        require(f"{cell_id}_artifact_log_hash", artifact.get("raw_log_sha256") == sha256_file(log_path))
        require(f"{cell_id}_artifact_plan_hash", artifact.get("plan_sha256") == sha256_file(args.plan))
        require(f"{cell_id}_artifact_identity", artifact.get("identity") == identity)
        require(f"{cell_id}_artifact_policy", artifact.get("cell", {}).get("promotion_lines") == promotion_lines)

        requests = request_map(artifact)
        expected_keys = {(prompt, repetition) for prompt in PROMPTS for repetition in (0, 1)}
        require(f"{cell_id}_request_keys", set(requests) == expected_keys)
        parsed = parse_raw_requests(log_path)
        require(f"{cell_id}_parsed_raw_count", set(parsed) == set(range(6)))
        finite = True
        deterministic = True
        for prompt_index, prompt in enumerate(PROMPTS):
            repetitions = []
            for repetition in (0, 1):
                request_id = repetition * len(PROMPTS) + prompt_index
                request = requests[(prompt, repetition)]
                raw_request = parsed[request_id]
                require(
                    f"{cell_id}_{prompt}_r{repetition}_raw_match",
                    request.get("output_token_ids") == raw_request["output_token_ids"]
                    and request.get("context_tokens") == raw_request["context_tokens"]
                    and request.get("finite_logits") == raw_request["finite_logits"],
                )
                finite = finite and request.get("finite_logits") is True
                repetitions.append(request.get("output_token_ids"))
            deterministic = deterministic and repetitions[0] == repetitions[1]
        require(f"{cell_id}_finite", finite)
        require(f"{cell_id}_deterministic", deterministic)
        all_finite = all_finite and finite
        all_deterministic = all_deterministic and deterministic
        normalized_rows.append(
            {
                "cell_id": cell_id,
                "artifact_hash": artifact.get("artifact_hash"),
                "file_sha256": sha256_file(artifact_path),
                "raw_log_sha256": sha256_file(log_path),
                "finite_logits": finite,
                "deterministic": deterministic,
            }
        )

    source_rows = {row["variant"]: row for row in source_audit.get("normalized_artifacts", [])}
    combined: dict[int, dict[str, Any]] = {}
    for variant in ("reference", "n5", "n6", "n7"):
        artifact_path = args.combined_dir / f"{variant}.json"
        artifact = load_json(artifact_path)
        source_row = source_rows.get(variant, {})
        require(f"source_{variant}_artifact_hash", artifact.get("artifact_hash") == hash_without(artifact, "artifact_hash"))
        require(f"source_{variant}_audit_artifact", artifact.get("artifact_hash") == source_row.get("artifact_hash"))
        require(f"source_{variant}_audit_file", sha256_file(artifact_path) == source_row.get("file_sha256"))
        if variant == "reference":
            source_reference = artifact
        else:
            combined[int(variant[1:])] = artifact

    reference_requests = request_map(normalized["reference"])
    source_reference_requests = request_map(source_reference)
    for key in reference_requests:
        require(
            f"cross_run_reference_{key[0]}_r{key[1]}",
            reference_requests[key]["output_token_ids"] == source_reference_requests[key]["output_token_ids"]
            and reference_requests[key]["context_tokens"] == source_reference_requests[key]["context_tokens"],
        )

    target_boundaries = []
    derived_cells = []
    for role, layer, prefix in (
        ("kv_a_proj_with_mqa", 11, "kv_l11"),
        ("shared_up_proj", 3, "shared_l3"),
    ):
        changed_by_n: dict[str, list[str]] = {}
        matching_ns = []
        divergent_ns = []
        for n in (5, 6, 7):
            cell_id = f"{prefix}_n{n}"
            candidate = request_map(normalized[cell_id])
            changed_prompts = sorted(
                prompt
                for prompt in PROMPTS
                if candidate[(prompt, 0)]["output_token_ids"] != reference_requests[(prompt, 0)]["output_token_ids"]
            )
            changed_by_n[str(n)] = changed_prompts
            (divergent_ns if changed_prompts else matching_ns).append(n)
            exact_matches = sum(
                candidate[(prompt, repetition)]["output_token_ids"]
                == reference_requests[(prompt, repetition)]["output_token_ids"]
                for prompt in PROMPTS
                for repetition in (0, 1)
            )
            derived_cells.append({"cell_id": cell_id, "exact_matches": exact_matches, "changed_prompt_ids": changed_prompts})
        target_boundaries.append(
            {
                "target": {"role": role, "layer": layer},
                "matching_ns": matching_ns,
                "divergent_ns": divergent_ns,
                "changed_prompt_ids_by_n": changed_by_n,
                "monotonic_threshold_claimed": False,
            }
        )

    factorial_findings = []
    for n in (5, 6, 7):
        kv = request_map(normalized[f"kv_l11_n{n}"])
        shared = request_map(normalized[f"shared_l3_n{n}"])
        combined_requests = request_map(combined[n])
        for prompt in PROMPTS:
            reference_tokens = reference_requests[(prompt, 0)]["output_token_ids"]
            kv_tokens = kv[(prompt, 0)]["output_token_ids"]
            shared_tokens = shared[(prompt, 0)]["output_token_ids"]
            combined_tokens = combined_requests[(prompt, 0)]["output_token_ids"]
            for repetition in (0, 1):
                require(
                    f"factorial_deterministic_n{n}_{prompt}_r{repetition}",
                    kv[(prompt, repetition)]["output_token_ids"] == kv_tokens
                    and shared[(prompt, repetition)]["output_token_ids"] == shared_tokens
                    and combined_requests[(prompt, repetition)]["output_token_ids"] == combined_tokens,
                )
            classification = effect_class(reference_tokens, kv_tokens, shared_tokens, combined_tokens)
            if classification != "stable_all":
                factorial_findings.append(
                    {
                        "n": n,
                        "prompt_id": prompt,
                        "effect_class": classification,
                        "combined_exact_source": exact_source(
                            reference_tokens, kv_tokens, shared_tokens, combined_tokens
                        ),
                        "kv_changed": kv_tokens != reference_tokens,
                        "shared_changed": shared_tokens != reference_tokens,
                        "combined_changed": combined_tokens != reference_tokens,
                        "first_difference_token_index": {
                            "kv": first_difference(reference_tokens, kv_tokens),
                            "shared": first_difference(reference_tokens, shared_tokens),
                            "combined": first_difference(reference_tokens, combined_tokens),
                        },
                        "outputs": {
                            "reference": reference_requests[(prompt, 0)].get("output_text"),
                            "kv_only": kv[(prompt, 0)].get("output_text"),
                            "shared_only": shared[(prompt, 0)].get("output_text"),
                            "combined": combined_requests[(prompt, 0)].get("output_text"),
                        },
                    }
                )

    require("analysis_schema", analysis.get("schema") == ANALYSIS_SCHEMA)
    require("analysis_hash", analysis.get("analysis_hash") == hash_without(analysis, "analysis_hash"))
    require("analysis_production_denied", analysis.get("production_write_allowed") is False)
    require("analysis_finite", analysis.get("all_finite_logits") is all_finite)
    require("analysis_deterministic", analysis.get("all_deterministic") is all_deterministic)
    require("analysis_boundaries", analysis.get("target_boundaries") == target_boundaries)
    analysis_cells = {row["cell_id"]: row for row in analysis.get("cells", [])}
    for row in derived_cells:
        claimed = analysis_cells.get(row["cell_id"], {})
        require(f"analysis_{row['cell_id']}_exact", claimed.get("exact_matches") == row["exact_matches"])
        require(
            f"analysis_{row['cell_id']}_changed_prompts",
            sorted({item["prompt_id"] for item in claimed.get("changed", [])}) == row["changed_prompt_ids"],
        )
    interaction_observed = any(
        row["effect_class"]
        in {
            "both_independently_causal",
            "joint_interaction_only",
            "kv_change_cancelled_in_combination",
            "shared_change_cancelled_in_combination",
            "both_changes_cancelled_in_combination",
        }
        or row["combined_exact_source"] == "interaction_distinct"
        for row in factorial_findings
    )
    require(
        "analysis_status",
        analysis.get("status") == "P9_SINGLE_TARGET_BOUNDARY_INTERACTION_FOUND"
        and interaction_observed,
        analysis.get("status"),
    )

    status = "P9_SINGLE_TARGET_EVIDENCE_INDEPENDENT_PASS" if not failures else "P9_SINGLE_TARGET_EVIDENCE_INDEPENDENT_FAIL"
    audit_core = {
        "schema": AUDIT_SCHEMA,
        "observed_at_utc": args.observed_at,
        "status": status,
        "inputs": {
            "run_dir": str(args.run_dir),
            "raw_result_sha256": sha256_file(raw_path),
            "raw_result_claimed_hash": raw.get("result_sha256"),
            "analysis_sha256": sha256_file(analysis_path),
            "analysis_claimed_hash": analysis.get("analysis_hash"),
            "plan_sha256": sha256_file(args.plan),
            "runner_sha256": sha256_file(args.runner),
            "source_audit_sha256": sha256_file(args.source_audit),
            "source_audit_claimed_hash": source_audit.get("audit_sha256"),
            "registration_sha256": sha256_file(args.registration),
            "job_metadata_sha256": sha256_file(metadata_path),
        },
        "tailnet_job": {
            key: metadata.get(key)
            for key in (
                "job_id",
                "request_hash",
                "state",
                "exit_code",
                "created_at",
                "ended_at",
                "elapsed_ms",
            )
        },
        "control_identity": identity,
        "normalized_artifacts": normalized_rows,
        "target_boundaries": target_boundaries,
        "factorial_findings": factorial_findings,
        "checks": checks,
        "failures": failures,
        "conclusion": {
            "physical_execution": raw.get("status"),
            "evidence_integrity": status,
            "all_finite_logits": all_finite,
            "all_deterministic": all_deterministic,
            "single_target_boundaries_characterized": not failures,
            "nonlinear_interaction_observed": interaction_observed,
            "production_release_authorized": False,
            "production_mutation_performed": False,
            "source_p9_quality_status": source_audit.get("conclusion", {}).get("p9_quality"),
        },
    }
    audit = {**audit_core, "audit_sha256": sha256_json(audit_core)}
    rendered = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
