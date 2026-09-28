from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.quality.artifact import (
        QualityArtifactError,
        load_run_artifact,
        sha256_json,
        validate_run_artifact,
    )
    from benchmarks.quality.corpus import load_corpus
    from benchmarks.quality.reference_evaluator import score_request
else:
    from .artifact import QualityArtifactError, load_run_artifact, sha256_json, validate_run_artifact
    from .corpus import load_corpus
    from .reference_evaluator import score_request

ADAPTER_SCHEMA = "beglin-quality-execution-adapter/1"
TOKEN_FIXTURE_SCHEMA = "beglin-quality-token-fixture/1"
REQUEST_PREFIX = "P9_QUALITY_REQUEST_V1 "
P9_LONG_CONTEXT_POSITIONS = 16384
REQUIRED_ENGINE_MARKERS = (
    "P9_QUALITY_REQUEST_V1",
    "BEGLIN_P9_QUALITY_METRICS",
    "BEGLIN_P9_COMPACT_KV",
    "mlx_gpu_p9_metrics_config",
    "mlx_gpu_p9_metrics_read",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_hex(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise QualityArtifactError(f"{name} must be lowercase 64-hex")
    return value


def load_adapter(path: Path) -> dict[str, Any]:
    raw = _read_json(path)
    if not isinstance(raw, dict) or raw.get("schema") != ADAPTER_SCHEMA:
        raise QualityArtifactError(f"adapter.schema must be {ADAPTER_SCHEMA}")
    if not isinstance(raw.get("adapter_id"), str) or not raw["adapter_id"]:
        raise QualityArtifactError("adapter_id must be non-empty")
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or [row.get("n") for row in candidates] != [4, 5, 6, 7]:
        raise QualityArtifactError("adapter candidates must be ordered n=4,5,6,7")
    if candidates[0].get("representation") != "production_q4g64" or candidates[0].get("promotion_lines") != []:
        raise QualityArtifactError("n4 must use the existing production_q4g64 representation")
    for row in candidates[1:]:
        n = row["n"]
        expected = [f"kv_a_proj_with_mqa 11 {n}", f"shared_up_proj 3 {n}"]
        if row.get("representation") != "qNg64" or row.get("promotion_lines") != expected:
            raise QualityArtifactError(f"n{n} promotion mapping differs from the pinned P8 target pair")
    registration = raw.get("tailnet_registration")
    if not isinstance(registration, dict):
        raise QualityArtifactError("tailnet_registration missing")
    if registration.get("argv") != ["python3", "beglin_p9_quality_fixture.py"]:
        raise QualityArtifactError("tailnet registration argv must stay exact")
    build = raw.get("engine", {}).get("build", {})
    kv_cache = build.get("kv_cache") if isinstance(build, dict) else None
    if kv_cache != {
        "format": "symmetric_int8_per_head_position",
        "scale_dtype": "float16",
        "attention": "full_causal",
        "scope": "isolated_p9_long_context_build_only",
    }:
        raise QualityArtifactError("P9 compact full-causal K/V cache contract differs from the pinned adapter")
    return raw


def _decode_token_bytes(entry: dict[str, Any]) -> bytes:
    encoded = entry.get("token_ids_zlib_base64")
    if not isinstance(encoded, str):
        raise QualityArtifactError(f"{entry.get('prompt_id')}: compressed token payload missing")
    try:
        raw = zlib.decompress(base64.b64decode(encoded, validate=True))
    except (ValueError, zlib.error) as exc:
        raise QualityArtifactError(f"{entry.get('prompt_id')}: invalid compressed token payload") from exc
    expected = _require_hex(entry.get("token_bytes_sha256"), "token_bytes_sha256")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise QualityArtifactError(f"{entry.get('prompt_id')}: token byte SHA mismatch")
    if len(raw) % 4:
        raise QualityArtifactError(f"{entry.get('prompt_id')}: token bytes are not int32 aligned")
    count = len(raw) // 4
    if count != entry.get("context_tokens"):
        raise QualityArtifactError(f"{entry.get('prompt_id')}: context_tokens mismatch")
    values = struct.unpack(f"<{count}i", raw)
    if any(value < 0 for value in values):
        raise QualityArtifactError(f"{entry.get('prompt_id')}: negative token id")
    return raw


def load_token_fixture(path: Path, corpus_path: Path | None = None) -> dict[str, Any]:
    raw = _read_json(path)
    if not isinstance(raw, dict) or raw.get("schema") != TOKEN_FIXTURE_SCHEMA:
        raise QualityArtifactError(f"token fixture schema must be {TOKEN_FIXTURE_SCHEMA}")
    _require_hex(raw.get("corpus_sha256"), "corpus_sha256")
    tokenizer = raw.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise QualityArtifactError("tokenizer identity missing")
    _require_hex(tokenizer.get("tokenizer_json_sha256"), "tokenizer_json_sha256")
    _require_hex(tokenizer.get("tokenizer_config_sha256"), "tokenizer_config_sha256")
    entries = raw.get("entries")
    if not isinstance(entries, list) or not entries:
        raise QualityArtifactError("token fixture entries missing")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("prompt_id"), str):
            raise QualityArtifactError("invalid token fixture entry")
        if entry["prompt_id"] in seen:
            raise QualityArtifactError(f"duplicate token prompt: {entry['prompt_id']}")
        seen.add(entry["prompt_id"])
        if not isinstance(entry.get("max_new_tokens"), int) or entry["max_new_tokens"] < 1:
            raise QualityArtifactError(f"{entry['prompt_id']}: invalid max_new_tokens")
        _decode_token_bytes(entry)
    if corpus_path is not None:
        corpus = load_corpus(corpus_path)
        if raw["corpus_sha256"] != corpus["corpus_sha256"]:
            raise QualityArtifactError("token fixture corpus SHA does not match canonical corpus")
        if [entry["prompt_id"] for entry in entries] != [entry["id"] for entry in corpus["entries"]]:
            raise QualityArtifactError("token fixture prompt order does not match canonical corpus")
    return raw


def materialize_token_fixture(fixture: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    files = []
    for index, entry in enumerate(fixture["entries"]):
        raw = _decode_token_bytes(entry)
        filename = f"{index:02d}-{entry['prompt_id']}.i32"
        path = output_dir / filename
        path.write_bytes(raw)
        manifest_rows.append(f"{path} {entry['max_new_tokens']}")
        files.append(
            {
                "prompt_id": entry["prompt_id"],
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "context_tokens": entry["context_tokens"],
            }
        )
    manifest = output_dir / "p9-prompts.manifest"
    manifest.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    return {
        "schema": "beglin-quality-materialized-fixture/1",
        "fixture_id": fixture["fixture_id"],
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "files": files,
    }


def _source_markers(repo: Path) -> dict[str, bool]:
    source = (repo / "qwen_infer.c").read_text(encoding="utf-8")
    source += (repo / "mlx_moe.cpp").read_text(encoding="utf-8")
    source += (repo / "mlx_moe.h").read_text(encoding="utf-8")
    return {marker: marker in source for marker in REQUIRED_ENGINE_MARKERS}


def preflight(
    repo: Path,
    adapter_path: Path,
    corpus_path: Path,
) -> dict[str, Any]:
    adapter = load_adapter(adapter_path)
    fixture_path = repo / adapter["token_fixture"]
    fixture = load_token_fixture(fixture_path, corpus_path)
    required_positions = max(
        entry["context_tokens"] + entry["max_new_tokens"]
        for entry in fixture["entries"]
    )
    cmake = (repo / "CMakeLists.txt").read_text(encoding="utf-8")
    qwen = (repo / "qwen_infer.c").read_text(encoding="utf-8")
    mlx = (repo / "mlx_moe.cpp").read_text(encoding="utf-8")
    long_context_build = all(
        needle in cmake + qwen + mlx
        for needle in (
            "BEGLIN_P9_LONG_CONTEXT",
            "BEGLIN_P9_COMPACT_KV",
            "BEGLIN_MOE_CBATCH_MAXPOS",
            "BEGLIN_MLA_MAXPOS",
        )
    )
    configured_positions = P9_LONG_CONTEXT_POSITIONS if long_context_build else 32
    markers = _source_markers(repo)
    blockers = []
    if configured_positions < required_positions:
        blockers.append(
            {
                "code": "CONTEXT_WINDOW_TOO_SMALL",
                "required_positions": required_positions,
                "configured_positions": configured_positions,
            }
        )
    missing_markers = sorted(marker for marker, present in markers.items() if not present)
    if missing_markers:
        blockers.append({"code": "ENGINE_METRICS_UNBOUND", "missing_markers": missing_markers})
    registration = adapter["tailnet_registration"]
    if registration.get("status") != "REGISTERED":
        blockers.append(
            {
                "code": "TAILNET_FIXTURE_UNREGISTERED",
                "host": registration.get("host"),
                "workspace": registration.get("workspace"),
                "argv": registration.get("argv"),
            }
        )
    non_registration = [row for row in blockers if row["code"] != "TAILNET_FIXTURE_UNREGISTERED"]
    if non_registration:
        status = "P9_EXECUTION_ADAPTER_BLOCKED"
    elif blockers:
        status = "P9_READY_FOR_FIXTURE_REGISTRATION"
    else:
        status = "P9_EXECUTION_ADAPTER_READY"
    result = {
        "schema": "beglin-quality-adapter-preflight/1",
        "status": status,
        "execution_allowed": not blockers,
        "adapter_id": adapter["adapter_id"],
        "adapter_sha256": sha256_json(adapter),
        "token_fixture_sha256": sha256_json(fixture),
        "corpus_sha256": fixture["corpus_sha256"],
        "prompt_count": len(fixture["entries"]),
        "repetitions": fixture["repetitions"],
        "required_positions": required_positions,
        "configured_positions": configured_positions,
        "engine_markers": markers,
        "blockers": blockers,
    }
    result["preflight_sha256"] = sha256_json(result)
    return result


def _parse_fields(raw: str) -> tuple[dict[str, str], list[int]]:
    head, separator, tail = raw.partition(" tokens:")
    if not separator:
        raise QualityArtifactError("P9 request line has no token delimiter")
    fields: dict[str, str] = {}
    for part in head.split():
        key, sep, value = part.partition("=")
        if sep:
            fields[key] = value
    try:
        tokens = [int(value) for value in tail.strip().split()] if tail.strip() else []
    except ValueError as exc:
        raise QualityArtifactError("P9 request line contains a non-integer output token") from exc
    return fields, tokens


def parse_engine_log(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        if not line.startswith(REQUEST_PREFIX):
            continue
        fields, tokens = _parse_fields(line[len(REQUEST_PREFIX):])
        required = {
            "req",
            "prompt",
            "context_tokens",
            "finite_logits",
            "nll_sum",
            "nll_count",
            "router_near",
            "router_decisions",
            "kva_mean_abs",
            "kva_rms",
            "shared_up_mean_abs",
            "shared_up_rms",
            "nout",
        }
        missing = sorted(required - set(fields))
        if missing:
            raise QualityArtifactError(f"P9 request line missing fields: {missing}")
        row = {
            "req": int(fields["req"]),
            "prompt": int(fields["prompt"]),
            "context_tokens": int(fields["context_tokens"]),
            "finite_logits": fields["finite_logits"] == "1",
            "nll_sum": float(fields["nll_sum"]),
            "nll_count": int(fields["nll_count"]),
            "router_near": int(fields["router_near"]),
            "router_decisions": int(fields["router_decisions"]),
            "kva_mean_abs": float(fields["kva_mean_abs"]),
            "kva_rms": float(fields["kva_rms"]),
            "shared_up_mean_abs": float(fields["shared_up_mean_abs"]),
            "shared_up_rms": float(fields["shared_up_rms"]),
            "nout": int(fields["nout"]),
            "tokens": tokens,
        }
        if fields["finite_logits"] not in {"0", "1"}:
            raise QualityArtifactError(f"request {row['req']} finite_logits must be 0 or 1")
        integer_fields = ("req", "prompt", "context_tokens", "nll_count", "router_near", "router_decisions", "nout")
        if any(row[name] < 0 for name in integer_fields):
            raise QualityArtifactError(f"request {row['req']} contains a negative count/index")
        float_fields = (
            "nll_sum",
            "kva_mean_abs",
            "kva_rms",
            "shared_up_mean_abs",
            "shared_up_rms",
        )
        if any(not math.isfinite(row[name]) for name in float_fields):
            raise QualityArtifactError(f"request {row['req']} contains NaN/Inf metrics")
        if row["nll_sum"] < 0 or any(row[name] < 0 for name in float_fields[1:]):
            raise QualityArtifactError(f"request {row['req']} contains a negative quality metric")
        if row["router_near"] > row["router_decisions"]:
            raise QualityArtifactError(f"request {row['req']} router near-tie count exceeds decisions")
        if row["nout"] != len(tokens):
            raise QualityArtifactError(f"request {row['req']} nout does not match token count")
        rows.append(row)
    if not rows:
        raise QualityArtifactError("engine log contains no P9_QUALITY_REQUEST_V1 rows")
    if sorted(row["req"] for row in rows) != list(range(len(rows))):
        raise QualityArtifactError("engine request ids are not contiguous from zero")
    return sorted(rows, key=lambda row: row["req"])


def _decoder(tokenizer_json: Path | None):
    if tokenizer_json is None:
        return lambda _: None
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    return lambda values: tokenizer.decode(values, skip_special_tokens=True)


def normalize_run(
    *,
    adapter: dict[str, Any],
    fixture: dict[str, Any],
    rows: list[dict[str, Any]],
    identity: dict[str, Any],
    variant_name: str,
    run_id: str,
    tokenizer_json: Path | None,
    reference_path: Path | None,
) -> dict[str, Any]:
    entries = fixture["entries"]
    expected_rows = len(entries) * fixture["repetitions"]
    if len(rows) != expected_rows:
        raise QualityArtifactError(f"expected {expected_rows} P9 rows, got {len(rows)}")
    if variant_name == "reference":
        variant = {"kind": "reference", "precision": adapter["reference"]["precision"], "n": None}
        policy = adapter["reference"]
        reference = None
        identity = dict(identity)
        identity["policy_hash"] = None
    else:
        match = re.fullmatch(r"n([4-7])", variant_name)
        if not match:
            raise QualityArtifactError("variant must be reference or n4..n7")
        n = int(match.group(1))
        variant = {"kind": "candidate", "precision": variant_name, "n": n}
        policy = next(item for item in adapter["candidates"] if item["n"] == n)
        if reference_path is None:
            raise QualityArtifactError("candidate normalization requires --reference-artifact")
        reference = load_run_artifact(reference_path)
        identity = dict(identity)
        identity["policy_hash"] = sha256_json(
            {"adapter_id": adapter["adapter_id"], "variant": variant_name, "policy": policy}
        )
    identity = dict(identity)
    identity["prompt_corpus_sha256"] = fixture["corpus_sha256"]
    decode = _decoder(tokenizer_json)
    reference_map = {}
    if reference is not None:
        reference_map = {
            (item["prompt_id"], item["repetition"]): item
            for item in reference["requests"]
        }
    requests = []
    promotion_count = len(policy["promotion_lines"])
    for row in rows:
        prompt_index = row["prompt"]
        if prompt_index < 0 or prompt_index >= len(entries):
            raise QualityArtifactError(f"request {row['req']} prompt index out of range")
        expected_prompt_index = row["req"] % len(entries)
        if prompt_index != expected_prompt_index:
            raise QualityArtifactError(
                f"request {row['req']} prompt index {prompt_index} != expected {expected_prompt_index}"
            )
        entry = entries[prompt_index]
        repetition = row["req"] // len(entries)
        if row["context_tokens"] != entry["context_tokens"]:
            raise QualityArtifactError(f"request {row['req']} context length differs from fixture")
        request = {
            "request_id": f"{run_id}:{entry['prompt_id']}:{repetition}",
            "prompt_id": entry["prompt_id"],
            "repetition": repetition,
            "context_tokens": row["context_tokens"],
            "output_token_ids": row["tokens"],
            "output_text": decode(row["tokens"]),
            "finite_logits": row["finite_logits"],
            "token_logprobs": None,
            "nll_sum": row["nll_sum"] if row["nll_count"] else None,
            "nll_token_count": row["nll_count"] or None,
            "corrections": 0,
            "promotions": promotion_count if row["req"] == 0 else 0,
            "activation_summaries": [
                {
                    "role": "kv_a_proj_with_mqa",
                    "layer": 11,
                    "expert_id": None,
                    "mean_abs": row["kva_mean_abs"],
                    "rms": row["kva_rms"],
                },
                {
                    "role": "shared_up_proj",
                    "layer": 3,
                    "expert_id": None,
                    "mean_abs": row["shared_up_mean_abs"],
                    "rms": row["shared_up_rms"],
                },
            ],
            "router": {
                "near_tie_count": row["router_near"],
                "decision_count": row["router_decisions"],
                "near_tie_rate": None,
            },
            "quality_score": 1.0,
        }
        if reference is not None:
            key = (entry["prompt_id"], repetition)
            if key not in reference_map:
                raise QualityArtifactError(f"reference artifact missing {key}")
            request["quality_score"] = score_request(reference_map[key], request)
        requests.append(request)
    return validate_run_artifact(
        {
            "schema": "beglin-quality-run/1",
            "run_id": run_id,
            "variant": variant,
            "identity": identity,
            "requests": requests,
        }
    )


def _write_or_print(value: Any, output: Path | None) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def main() -> None:
    repo_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="P9 XOX execution adapter")
    sub = parser.add_subparsers(dest="command", required=True)

    preflight_parser = sub.add_parser("preflight")
    preflight_parser.add_argument("--repo", type=Path, default=repo_default)
    preflight_parser.add_argument(
        "--adapter", type=Path, default=repo_default / "configs/quality/p9_execution_adapter_v1.json"
    )
    preflight_parser.add_argument(
        "--corpus", type=Path, default=repo_default / "configs/quality/deterministic_corpus_v1.json"
    )
    preflight_parser.add_argument("--output", type=Path)

    materialize_parser = sub.add_parser("materialize")
    materialize_parser.add_argument(
        "--fixture", type=Path, default=repo_default / "configs/quality/p9_token_fixture_v1.json"
    )
    materialize_parser.add_argument("--output-dir", type=Path, required=True)
    materialize_parser.add_argument("--output", type=Path)

    normalize_parser = sub.add_parser("normalize")
    normalize_parser.add_argument(
        "--adapter", type=Path, default=repo_default / "configs/quality/p9_execution_adapter_v1.json"
    )
    normalize_parser.add_argument(
        "--fixture", type=Path, default=repo_default / "configs/quality/p9_token_fixture_v1.json"
    )
    normalize_parser.add_argument("--log", type=Path, required=True)
    normalize_parser.add_argument("--identity", type=Path, required=True)
    normalize_parser.add_argument("--variant", required=True)
    normalize_parser.add_argument("--run-id", required=True)
    normalize_parser.add_argument("--tokenizer-json", type=Path)
    normalize_parser.add_argument("--reference-artifact", type=Path)
    normalize_parser.add_argument("--output", type=Path)

    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.repo, args.adapter, args.corpus)
        _write_or_print(result, args.output)
        return
    if args.command == "materialize":
        result = materialize_token_fixture(load_token_fixture(args.fixture), args.output_dir)
        _write_or_print(result, args.output)
        return
    adapter = load_adapter(args.adapter)
    fixture = load_token_fixture(args.fixture)
    rows = parse_engine_log(args.log.read_text(encoding="utf-8"))
    result = normalize_run(
        adapter=adapter,
        fixture=fixture,
        rows=rows,
        identity=_read_json(args.identity),
        variant_name=args.variant,
        run_id=args.run_id,
        tokenizer_json=args.tokenizer_json,
        reference_path=args.reference_artifact,
    )
    _write_or_print(result, args.output)


if __name__ == "__main__":
    main()
