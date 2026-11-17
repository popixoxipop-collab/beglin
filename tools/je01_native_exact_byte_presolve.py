#!/usr/bin/env python3
"""JE01 source-locked AF8 codec-byte PRE-SOLVE, not native weight evidence.

Only the original q4 AF payload byte counts are physically measured. n=2/6
qNg64 (bit-planes + f32 scale), n=8 q8g64 (signed int8 + f32 scale), and
unchanged native q4 AF (packed + f32 scale + f32 bias) are PROJECTED from
their existing codec contracts. No BF16 tensor, packed alternative, GPU NLL,
or selected Bayesian plan is created here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

SOURCE_RECEIPT_SHA = "f47bbccbc3878a70dccb9f4b7f5f7f5bd43087aeec58c948924ef85bb428ca4c"
EXPECTED_BASELINE_BYTES = 43_253_760
TIERS = (2, 4, 6, 8)
ROLE_NAMES = ("gate_proj", "up_proj", "down_proj")
DEFAULT_LEDGER = (
    Path(__file__).resolve().parents[1]
    / "docs/research/JE01_ORIGINAL_AF_EIGHT_EXPERT_REAL_BASELINE_PACKED_BYTE_LEDGER_20261118.json"
)
SCHEMA = "beglin-je01-af8-native-codec-byte-presolve-only/1"


def canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_ledger(d):
    if d.get("schema") != "beglin-je01-actual-original-AF-eight-expert-exact-baseline-bytes/1":
        raise ValueError("JE01_SOURCE_LEDGER_SCHEMA_DRIFT")
    if d.get("status") != "EIGHT_SELECTED_ORIGINAL_AF_BASELINE_BYTES_AUDITED":
        raise ValueError("JE01_SOURCE_LEDGER_STATUS_DRIFT")
    if d.get("real_256_selected_expert_baseline_eight_bytes") != EXPECTED_BASELINE_BYTES:
        raise ValueError("JE01_SOURCE_BASELINE_BYTES_DRIFT")
    if d.get("real_original_baseline_bit_width_per_selected_expert") != 4:
        raise ValueError("JE01_SOURCE_BASELINE_BITS_DRIFT")
    slots = d.get("selected_slots")
    if not isinstance(slots, list) or len(slots) != 8:
        raise ValueError("JE01_SOURCE_EIGHT_SLOTS_MISSING")
    seen = set()
    for s in slots:
        identity = (s.get("layer"), s.get("expert"))
        if (identity in seen or type(identity[0]) is not int or type(identity[1]) is not int
                or not 1 <= identity[0] <= 26 or not 0 <= identity[1] < 64):
            raise ValueError("JE01_SOURCE_SLOT_ID_INVALID")
        seen.add(identity)
        if s.get("baseline_assumed_native_AF_bits") != 4:
            raise ValueError("JE01_SOURCE_SLOT_BASELINE_BITS")
        if s.get("calibration_last_token_routed_calls", 0) < 32:
            raise ValueError("JE01_SOURCE_ROUTING_COUNT")
        roles = s.get("roles")
        if not isinstance(roles, list) or tuple(r.get("role") for r in roles) != ROLE_NAMES:
            raise ValueError("JE01_SOURCE_ROLES_INVALID")
        for r in roles:
            p = r.get("packed_weight_bytes")
            sc = r.get("individual_expert_scale_f32_bytes")
            bi = r.get("individual_expert_bias_f32_bytes")
            if (type(p) is not int or type(sc) is not int or type(bi) is not int
                    or (p, sc, bi) != (1_441_792, 180_224, 180_224)
                    or r.get("individual_expert_bits") != 4
                    or r.get("total_expert_role_bytes") != p + sc + bi):
                raise ValueError("JE01_SOURCE_ROLE_BYTE_LAYOUT_DRIFT")
            intervals = r.get("source_Af_global_byte_regions", {})
            for key, count in (("packed", p), ("scale", sc), ("bias", bi)):
                span = intervals.get(key)
                if (not isinstance(span, list) or len(span) != 2
                        or type(span[0]) is not int or type(span[1]) is not int
                        or span[1] - span[0] != count
                        or not 0 <= span[0] < span[1] <= d["source_original_AF_total_bytes"]):
                    raise ValueError("JE01_SOURCE_REGION_BYTES_DRIFT")
        if sum(r["total_expert_role_bytes"] for r in roles) != s.get("actual_packed_baseline_bytes"):
            raise ValueError("JE01_SOURCE_SLOT_BYTES_DRIFT")
    if sum(s["actual_packed_baseline_bytes"] for s in slots) != EXPECTED_BASELINE_BYTES:
        raise ValueError("JE01_SOURCE_TOTAL_BYTE_MISMATCH")
    return sorted(slots, key=lambda s: (s["layer"], s["expert"]))


def read_verified_ledger(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("JE01_SOURCE_LEDGER_NOT_OBJECT")
    asserted = record.pop("receipt_sha256", None)
    if asserted != SOURCE_RECEIPT_SHA or canonical(record) != SOURCE_RECEIPT_SHA:
        raise ValueError("JE01_SOURCE_LEDGER_SHA_MISMATCH")
    validate_ledger(record)
    return record


def projected_slot_bytes(slot, bits):
    if type(bits) is not int or bits not in TIERS:
        raise ValueError("JE01_UNSUPPORTED_BIT_TIER")
    if bits == 4:
        return slot["actual_packed_baseline_bytes"]
    # qNg64 n=2/6 and q8g64 n=8 retain one f32 scale per group, NO
    # native q4 affine f32 bias array. Existing GPU binder confirms
    # q8g64 signed int8 and refuses qNg64 bit-plane n=8.
    value = 0
    for role in slot["roles"]:
        packed4 = role["packed_weight_bytes"]
        assert (packed4 * bits) % 4 == 0
        value += packed4 * bits // 4 + role["individual_expert_scale_f32_bytes"]
    return value


def presolve(ledger):
    slots = validate_ledger(ledger)
    base = sum(projected_slot_bytes(s, 4) for s in slots)
    if base != EXPECTED_BASELINE_BYTES:
        raise ValueError("JE01_BASELINE_PROJECTION_DRIFT")
    costs = [tuple(projected_slot_bytes(s, b) for b in TIERS) for s in slots]
    group_counts = Counter()
    signatures = []
    for choice in itertools.product(TIERS, repeat=8):
        total = sum(costs[i][TIERS.index(b)] for i, b in enumerate(choice))
        if total != base:
            continue
        counts = tuple(choice.count(b) for b in TIERS)
        group_counts[counts] += 1
        signatures.append(",".join(
            "%d:%d=%d" % (s["layer"], s["expert"], b)
            for s, b in zip(slots, choice)
        ))
    patterns = {
        "2:%d,4:%d,6:%d,8:%d" % counts: count
        for counts, count in sorted(group_counts.items())
    }
    return {
        "schema": SCHEMA,
        "status": "CODEC_FORMULA_PRESOLVE_ONLY_NO_ALTERNATE_WEIGHT_FILES",
        "original_AF_source_ledger_receipt_sha256": SOURCE_RECEIPT_SHA,
        "baseline_actually_audited_AF_bytes": base,
        "bits_per_slot": list(TIERS),
        "projected_bytes_one_expert_gate_up_down": {
            str(bits): projected_slot_bytes(slots[0], bits) for bits in TIERS
        },
        "projected_equal_byte_catalog_count": len(signatures),
        "projected_equal_byte_nonbaseline_count": len(signatures) - 1,
        "projected_pattern_counts": patterns,
        "projected_catalog_sha256": hashlib.sha256(
            ("\n".join(signatures) + "\n").encode()
        ).hexdigest(),
        "source_bf16_to_original_AF_identity_parity": "NOT_TESTED",
        "real_2_6_8_packed_weight_artifact_count": 0,
        "real_alternative_actual_file_byte_budget_audit": "NOT_RUN",
        "real_expert_precision_paired_NLL": "NOT_MEASURED",
        "heldout_and_ood_labels_accessed": False,
        "bayesian_plan_eligible": False,
        "production_promotion_allowed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    opts = parser.parse_args()
    try:
        data = presolve(read_verified_ledger(opts.ledger))
        data["receipt_sha256"] = canonical(data)
        print(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError, AssertionError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)[:240]}))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
