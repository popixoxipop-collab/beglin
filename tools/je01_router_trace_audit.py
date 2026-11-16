#!/usr/bin/env python3
"""JE01 fail-closed independent audit of actual routed MoE traces.

Pure proof-consumer, not an instrumentation engine: the trusted isolated
native backend must export pre-top-k 64-way router probabilities. Eager
mlx_gpu_layer_step currently exposes selected IDs/weights only; a separate
instrumented build is required for full scores and top-k boundary checks.
No inference, weights, loss labels or production changes happen here.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

SCHEMA = "BEGLIN-JE01-router-topk-trace-audit/1"
HEX = re.compile(r"^[0-9a-f]{64}$")
ROUTED_EXPERTS = 64
TOP_K = 6
CALIBRATION_PROMPTS = 256
MIN_INDEPENDENT_PROMPTS = 32
PILOT_SLOTS = 8
REQUIRED_FIELDS = frozenset(("sample_sha256", "layer", "token_pos",
                             "router_probs64", "selected_expert_ids"))


def _canonical(obj):
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _pin(sha, tag):
    if not isinstance(sha, str) or not HEX.fullmatch(sha):
        raise ValueError("JE01_" + tag + "_UNPINNED")


def _validate(events, *, checkpoint_sha256, dataset_sha256,
              expected_sample_sha256):
    _pin(checkpoint_sha256, "CHECKPOINT")
    _pin(dataset_sha256, "DATASET")
    if not isinstance(expected_sample_sha256, (list, tuple)) or len(
        expected_sample_sha256
    ) != CALIBRATION_PROMPTS:
        raise ValueError("JE01_FROZEN_CALIBRATION_256_REQUIRED")
    if not all(isinstance(x, str) and HEX.fullmatch(x)
               for x in expected_sample_sha256):
        raise ValueError("JE01_SAMPLE_SHA_INVALID")
    ids = set(expected_sample_sha256)
    if len(ids) != CALIBRATION_PROMPTS:
        raise ValueError("JE01_SAMPLE_DUPLICATE")
    if not isinstance(events, list) or not events:
        raise ValueError("JE01_TRACES_MISSING")
    seen_events = set()
    counted_samples = set()
    from collections import defaultdict
    routed = defaultdict(set)
    minimum_margin = float("inf")
    for record in events:
        if not isinstance(record, dict) or set(record) != REQUIRED_FIELDS:
            raise ValueError("JE01_ROUTER_TRACE_FIELDS_OR_OUTCOME_LEAKAGE")
        sample, layer, pos = (
            record["sample_sha256"], record["layer"], record["token_pos"])
        if sample not in ids:
            raise ValueError("JE01_UNFROZEN_SAMPLE_ROUTE")
        if type(layer) is not int or layer < 0 or type(pos) is not int or pos < 0:
            raise ValueError("JE01_ROUTER_LAYER_OR_POS")
        key = (sample, layer, pos)
        if key in seen_events:
            raise ValueError("JE01_DUPLICATE_ROUTER_EVENT")
        seen_events.add(key)
        counted_samples.add(sample)
        probs = record["router_probs64"]
        selected = record["selected_expert_ids"]
        if (not isinstance(probs, list) or len(probs) != ROUTED_EXPERTS or
            any(type(p) not in (int, float) or not math.isfinite(p) or p < 0
                for p in probs) or abs(sum(probs) - 1) > 1e-5):
            raise ValueError("JE01_ROUTER_SOFTMAX_INVALID")
        if (not isinstance(selected, list) or len(selected) != TOP_K or
            any(type(e) is not int or not 0 <= e < ROUTED_EXPERTS
                for e in selected) or len(set(selected)) != TOP_K):
            raise ValueError("JE01_ROUTER_TOPK_ID_INVALID")
        actual = sorted(range(ROUTED_EXPERTS),
                        key=lambda e: (-probs[e], e))[:TOP_K]
        ranked = sorted(probs, reverse=True)
        boundary = ranked[TOP_K-1] - ranked[TOP_K]
        if boundary <= 1e-12:
            raise ValueError("JE01_ROUTER_TOPK_BOUNDARY_TIE")
        if set(selected) != set(actual):
            raise ValueError("JE01_ROUTER_TOPK_MISMATCH")
        minimum_margin = min(minimum_margin, boundary)
        for expert in selected:
            routed[(layer, expert)].add(sample)
    if counted_samples != ids:
        raise ValueError("JE01_ROUTER_COVERAGE_INCOMPLETE")
    summary = sorted(({"layer": layer, "expert_id": expert,
                       "distinct_calibration_prompts": len(samples)}
                      for (layer, expert), samples in routed.items()),
                     key=lambda row: (-row["distinct_calibration_prompts"],
                                      row["layer"], row["expert_id"]))
    return {"all": summary, "minimum_margin": minimum_margin,
            "total_router_events": len(seen_events),
            "calibration_samples": CALIBRATION_PROMPTS}


def freeze_pilot_slots(events, *, checkpoint_sha256, dataset_sha256,
                       expected_sample_sha256):
    summary = _validate(events, checkpoint_sha256=checkpoint_sha256,
                        dataset_sha256=dataset_sha256,
                        expected_sample_sha256=expected_sample_sha256)
    eligible = [x for x in summary["all"]
                if x["distinct_calibration_prompts"] >= MIN_INDEPENDENT_PROMPTS]
    if len(eligible) < PILOT_SLOTS:
        raise ValueError("JE01_INSUFFICIENT_ROUTED_EXPERTS_WITH_32_PROMPTS")
    selected = eligible[:PILOT_SLOTS]
    if len({(x["layer"], x["expert_id"]) for x in selected}) != PILOT_SLOTS:
        raise ValueError("JE01_DUPLICATE_PILOT_EXPERT")
    state = {
        "schema": SCHEMA,
        "stage": "FROZEN_SELECTION_FROM_LABEL_FREE_ROUTER_INPUTS",
        "checkpoint_sha256": checkpoint_sha256,
        "data_manifest_sha256": dataset_sha256,
        "sample_id_manifest_sha256": _canonical(sorted(expected_sample_sha256)),
        "total_router_events": summary["total_router_events"],
        "total_unique_calibration_prompts": summary["calibration_samples"],
        "minimum_topk_boundary_gap": summary["minimum_margin"],
        "selected_slots": selected,
        "gold_token_nll_or_labels_used": False,
        "external_original_native_provenance_verified": False,
        "production_promotion_authorized": False,
    }
    state["state_sha256"] = _canonical(state)
    return state


def compare_frozen_router_traces(baseline_events, alternative_events, *,
                                expected_sample_sha256):
    """Hard zero-flip requirement; strict event identity, not a best-effort join."""
    ids = set(expected_sample_sha256)
    if not ids or any(not HEX.fullmatch(x) for x in ids):
        raise ValueError("JE01_FROZEN_SAMPLE_IDS_REQUIRED")
    def collect(events):
        result = {}
        if not isinstance(events, list):
            raise ValueError("JE01_TRACES_REQUIRED")
        for event in events:
            if not isinstance(event, dict) or set(event) != REQUIRED_FIELDS:
                raise ValueError("JE01_ROUTE_EVENT_FIELDS")
            sample = event["sample_sha256"]
            if sample not in ids:
                raise ValueError("JE01_ROUTER_UNFROZEN_SAMPLE")
            key = (sample, event["layer"], event["token_pos"])
            if key in result:
                raise ValueError("JE01_ROUTER_EVENT_DUPLICATE")
            choices = event["selected_expert_ids"]
            if not isinstance(choices, list) or len(choices) != TOP_K or len(
                set(choices)
            ) != TOP_K or any(type(c) is not int or not 0 <= c < 64 for c in choices):
                raise ValueError("JE01_INVALID_ROUTER_CHOICES")
            result[key] = frozenset(choices)
        return result
    base, alt = collect(baseline_events), collect(alternative_events)
    if set(base) != set(alt):
        raise ValueError("JE01_ROUTING_EVENT_ALIGNMENT_FAILED")
    flips = sum(base[k] != alt[k] for k in base)
    return {"schema": SCHEMA, "event_pairs": len(base),
            "topk_flip_count": flips, "no_observed_flip": flips == 0,
            "production_promotion_authorized": False}
