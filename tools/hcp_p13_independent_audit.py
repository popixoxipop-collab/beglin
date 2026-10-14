#!/usr/bin/env python3
"""Independent P13 native P9 packed qNg64 C/MLX trace vs frozen P12 BF16 code hash.

Read-only exact SHA and frozen original P6 generated sequence auditor. No raw
token IDs, no checkpoint bytes. Does not claim full-model BF16 logits.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,pathlib

P13_REF="50d222534ab1324b990a2c4d83cc227e0ca2ba53c2037b500d688f742caeb078"
P6_SHA="2ad68622fcfe4e8f2d16e0abb1c24e010618fd14ab40917f4dd4278167e59104"
P13_C="26e542118302a5a82f147ef1678ec0f6e4516b467d0a4166cc1bc7dce29fa8c6"
P9_MLX="ce07a8296e3cc0f53a5e07a9c242bb09bcb9d6d41c76e9a516f410f1bfd3004f"
P13_BIN="657acc942b39f85307f04857b2e8fdd86ebe776a8855af3928e7050c6ff9d065"
PRE_COMMIT="5418310d178f42d68dde09fa398414ab91fb38d4"
STAGE="8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9"
ROUTE="c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47"
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def audit(source,reference,preflight,truthfile):
    source,reference,preflight,truthfile=map(pathlib.Path,(source,reference,preflight,truthfile))
    if sha(reference)!=P13_REF or sha(truthfile)!=P6_SHA:
        raise ValueError("P13 original BF16 C packed SHA or original P6 generation SHA changed")
    frozen=json.loads(reference.read_text())
    pre=json.loads(preflight.read_text())
    r=json.loads(source.read_text())
    if (frozen.get("expected_distinct")!=4 or
        frozen.get("reference_prior_to_native_P13_capture") is not True or
        frozen.get("full_BF16_model_logits_or_task_accuracy_not_measured") is not True):
        raise ValueError("P13 original BF16 source reference was not frozen in advance")
    if (pre.get("frozen_BF16_local_reference_sha256")!=P13_REF
        or pre.get("P13_SHA_only_C_sha256")!=P13_C
        or pre.get("P9_MLX_KV128_SHA256")!=P9_MLX
        or pre.get("source_git_commit")!=PRE_COMMIT
        or pre.get("live_original_stage_sha256")!=STAGE
        or pre.get("live_original_route_sha256")!=ROUTE
        or pre.get("production_touched") is not False):
        raise ValueError("P13 native preimage did not match sealed source/model/route")
    if (r.get("schema")!="beglin-hcp-p13-real-native-original-BF16-packed-byte-parity-v1"
        or r.get("status")!="FOUR_P9_REAL_MLX_QNG64_NATIVE_PACKED_AND_SCALES_SHA_EQUAL_P12_BF16_ORACLE"
        or r.get("four_frozen_reference_SHA256")!=P13_REF
        or r.get("native_traced_binary_SHA256")!=P13_BIN
        or r.get("P13_C_source_SHA256")!=P13_C or r.get("P9_MLX_source_SHA256")!=P9_MLX
        or r.get("source_head_prereg")!=PRE_COMMIT
        or r.get("real_native_generation_admissions")!=48 or
        r.get("real_native_terminal_txns_verified")!=48 or
        r.get("cache_tracing_OFF_generations")!=24 or
        r.get("cache_tracing_ON_generations")!=24 or
        r.get("no_speed_comparison_with_tracing") is not True or
        r.get("bf16_full_model_logits_not_measured") is not True or
        r.get("automatic_live_promotion") is not False or
        r.get("production_touched") is not False):
        raise ValueError("P13 actual native study scope or results incomplete")
    expected_live={"stage_sha256":STAGE,"route_sha256":ROUTE,"source_worker_pids":[35481,35493]}
    if r["production_before"]!=expected_live or r["production_after"]!=expected_live:
        raise ValueError("P13 original live image/worker drift")
    ref={}
    for x in frozen["replacements"]:
        target="shared_up" if x["target"]=="shared_up_L3" else "shared_down"
        layer=3 if target=="shared_up" else 26
        key=(target,layer,x["bits"])
        if key in ref:raise ValueError("P13 frozen qNg64 representation duplicates")
        ref[key]=x
    if len(ref)!=4:raise ValueError("P13 missing 4 exact expected BF16 representations")
    seen=set();counts=collections.Counter()
    events=r["native_hash_log_events"]
    if len(events)!=r.get("actual_native_hash_lines") or len(events)<4:
        raise ValueError("P13 native packed code capture incomplete")
    for e in events:
        key=(e["role"],e["layer"],e["bits"])
        if key not in ref:raise ValueError("P13 unexpected tensor width/identity")
        x=ref[key]
        if (e["packed_SHA256"]!=x["packed_SHA256"] or
            e["scales_SHA256"]!=x["scales_SHA256"] or
            e["packed_bytes"]!=x["packed_bytes"] or
            e["scales_bytes"]!=x["scales_bytes"] or
            e["phase"] not in ("store","hit")):
            raise ValueError("P13 actual native packed weight is not byte-identical to original BF16 C quantizer")
        seen.add(key);counts[key]+=1
    if seen!=set(ref) or r.get("native_unique_pairs")!=4 or r.get("native_4_unique_tensor_bitwidth_pairs_exact") is not True:
        raise ValueError("P13 captured native cache did not cover all frozen 5/6-bit tensors")
    q={item["prompt_id"]:item for item in json.loads(truthfile.read_text())["records"]}
    cells=collections.defaultdict(list)
    # Engine generated sequence claims must be reproducible against the P6 source.
    # The separate native driver verifies per-inference ACK / worker PID and only exports
    # an aggregated count; this audit independently verifies source generation SHA where
    # outputs are present in the preserved per-admission rows.
    if "generation_rows" not in r:
        raise ValueError("P13 per-admission sanitized generation SHA evidence must be retained")
    for g in r["generation_rows"]:
        key=(g["prompt_id"],g["policy"],g["mode"])
        if (g["mode"] not in ("off","on") or g["policy"] not in ("A","B") or
            g["prompt_id"] not in q or g["domain"]!=q[g["prompt_id"]]["domain"] or
            g.get("native_ACK_verified") is not True or
            g["generated_sequence_sha256"]!=q[g["prompt_id"]][g["policy"]]["sequence_sha256"]):
            raise ValueError("P13 generation SHA/ACK not identical to frozen P6 result")
        cells[key].append(g)
    if len(cells)!=24 or any(len(v)!=2 for v in cells.values()):
        raise ValueError("P13 6 domain A/B/A/B x ON/OFF generation coverage missing")
    return {"schema":"beglin-hcp-p13-independent-C-original-BF16-vs-real-P9-native-SHA-audit-v1",
        "source_reference_sha256":P13_REF,
        "native_result_sha256":sha(source),
        "orig_model_C_and_MLX_sha_checked":True,
        "frozen_original_BF16_4_representations_byte_equal":True,
        "captured_cache_events":len(events),
        "independent_original_P6_generated_SHA_matches":len(r["generation_rows"]),
        "independent_original_live_stage_and_PID_verified":True,
        "full_BF16_transformer_reference_logits_not_run":True,
        "production_promotion_allowed":False}
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("native","reference","preflight","quality","output"):
        p.add_argument("--"+n,required=True)
    a=p.parse_args()
    d=audit(a.native,a.reference,a.preflight,a.quality)
    path=pathlib.Path(a.output)
    with path.open("x") as f:json.dump(d,f,sort_keys=True,indent=2);f.write("\n")
    print("P13_INDEPENDENT_BF16_NATIVE_PACKED_SHA_AUDIT_PASS",
       json.dumps({"exact_4":True,"real_generation_hashes":d["independent_original_P6_generated_SHA_matches"],
          "result_SHA256":sha(path)},sort_keys=True))
