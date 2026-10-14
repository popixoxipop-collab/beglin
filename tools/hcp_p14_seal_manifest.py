#!/usr/bin/env python3
"""Seal scoped P14 sanitized evidence + immutable original XOX archive pointer."""
import hashlib,json,pathlib,subprocess

ROOT=pathlib.Path("/Users/eoe/mcp-sandbox/tailnet-commander/hcp-p14-multiworker-bf16-20261014/repository")
E=ROOT/"experiments/hcp_p14/evidence/native_xox_20261014"
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def main():
    if not E.is_dir() or (E/"MANIFEST.json").exists():
        raise RuntimeError("P14 manifest must be written only once")
    if subprocess.check_output(["/usr/bin/git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()!="93ac73bb6feefe4bf020b7882fbc0bf3d8d977e7":
        raise RuntimeError("P14 source branch changed after preregistration")
    expected={
        "P14_AUDIT_SUMMARY.json":"236475031894ea9ef1df87ff7c9a46a36f770fa89e0cc15373254927aedee470",
        "P14_EOE_AUDIT.json":"d76f65e73bf816c368446ebfbbc065caa8da4400eccdb15b792b7b5a896c7bb5",
        "BF16_TEACHER_FEASIBILITY.json":"14ca8badf6221b4d144049a74203bb6b8d4cacd047b861a1922ad3434abac9ec",
        "P14_REGRESSION_RECEIPT.json":"02bd15f6cf6f26f39d5966f5cf8d8eb88f19820aa22096ec290f15cbc216c82b"}
    for file,s in expected.items():
        if sha(E/file)!=s:raise RuntimeError("P14 evidence not immutable: "+file)
    regress=json.loads((E/"P14_REGRESSION_RECEIPT.json").read_text())
    assert regress["tests_passed"]==213 and regress["tests_failed"]==0
    report=json.loads((E/"P14_EOE_AUDIT.json").read_text())
    assert report["true_dual_native_responses_passed_in_original_XOX_audit"]==48
    assert report["original_XOX_time_overlap_pairs"]==24
    files={}
    for file in E.rglob("*"):
        if file.is_dir():continue
        if not file.is_file() or file.is_symlink():
            raise RuntimeError("P14 EOE evidence not a regular file")
        files[str(file.relative_to(E))]={"sha256":sha(file),"bytes":file.stat().st_size}
    source={p.name:sha(p) for p in sorted((ROOT/"tools").glob("hcp_p14_*.py"))}
    handoff=ROOT/"docs/BEGLIN_HCP_P14_MULTIWORKER_BF16_2026-10-14.md"
    prereg=ROOT/"docs/BEGLIN_HCP_P14_PREREG_2026-10-14.md"
    if sha(prereg)!="2da76c7d6164a104a96bad4af23d2620f993ae3632bbaaaf674789b292ffa207":
        raise ValueError("P14 original preregistration changed")
    manifest={
        "schema":"beglin-hcp-p14-original-native-worker-and-BF16-envelope-final-manifest-v1",
        "preregister_commit":"93ac73bb6feefe4bf020b7882fbc0bf3d8d977e7",
        "P13_parent_commit":"dc4d2dbc999633e75ff5ff0a430642ceefc830b3",
        "preregister_SHA256":sha(prereg),
        "final_handoff_SHA256":sha(handoff),
        "original_model_index_SHA256":"d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf",
        "P9_native_GPU_binary_SHA256":"103549b7f7f4a3d142be883013956b4f1437759c8be76bff31f2451956df37fd",
        "P11_fault_only_native_binary_SHA256":"a5fd3f590ad6bbddcb69e690ccf738b711b8f7ceb3319f99ee933123f5ac7e78",
        "two_independent_real_GPU_worker_processes":True,
        "simultaneous_native_real_admissions":48,
        "overlapped_admission_pairs":24,
        "native_survivor_new_admissions_after_peer_fault":6,
        "peer_original_ambiguous_transaction_status":"UNKNOWN_NOT_RETRIED",
        "peer_new_B_A_B_admissions":3,
        "actual_peak_two_GPU_worker_RSS_GiB":11.71207,
        "P14_XOX_raw_original_response_JSON_SHA256":"98aa76b19129a7634061595e98463267f506435069c3714db7f2b0a3ff7e9a71",
        "P14_XOX_original_forensic_archive_directory":"/Users/xox/mcp-sandbox/tailnet-commander/hcp-p14-native-forensic-20261014",
        "P14_XOX_original_forensic_manifest_SHA256":"b425aa8ba5ce7a2939cc8c2ec508024f020e1f0c127eb2afdab1ec4c5c736976",
        "original_response_level_independent_audit_performed_on_XOX":True,
        "original_response_level_independent_audit_reperformed_on_EOE":False,
        "BF16_original_layer_count":27,
        "BF16_original_tensor_header_count":5291,
        "BF16_biggest_layer_weights_GiB":1.0894,
        "BF16_full_teacher_forward_executed":False,
        "BF16_full_vocab_logits_measured":False,
        "quality_noninferiority_certified":False,
        "P8_to_P14_regression_tests_PASSED":213,
        "P8_to_P14_regression_tests_FAILED":0,
        "scripts_SHA256":source,
        "EOE_sanitized_evidence_SHA256":files,
        "original_stage_SHA256":"8daf7c2b7f22ab0321131d67ede9c74b423fa305132bf8071243d41285f68fd9",
        "original_route_SHA256":"c1b3400cfbd75893191debe6eaf388afc3e26a9866c1f4d43f3778906df44c47",
        "original_worker_PIDs":[35481,35493],
        "production_changed":False,
        "main_merged":False,
        "automatic_live_promotion":False
    }
    p=E/"MANIFEST.json"
    with p.open("x") as f:json.dump(manifest,f,sort_keys=True,indent=2);f.write("\n")
    assert json.loads(p.read_text())["P8_to_P14_regression_tests_PASSED"]==213
    print("P14_INDEPENDENT_MANIFEST_SEALED",json.dumps({
        "manifest_sha256":sha(p),
        "handoff_sha256":sha(handoff),
        "evidence_files":len(files),"source_scripts":len(source),
        "regressions":213,"full_BF16_teacher":False},sort_keys=True))

if __name__=="__main__":main()
